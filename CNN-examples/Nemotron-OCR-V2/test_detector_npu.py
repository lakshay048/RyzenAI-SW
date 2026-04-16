import os
import argparse
from typing import Optional, Tuple
import numpy as np
import onnxruntime as ort
from pathlib import Path
import time
import subprocess
import platform
import cv2

from utils.preprocessing import preprocess_image
from utils.postprocessing import (
    extract_detections,
    non_maximum_suppression,
    scale_and_format,
    draw_boxes
)

os.environ["XLNX_ENABLE_CACHE"] = "1"
os.environ["PATH"] += (
    os.pathsep + f"{os.environ['CONDA_PREFIX']}\\Lib\\site-packages\\flexmlrt\\lib"
)

XRT_SMI_PATH = "C:\\Windows\\System32\\AMD\\xrt-smi.exe"


def configure_npu_power(p_mode: Optional[str] = None) -> Tuple[int, str, str]:
    """
    Configures the NPU power state using xrt-smi.exe.

    Args:
        p_mode (string, optional): The desired power mode (p-mode).
            If None, displays current status.
            Refer to xrt-smi documentation for valid p-modes.
    Returns:
        tuple: (return_code, stdout, stderr) from the subprocess call.
               return_code is an integer, stdout and stderr are strings.
    Raises:
        OSError: If xrt-smi.exe is not found.
    """

    if platform.system() != "Windows":
        return (-1, "xrt-smi.exe is only available on Windows.", "")

    try:
        if p_mode is not None:
            command = [XRT_SMI_PATH, "configure", "--pmode", str(p_mode)]
        else:
            command = [
                XRT_SMI_PATH,
                "examine",
                "--report",
                "platform",
            ]

        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        stdout, stderr = process.communicate()
        return_code = process.returncode

        if return_code != 0:
            print(f"Error executing xrt-smi.exe: {stderr}")

        return return_code, stdout, stderr

    except FileNotFoundError:
        raise OSError("xrt-smi.exe not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return -1, "", str(e)


def get_cache_key_from_model_path(model_path):
    model_name = Path(model_path).stem
    return f"modelcachekey_{model_name}_bf16"

def main(
    model_file: str,
    image_path: str,
    vaip_config: str,
    cache_dir: str = "./cache",
    pmode: str = "performance",
    visualize: bool = False,
    detection_threshold: float = 0.7,
    npu_threshold: Optional[float] = None,
    output_dir: str = "./detection_results",
):
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model file not found: {model_file}")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    if not os.path.exists(vaip_config):
        raise FileNotFoundError(f"VAIP config file not found: {vaip_config}")

    cache_key = get_cache_key_from_model_path(model_file)
    print(f"Using cache directory: {cache_dir}")
    print(f"Cache key: {cache_key}")
    print(f"Flow type: BF16")

    import onnx
    model = onnx.load(model_file)
    input_shape = [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]
    if len(input_shape) == 4:
        if input_shape[1] == 3 or input_shape[1] < input_shape[2]:
            input_format = 'NCHW'
        elif input_shape[3] == 3 or input_shape[3] < input_shape[2]:
            input_format = 'NHWC'
        else:
            input_format = 'NCHW'
    else:
        input_format = 'NCHW' 

    print(f"\nModel input shape: {input_shape}")
    print(f"Detected input format: {input_format}")

    print(f"\nPreprocessing image: {image_path}")
    onnx_input, original_img, original_size = preprocess_image(image_path, output_format=input_format)

    # Create CPU session
    print("\nCreating CPU session...")
    cpu_session = ort.InferenceSession(
        model_file,
        providers=["CPUExecutionProvider"],
    )

    input_name = cpu_session.get_inputs()[0].name
    print(f"Model input name: {input_name}")

    # Configure NPU power mode
    print(f"\nConfiguring NPU power mode to: {pmode}")
    ret_code, stdout, stderr = configure_npu_power(pmode)
    print(stdout)
    if ret_code != 0:
        print("Error configuring NPU power mode.")
        print(stderr)

    # Create NPU session
    print("\nCreating NPU session (this may take a while on first run)...")
    npu_provider_options = {
        "config_file": vaip_config,
        "cacheDir": cache_dir,
        "cacheKey": cache_key,
    }

    npu_session = ort.InferenceSession(
        model_file,
        providers=["VitisAIExecutionProvider"],
        provider_options=[npu_provider_options],
    )

    # Run inference on CPU
    print("\nRunning CPU inference...")
    cpu_start = time.time()
    cpu_results = cpu_session.run(None, {input_name: onnx_input})
    cpu_time = time.time() - cpu_start
    print(f"CPU inference complete in {cpu_time*1000:.2f} ms. Number of outputs: {len(cpu_results)}")
    for i, tensor in enumerate(cpu_results):
        print(f"  Output {i}: shape={tensor.shape}, dtype={tensor.dtype}, range=[{tensor.min():.3f}, {tensor.max():.3f}]")

    # Run inference on NPU
    print("\nRunning NPU inference...")
    npu_start = time.time()
    npu_results = npu_session.run(None, {input_name: onnx_input})
    npu_time = time.time() - npu_start
    print(f"NPU inference complete in {npu_time*1000:.2f} ms. Number of outputs: {len(npu_results)}")
    for i, tensor in enumerate(npu_results):
        print(f"  Output {i}: shape={tensor.shape}, dtype={tensor.dtype}, range=[{tensor.min():.3f}, {tensor.max():.3f}]")

    # Performance Summary
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    print(f"  CPU time: {cpu_time*1000:.2f} ms")
    print(f"  NPU time: {npu_time*1000:.2f} ms")
    speedup = cpu_time / npu_time if npu_time > 0 else 0
    print(f"  Speedup: {speedup:.2f}x {'(NPU faster)' if speedup > 1 else '(CPU faster)'}")
    print("="*60)

    # Visualize detections
    if visualize and len(cpu_results) >= 2:
        print("\n" + "="*60)
        print("VISUALIZING DETECTIONS")
        print("="*60)

        print(f"\nExtracting CPU detections (threshold={detection_threshold})...")

        # Debug: Check CPU confidence map statistics
        cpu_conf_map = cpu_results[0][0]
        cpu_conf_sigmoid = 1 / (1 + np.exp(-cpu_conf_map))
        print(f"CPU confidence map (raw) - min: {cpu_conf_map.min():.6f}, max: {cpu_conf_map.max():.6f}, mean: {cpu_conf_map.mean():.6f}")
        print(f"CPU confidence map (sigmoid) - min: {cpu_conf_sigmoid.min():.6f}, max: {cpu_conf_sigmoid.max():.6f}, mean: {cpu_conf_sigmoid.mean():.6f}")
        print(f"CPU values > {detection_threshold}: {(cpu_conf_sigmoid > detection_threshold).sum()}")

        cpu_detections = extract_detections(cpu_results[0], cpu_results[1], threshold=detection_threshold)
        print(f"CPU found {len(cpu_detections)} raw detections")

        print("Applying NMS to CPU detections...")
        cpu_detections = non_maximum_suppression(cpu_detections, iou_threshold=0.5)
        print(f"After NMS: {len(cpu_detections)} CPU detections")

        cpu_scaled = scale_and_format(cpu_detections, original_size)

        # Use separate threshold for NPU if specified
        npu_det_threshold = npu_threshold if npu_threshold is not None else detection_threshold
        print(f"\nExtracting NPU detections (threshold={npu_det_threshold})...")

        # Debug: Check NPU confidence map statistics
        npu_conf_map = npu_results[0][0]
        npu_conf_sigmoid = 1 / (1 + np.exp(-npu_conf_map))
        print(f"NPU confidence map (raw) - min: {npu_conf_map.min():.6f}, max: {npu_conf_map.max():.6f}, mean: {npu_conf_map.mean():.6f}")
        print(f"NPU confidence map (sigmoid) - min: {npu_conf_sigmoid.min():.6f}, max: {npu_conf_sigmoid.max():.6f}, mean: {npu_conf_sigmoid.mean():.6f}")
        print(f"NPU values > {npu_det_threshold}: {(npu_conf_sigmoid > npu_det_threshold).sum()}")

        npu_detections = extract_detections(npu_results[0], npu_results[1], threshold=npu_det_threshold)
        print(f"NPU found {len(npu_detections)} raw detections")

        print("Applying NMS to NPU detections...")
        npu_detections = non_maximum_suppression(npu_detections, iou_threshold=0.5)
        print(f"After NMS: {len(npu_detections)} NPU detections")

        npu_scaled = scale_and_format(npu_detections, original_size)

        cpu_vis = draw_boxes(original_img, cpu_scaled, color=(0, 255, 0), thickness=2)  # Green for CPU
        npu_vis = draw_boxes(original_img, npu_scaled, color=(255, 0, 0), thickness=2)  # Blue for NPU

        comparison = np.hstack([cpu_vis, npu_vis])

        os.makedirs(output_dir, exist_ok=True)
        cpu_path = os.path.join(output_dir, "cpu_detections.png")
        npu_path = os.path.join(output_dir, "npu_detections.png")
        comparison_path = os.path.join(output_dir, "cpu_npu_comparison.png")

        cv2.imwrite(cpu_path, cpu_vis)
        cv2.imwrite(npu_path, npu_vis)
        cv2.imwrite(comparison_path, comparison)

        print(f"\nSaved visualizations:")
        print(f"  CPU detections: {cpu_path}")
        print(f"  NPU detections: {npu_path}")
        print(f"  Comparison:     {comparison_path}")

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test NPU compiled model against CPU using real image input",
    )
    parser.add_argument(
        "onnx_model",
        type=str,
        help="Path to the ONNX model file",
    )

    parser.add_argument(
        "--image_path",
        type=str,
        help="Path to the test image (e.g., test.jpg)",
    )

    parser.add_argument(
        "--vai-config",
        type=str,
        required=True,
        help="Path to the vaip configuration json file",
    )

    parser.add_argument(
        "--cache-dir",
        required=False,
        type=str,
        default="./cache",
        help="Path to the cache directory (default: ./cache)",
    )

    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize and save detection results from CPU and NPU",
    )

    parser.add_argument(
        "--detection-threshold",
        type=float,
        default=0.7,
        help="Confidence threshold for CPU detections (default: 0.7)",
    )

    parser.add_argument(
        "--npu-threshold",
        type=float,
        default=None,
        help="Confidence threshold for NPU detections (default: same as --detection-threshold)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./detection_results",
        help="Directory to save visualization results (default: ./detection_results)",
    )

    parser.add_argument(
        "--pmode",
        type=str,
        choices=["default", "powersaver", "balanced", "performance", "turbo"],
        default="performance",
        help="NPU power mode (default: performance)",
    )

    args = parser.parse_args()

    success = main(
        args.onnx_model,
        args.image_path,
        args.vai_config,
        args.cache_dir,
        args.pmode,
        args.visualize,
        args.detection_threshold,
        args.npu_threshold,
        args.output_dir,
    )

    exit(0 if success else 1)
