import os
import argparse
from typing import Optional, Tuple, List, Dict
import numpy as np
import onnxruntime as ort
from pathlib import Path
import time
import cv2
import json

from utils.preprocessing import preprocess_image
from utils.postprocessing import (
    process_detections_with_features,
    draw_boxes,
    extract_text_from_detections,
    visualize_nearest_neighbors,
    visualize_detections_with_text
)
from utils.util import (
    configure_npu_power,
    load_charset,
    decode_tokens_to_text,
    export_results_to_json,
    load_onnx_model
)

os.environ["XLNX_ENABLE_CACHE"] = "1"
os.environ["PATH"] += (
    os.pathsep + f"{os.environ['CONDA_PREFIX']}\\Lib\\site-packages\\flexmlrt\\lib"
)


def get_cache_key_from_model_path(model_path):
    model_name = Path(model_path).stem
    return f"modelcachekey_{model_name}_bf16"


def process_relational(
    detections: List[Dict],
    relational_session: ort.InferenceSession,
    original_size: Tuple[int, int]
) -> Tuple[List[List[int]], np.ndarray]:
    """
    Process relational model to find nearest neighbors and reading order.

    Args:
        detections: List of detections with quad coordinates and recognizer features
        relational_session: ONNX Runtime session for relational model
        original_size: Original image size (h, w)

    Returns:
        Tuple of (neighbor_indices, null_logits)
        - neighbor_indices: List of top-15 neighbor indices for each detection
        - null_logits: Null probabilities for filtering invalid regions
    """
    num_detections = len(detections)
    if num_detections == 0:
        return [], np.array([])

    input_names = [inp.name for inp in relational_session.get_inputs()]
    print(f"    Relational input names: {input_names}")

    batch_size = 128
    num_to_process = min(num_detections, batch_size)

    # Prepare original quads
    original_quads = np.zeros((batch_size, 4, 2), dtype=np.float32)
    for i in range(num_to_process):
        quad = detections[i]['quad']
        original_quads[i] = np.array(quad, dtype=np.float32)

    # Prepare rectified quads
    rectified_quads = np.zeros((batch_size, 128, 2, 3), dtype=np.float32)
    for i in range(num_to_process):
        if 'relational_quad' in detections[i]:
            rectified_quads[i] = detections[i]['relational_quad']
        else:
            print(f" No relational_quad for detection {i}")

    # Prepare recognizer features
    recog_features = np.zeros((batch_size, 32, 256), dtype=np.float32)
    for i in range(num_to_process):
        if 'recognizer_output_features' in detections[i]:
            recog_features[i] = detections[i]['recognizer_output_features']
        else:
            print(f"No recognizer_output_features for detection {i}")

    relational_inputs = {
        input_names[0]: rectified_quads,
        input_names[1]: original_quads,
        input_names[2]: recog_features
    }

    relational_outputs = relational_session.run(None, relational_inputs)

    null_logits = relational_outputs[0][:num_to_process]
    neighbor_logits = relational_outputs[1][:num_to_process]
    neighbor_indices = relational_outputs[2][:num_to_process]

    return neighbor_indices.tolist(), null_logits


def main(
    detector_model: str,
    recognizer_model: str,
    relational_model: str,
    image_path: str,
    vaip_config: str,
    cache_dir: str = "./cache",
    pmode: str = "performance",
    detection_threshold: float = 0.7,
    npu_threshold: Optional[float] = None,
    output_dir: str = "./e2e_results",
):

    # Start total timer
    total_start_time = time.time()

    for path, name in [(detector_model, "Detector"),
                       (recognizer_model, "Recognizer"),
                       (relational_model, "Relational"),
                       (image_path, "Image"),
                       (vaip_config, "VAIP config")]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} file not found: {path}")

    os.makedirs(output_dir, exist_ok=True)

    # Configure NPU power mode
    print(f"\n Configuring NPU power mode: {pmode}")
    ret_code, stdout, stderr = configure_npu_power(pmode)
    if ret_code == 0:
        print("NPU configured")
    else:
        print(f"{stderr}")

    # Preprocess image
    print(f"\nPreprocessing image: {image_path}")
    onnx_input, original_img, original_size = preprocess_image(image_path, output_format='NCHW')
    print(f"  Original size: {original_size}, Input shape: {onnx_input.shape}")

    # Load detector models
    print(f"\n[DETECTOR] Loading models")
    detector_cache_key = get_cache_key_from_model_path(detector_model)
    cpu_detector_session = load_onnx_model(detector_model, device="CPU")
    npu_detector_session = load_onnx_model(
        detector_model, device="NPU",
        vaip_config=vaip_config,
        cache_dir=cache_dir,
        cache_key=detector_cache_key
    )
    detector_input_name = cpu_detector_session.get_inputs()[0].name

    # Run detector inference
    print("\n[DETECTOR] Running inference (CPU vs NPU)")

    # CPU detector
    cpu_det_start = time.time()
    cpu_det_results = cpu_detector_session.run(None, {detector_input_name: onnx_input})
    cpu_det_time = time.time() - cpu_det_start

    # NPU detector
    npu_det_start = time.time()
    npu_det_results = npu_detector_session.run(None, {detector_input_name: onnx_input})
    npu_det_time = time.time() - npu_det_start

    print(f"  CPU: {cpu_det_time*1000:.2f} ms | NPU: {npu_det_time*1000:.2f} ms")
    print(f"  Speedup: {cpu_det_time/npu_det_time:.2f}x")

    # Process detections with features (CPU)
    print("\nProcessing CPU detections...")
    cpu_scaled = process_detections_with_features(
        cpu_det_results,
        original_size,
        detection_threshold=detection_threshold,
        iou_threshold=0.5,
        device_name="CPU"
    )

    # Process detections with features (NPU)
    print("\nProcessing NPU detections...")
    npu_det_threshold = npu_threshold if npu_threshold is not None else detection_threshold
    npu_scaled = process_detections_with_features(
        npu_det_results,
        original_size,
        detection_threshold=npu_det_threshold,
        iou_threshold=0.5,
        device_name="NPU"
    )

    print(f"\n  Final: CPU={len(cpu_scaled)} detections, NPU={len(npu_scaled)} detections")

    # Save detector visualization
    cpu_det_vis = draw_boxes(original_img, cpu_scaled, color=(0, 255, 0), thickness=2)
    npu_det_vis = draw_boxes(original_img, npu_scaled, color=(255, 0, 0), thickness=2)
    cv2.imwrite(os.path.join(output_dir, "1_detector_cpu.png"), cpu_det_vis)
    cv2.imwrite(os.path.join(output_dir, "1_detector_npu.png"), npu_det_vis)
    cv2.imwrite(os.path.join(output_dir, "1_detector_comparison.png"),
                np.hstack([cpu_det_vis, npu_det_vis]))

    # Load charset
    idx_to_char = load_charset("utils/charset.txt")

    # Load recognizer models
    print(f"\n[RECOGNIZER] Loading models")
    recognizer_cache_key = get_cache_key_from_model_path(recognizer_model)
    cpu_recognizer_session = load_onnx_model(recognizer_model, device="CPU")
    npu_recognizer_session = load_onnx_model(
        recognizer_model, device="NPU",
        vaip_config=vaip_config,
        cache_dir=cache_dir,
        cache_key=recognizer_cache_key
    )
    recognizer_input_name = cpu_recognizer_session.get_inputs()[0].name

    print("\n[RECOGNIZER] Running inference (extracting text)")

    # Extract text from CPU detections using CPU recognizer
    cpu_with_text = []
    if len(cpu_scaled) > 0:
        print(f"  Processing {len(cpu_scaled)} CPU detections with CPU recognizer...")
        cpu_recog_start = time.time()
        cpu_with_text = extract_text_from_detections(
            cpu_scaled,
            cpu_det_results[2],
            cpu_recognizer_session,
            recognizer_input_name,
            idx_to_char
        )
        cpu_recog_time = time.time() - cpu_recog_start
        print(f"  CPU recognizer: {cpu_recog_time*1000:.2f} ms ({cpu_recog_time/len(cpu_scaled)*1000:.2f} ms/detection)")

    # Extract text from NPU detections using NPU recognizer
    npu_with_text = []
    if len(npu_scaled) > 0:
        print(f"  Processing {len(npu_scaled)} NPU detections with NPU recognizer...")
        npu_recog_start = time.time()
        npu_with_text = extract_text_from_detections(
            npu_scaled,
            npu_det_results[2],
            npu_recognizer_session,
            recognizer_input_name,
            idx_to_char
        )
        npu_recog_time = time.time() - npu_recog_start
        print(f"  NPU recognizer: {npu_recog_time*1000:.2f} ms ({npu_recog_time/len(npu_scaled)*1000:.2f} ms/detection)")

    if len(cpu_with_text) > 0:
        cpu_text_vis = visualize_detections_with_text(
            original_img, cpu_with_text, show_text=True,
            color=(0, 255, 0), thickness=2
        )
        cv2.imwrite(os.path.join(output_dir, "2_recognizer_cpu.png"), cpu_text_vis)

    if len(npu_with_text) > 0:
        npu_text_vis = visualize_detections_with_text(
            original_img, npu_with_text, show_text=True,
            color=(255, 0, 0), thickness=2
        )
        cv2.imwrite(os.path.join(output_dir, "2_recognizer_npu.png"), npu_text_vis)

    if len(cpu_with_text) > 0 and len(npu_with_text) > 0:
        cv2.imwrite(os.path.join(output_dir, "2_recognizer_comparison.png"),
                    np.hstack([cpu_text_vis, npu_text_vis]))

    if len(cpu_with_text) > 0 or len(npu_with_text) > 0:
        print(f"\n[RELATIONAL] Loading models")
        relational_cache_key = get_cache_key_from_model_path(relational_model)
        cpu_relational_session = load_onnx_model(relational_model, device="CPU")
        npu_relational_session = load_onnx_model(
            relational_model, device="NPU",
            vaip_config=vaip_config,
            cache_dir=cache_dir,
            cache_key=relational_cache_key
        )

        print("\n[RELATIONAL] Running inference (finding nearest neighbors)")

        # Process CPU detections
        cpu_neighbors = []
        if len(cpu_with_text) > 0:
            print(f"  Processing {len(cpu_with_text)} CPU detections with CPU relational...")
            cpu_rel_start = time.time()
            cpu_neighbors, cpu_null_logits = process_relational(
                cpu_with_text,
                cpu_relational_session,
                original_size
            )
            cpu_rel_time = time.time() - cpu_rel_start
            print(f"  CPU relational: {cpu_rel_time*1000:.2f} ms")

        # Process NPU detections
        npu_neighbors = []
        if len(npu_with_text) > 0:
            print(f"  Processing {len(npu_with_text)} NPU detections with NPU relational...")
            npu_rel_start = time.time()
            npu_neighbors, npu_null_logits = process_relational(
                npu_with_text,
                npu_relational_session,
                original_size
            )
            npu_rel_time = time.time() - npu_rel_start
            print(f"  NPU relational: {npu_rel_time*1000:.2f} ms")

        # Visualize nearest neighbor graphs
        if len(cpu_neighbors) > 0:
            cpu_rel_vis = visualize_nearest_neighbors(
                original_img, cpu_with_text, cpu_neighbors,
                max_neighbors=3, color=(0, 255, 255)
            )
            cv2.imwrite(os.path.join(output_dir, "3_relational_cpu.png"), cpu_rel_vis)

        if len(npu_neighbors) > 0:
            npu_rel_vis = visualize_nearest_neighbors(
                original_img, npu_with_text, npu_neighbors,
                max_neighbors=3, color=(255, 255, 0)
            )
            cv2.imwrite(os.path.join(output_dir, "3_relational_npu.png"), npu_rel_vis)

        if len(cpu_neighbors) > 0 and len(npu_neighbors) > 0:
            cv2.imwrite(os.path.join(output_dir, "3_relational_comparison.png"),
                        np.hstack([cpu_rel_vis, npu_rel_vis]))

    # CPU results
    if len(cpu_with_text) > 0:
        cpu_timing = {
            "detector_ms": cpu_det_time * 1000,
            "recognizer_ms": cpu_recog_time * 1000 if 'cpu_recog_time' in locals() else 0,
            "relational_ms": cpu_rel_time * 1000 if 'cpu_rel_time' in locals() else 0,
            "total_ms": (cpu_det_time + cpu_recog_time + (cpu_rel_time if 'cpu_rel_time' in locals() else 0)) * 1000
        }
        cpu_json_path = os.path.join(output_dir, "cpu_results.json")
        export_results_to_json(
            cpu_with_text,
            cpu_neighbors if 'cpu_neighbors' in locals() else [],
            cpu_json_path,
            cpu_timing,
            "CPU"
        )
        print(f"  Saved CPU results to: {cpu_json_path}")

    # NPU results
    if len(npu_with_text) > 0:
        npu_timing = {
            "detector_ms": npu_det_time * 1000,
            "recognizer_ms": npu_recog_time * 1000 if 'npu_recog_time' in locals() else 0,
            "relational_ms": npu_rel_time * 1000 if 'npu_rel_time' in locals() else 0,
            "total_ms": (npu_det_time + npu_recog_time + (npu_rel_time if 'npu_rel_time' in locals() else 0)) * 1000
        }
        npu_json_path = os.path.join(output_dir, "npu_results.json")
        export_results_to_json(
            npu_with_text,
            npu_neighbors if 'npu_neighbors' in locals() else [],
            npu_json_path,
            npu_timing,
            "NPU"
        )
        print(f"  Saved NPU results to: {npu_json_path}")

    # Calculate total time
    total_time = time.time() - total_start_time

    # Print summary
    print("\n" + "="*80)
    print("PIPELINE SUMMARY")
    print("="*80)
    print(f"Total Pipeline Time: {total_time:.2f}s")
    print(f"\nDetector Performance:")
    print(f"  CPU: {cpu_det_time*1000:.2f} ms | NPU: {npu_det_time*1000:.2f} ms | Speedup: {cpu_det_time/npu_det_time:.2f}x")
    print(f"Detections Found:")
    print(f"  CPU: {len(cpu_scaled)} boxes | NPU: {len(npu_scaled)} boxes")

    if len(cpu_with_text) > 0 or len(npu_with_text) > 0:
        print(f"\nRecognizer Performance:")
        if len(cpu_with_text) > 0:
            print(f"  CPU: {cpu_recog_time*1000:.2f} ms ({cpu_recog_time/len(cpu_scaled)*1000:.2f} ms/detection)")
        if len(npu_with_text) > 0:
            print(f"  NPU: {npu_recog_time*1000:.2f} ms ({npu_recog_time/len(npu_scaled)*1000:.2f} ms/detection)")
        if len(cpu_with_text) > 0 and len(npu_with_text) > 0:
            print(f"  Speedup: {cpu_recog_time/npu_recog_time:.2f}x")
        print(f"Text Regions Recognized:")
        print(f"  CPU: {len(cpu_with_text)} | NPU: {len(npu_with_text)}")

    if 'cpu_neighbors' in locals() and 'npu_neighbors' in locals():
        if len(cpu_neighbors) > 0 or len(npu_neighbors) > 0:
            print(f"\nRelational Performance:")
            if len(cpu_neighbors) > 0:
                print(f"  CPU: {cpu_rel_time*1000:.2f} ms")
            if len(npu_neighbors) > 0:
                print(f"  NPU: {npu_rel_time*1000:.2f} ms")
            if len(cpu_neighbors) > 0 and len(npu_neighbors) > 0:
                print(f"  Speedup: {cpu_rel_time/npu_rel_time:.2f}x")

    print(f"\nOutput files:")
    print(f"  Images:")
    print(f"    {output_dir}/1_detector_cpu.png - CPU detector boxes")
    print(f"    {output_dir}/1_detector_npu.png - NPU detector boxes")
    print(f"    {output_dir}/1_detector_comparison.png - Side-by-side comparison")
    print(f"    {output_dir}/2_recognizer_cpu.png - CPU detector + recognizer")
    print(f"    {output_dir}/2_recognizer_npu.png - NPU detector + recognizer")
    print(f"    {output_dir}/2_recognizer_comparison.png - Side-by-side comparison")
    if 'cpu_neighbors' in locals() or 'npu_neighbors' in locals():
        if len(cpu_neighbors) > 0 or len(npu_neighbors) > 0:
            print(f"    {output_dir}/3_relational_cpu.png - CPU nearest neighbor graph")
            print(f"    {output_dir}/3_relational_npu.png - NPU nearest neighbor graph")
            print(f"    {output_dir}/3_relational_comparison.png - Side-by-side comparison")
    print(f"  JSON Results:")
    print(f"    {output_dir}/cpu_results.json - CPU extracted text & metrics")
    print(f"    {output_dir}/npu_results.json - NPU extracted text & metrics")
    print("="*80)

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="End-to-end OCR pipeline test: Detector + Recognizer + Relational (CPU vs NPU)",
    )

    parser.add_argument(
        "--detector",
        type=str,
        required=True,
        help="Path to detector ONNX model",
    )

    parser.add_argument(
        "--recognizer",
        type=str,
        required=True,
        help="Path to recognizer ONNX model",
    )

    parser.add_argument(
        "--relational",
        type=str,
        required=True,
        help="Path to relational ONNX model",
    )

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to test image",
    )

    parser.add_argument(
        "--vai-config",
        type=str,
        required=True,
        help="Path to VitisAI configuration JSON",
    )

    parser.add_argument(
        "--cache-dir",
        type=str,
        default="./cache",
        help="Cache directory (default: ./cache)",
    )

    parser.add_argument(
        "--pmode",
        type=str,
        choices=["default", "powersaver", "balanced", "performance", "turbo"],
        default="performance",
        help="NPU power mode (default: performance)",
    )

    parser.add_argument(
        "--detection-threshold",
        type=float,
        default=0.7,
        help="Detection confidence threshold for CPU (default: 0.7)",
    )

    parser.add_argument(
        "--npu-threshold",
        type=float,
        default=None,
        help="Detection confidence threshold for NPU (default: same as CPU)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="./e2e_results",
        help="Output directory for results (default: ./e2e_results)",
    )


    args = parser.parse_args()

    success = main(
        detector_model=args.detector,
        recognizer_model=args.recognizer,
        relational_model=args.relational,
        image_path=args.image,
        vaip_config=args.vai_config,
        cache_dir=args.cache_dir,
        pmode=args.pmode,
        detection_threshold=args.detection_threshold,
        npu_threshold=args.npu_threshold,
        output_dir=args.output_dir,
    )

    exit(0 if success else 1)
