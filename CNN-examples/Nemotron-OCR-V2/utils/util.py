import subprocess
import platform
import json
import numpy as np
from typing import Optional, Tuple, List, Dict

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
            command = [XRT_SMI_PATH, "examine", "--report", "platform"]

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


def load_charset(charset_path: str = "utils/charset.txt") -> Dict[int, str]:
    with open(charset_path, 'r', encoding='utf-8') as f:
        charset = json.load(f)

    idx_to_char = {0: '<PAD>', 1: '<EOS>', 2: '<UNK>'}
    for i, char in enumerate(charset):
        idx_to_char[i + 3] = char

    return idx_to_char


def decode_tokens_to_text(token_ids: np.ndarray, idx_to_char: Dict[int, str]) -> Tuple[str, float]:
    eos_pos = np.where(token_ids == 1)[0]
    seq_len = eos_pos[0] if len(eos_pos) > 0 else len(token_ids)

    text = ""
    for token_id in token_ids[:seq_len]:
        if token_id > 2:  # Valid character
            text += idx_to_char.get(token_id, '?')

    return text if text else "[empty]", 1.0


def export_results_to_json(
    detections: List[Dict],
    neighbors: List[List[int]],
    output_path: str,
    inference_times: Dict[str, float],
    device: str
):
    results = {
        "device": device,
        "inference_times_ms": inference_times,
        "num_detections": len(detections),
        "text_regions": []
    }

    for i, det in enumerate(detections):
        region = {
            "id": i,
            "text": det.get('text', ''),
            "confidence": float(det.get('confidence', 0)),
            "text_confidence": float(det.get('text_confidence', 0)),
            "bounding_box": {
                "quad": [[float(x), float(y)] for x, y in det['quad']],
                "normalized": {
                    "left": float(det.get('left', 0)),
                    "right": float(det.get('right', 0)),
                    "upper": float(det.get('upper', 0)),
                    "lower": float(det.get('lower', 0))
                }
            }
        }

        if i < len(neighbors):
            region["nearest_neighbors"] = [int(n) for n in neighbors[i][:5]]  # Top 5

        results["text_regions"].append(region)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def load_onnx_model(
    model_path: str,
    device: str = "CPU",
    vaip_config: Optional[str] = None,
    cache_dir: str = "./cache",
    cache_key: Optional[str] = None
) -> 'ort.InferenceSession':
    """
    Load ONNX model for CPU or NPU execution.

    Args:
        model_path: Path to ONNX model file
        device: "CPU" or "NPU"
        vaip_config: Path to VitisAI config JSON (required for NPU)
        cache_dir: Cache directory for NPU compilation
        cache_key: Cache key for NPU (auto-generated if None)

    Returns:
        ONNX Runtime InferenceSession
    """
    import onnxruntime as ort
    from pathlib import Path

    if device.upper() == "CPU":
        print(f"  Loading {Path(model_path).stem} on CPU...")
        session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"]
        )
    elif device.upper() == "NPU":
        if cache_key is None:
            # Auto-generate cache key from model name
            model_name = Path(model_path).stem
            cache_key = f"modelcachekey_{model_name}_bf16"

        print(f"  Loading {Path(model_path).stem} on NPU...")
        print(f"    Cache key: {cache_key}")

        npu_options = {
            "config_file": vaip_config,
            "cacheDir": cache_dir,
            "cacheKey": cache_key,
        }

        session = ort.InferenceSession(
            model_path,
            providers=["VitisAIExecutionProvider"],
            provider_options=[npu_options]
        )
    else:
        raise ValueError(f"Invalid device: {device}. Must be 'CPU' or 'NPU'")

    return session
