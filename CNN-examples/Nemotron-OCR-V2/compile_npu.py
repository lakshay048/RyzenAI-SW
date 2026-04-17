import os
import onnxruntime as ort
import argparse
from typing import Union, Tuple, Optional
from onnx import ModelProto, load, checker, save
import tempfile
from pathlib import Path
from onnx import version_converter, shape_inference
import time
import numpy as np

os.environ["DEBUG_LOG_LEVEL"] = "info"
os.environ["DEBUG_VAIML_PARTITION"] = "2"


def get_opset(model: ModelProto) -> Union[None, int]:
    for import_ in model.opset_import:
        if import_.domain == "" or import_.domain == "ai.onnx":
            return import_.version
    return None

def get_cache_key_from_model_path(model_path):
    model_name = Path(model_path).stem  # Gets the filename without extension
    return f"modelcachekey_{model_name}_bf16"

def generate_dummy_input(model_proto: ModelProto) -> dict:
    inputs = {}
    for input_tensor in model_proto.graph.input:
        name = input_tensor.name
        shape = [dim.dim_value if dim.dim_value > 0 else 1 for dim in input_tensor.type.tensor_type.shape.dim]
        dtype = input_tensor.type.tensor_type.elem_type

        # Map ONNX data types to numpy
        if dtype == 1:  # FLOAT
            np_dtype = np.float32
        elif dtype == 11:  # DOUBLE
            np_dtype = np.float64
        elif dtype == 6:  # INT32
            np_dtype = np.int32
        elif dtype == 7:  # INT64
            np_dtype = np.int64
        else:
            np_dtype = np.float32  # Default fallback

        inputs[name] = np.random.rand(*shape).astype(np_dtype)

    return inputs

def benchmark_inference(
    session: ort.InferenceSession,
    input_data: dict,
    num_runs: int = 100,
    warmup_runs: int = 1
) -> Tuple[float, float, float]:
    # Warmup runs
    for _ in range(warmup_runs):
        _ = session.run(None, input_data)

    start_time = time.time()
    for _ in range(num_runs):
        _ = session.run(None, input_data)
    end_time = time.time()

    total_time = end_time - start_time
    avg_latency_ms = (total_time / num_runs) * 1000  # Convert to milliseconds
    throughput_per_sec = num_runs / total_time if total_time > 0 else 0

    return total_time, avg_latency_ms, throughput_per_sec

def optimize_onnx_model(
    model_proto: ModelProto,
    model_filepath: Optional[str] = None,
    optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
) -> Tuple[ModelProto, bool]:

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = optimization_level
            sess_options.optimized_model_filepath = str(
                os.path.join(tmpdir, "opt.onnx")
            )
            _ = ort.InferenceSession(
                model_filepath if model_filepath else model_proto.SerializeToString(),
                sess_options,
                providers=["CPUExecutionProvider"],
            )
        except Exception as e:
            print(f"Error loading model into inference session: {e}")
            return model_proto, False

        try:
            checker.check_model(sess_options.optimized_model_filepath, full_check=True)
            model_proto = load(
                sess_options.optimized_model_filepath, load_external_data=False
            )
            model_proto = shape_inference.infer_shapes(
                model_proto,
                strict_mode=True,
                data_prop=True,
            )
            return model_proto, True

        except checker.ValidationError:
            print("Model did not pass checker!")
            return model_proto, False


def compile_model(
    model_file: str,
    vai_config: str,
    optimize: bool = False,
    opset: Optional[int] = None,
    cache_dir: str = "./cache",
    benchmark: bool = False,
    num_runs: int = 100
) -> None:

    assert os.path.exists(model_file)
    assert os.path.exists(vai_config)

    model_proto = load(model_file)

    if optimize:
        model_proto, optimizer_passed = optimize_onnx_model(model_proto)

        if optimizer_passed:
            model_file = os.path.splitext(os.path.basename(model_file))[0] + "_opt.onnx"
            save(model_proto, model_file)
        else:
            print("Optimizer Failed!")

    if opset and get_opset(model_proto) != opset:
        model_proto = version_converter.convert_version(model_proto, opset)

    print(f"Model opset is {get_opset(model_proto)}")
    print(f"Flow type: BF16")
    print(f"Cache directory: {cache_dir}")

    cache_key = get_cache_key_from_model_path(model_file)
    print(f"Cache key: {cache_key}")

    provider_options = [{
        "config_file": vai_config,
        'cacheDir': cache_dir,
        'cacheKey': cache_key
    }]

    session = ort.InferenceSession(
        model_file,
        providers=["VitisAIExecutionProvider"],
        provider_options=provider_options,
    )

    print("\n=== Compilation Successful ===\n")

    # Run performance benchmark if requested
    if benchmark:
        print("=== Performance Benchmarking ===")
        print(f"Running {num_runs} inferences...")

        input_data = generate_dummy_input(model_proto)

        try:
            total_time, avg_latency_ms, throughput_per_sec = benchmark_inference(
                session, input_data, num_runs=num_runs
            )

            print(f"\nBenchmark Results:")
            print(f"  Total runs:            {num_runs}")
            print(f"  Total time:            {total_time:.4f} seconds")
            print(f"  Average latency:       {avg_latency_ms:.2f} ms/inference")
            print(f"  Throughput:            {throughput_per_sec:.2f} inferences/second")
            print("=" * 40)

        except Exception as e:
            print(f"Benchmarking failed: {e}")
            print("Note: Compilation was successful, but benchmarking encountered an error.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Utility script to make the process of running a VAIML compile streamlined.",
    )
    parser.add_argument(
        "onnx_model",
        type=str,
        help="Provide the ONNX model file.",
    )

    parser.add_argument(
        "--vai-config",
        type=str,
        required=True,
        help="Path to the vaip configuration json file.",
    )

    parser.add_argument(
        "--opset",
        type=int,
        required=False,
        help="Force an opset version on the model before compilation.",
    )

    parser.add_argument(
        "--optimize",
        action="store_true",
        help=(
            "If supplied, the input ONNX model will be optimized and saved to a new file. "
            "The compilation process will then use this optimized model."
        ),
    )


    parser.add_argument(
        "--cache-dir",
        type=str,
        required=False,
        default="./cache",
        help="Cache directory path (default: ./cache)",
    )

    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run performance benchmarking after compilation",
    )

    parser.add_argument(
        "--num-runs",
        type=int,
        default=100,
        help="Number of inference runs for benchmarking (default: 100)",
    )

    args = parser.parse_args()

    compile_model(
        args.onnx_model,
        args.vai_config,
        args.optimize,
        args.opset,
        args.cache_dir,
        args.benchmark,
        args.num_runs
    )
