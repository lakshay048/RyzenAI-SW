#!/usr/bin/env python
"""
Run VLM benchmarks across all image sizes and combine results.

Spawns a separate process per image size for clean memory isolation,
then combines results into a summary table and JSON file.

Usage:
    # Basic usage (requires model path)
    python run_all_benchmarks.py -m /path/to/model

    # With custom image and output directory
    python run_all_benchmarks.py -m /path/to/model -i test_image.jpg -o ./results

    # Run specific sizes with verbose output
    python run_all_benchmarks.py -m /path/to/model --sizes 1024x1024 2048x2048 -v
"""

import argparse
import subprocess
import json
import os
import sys
import time
from pathlib import Path

# ============================================================================
# DEFAULT CONFIGURATION
# ============================================================================

DEFAULT_SOURCE_IMAGE = "./test_image.jpg"
DEFAULT_OUTPUT_DIR = "./benchmark_results"
DEFAULT_PROMPT = "Describe this image in detail."
DEFAULT_MAX_TOKENS = 30
DEFAULT_MAX_LENGTH = 16384
DEFAULT_NUM_ITERATIONS = 2
DEFAULT_WARMUP_ITERATIONS = 1
DEFAULT_SLEEP_BETWEEN_RUNS = 2  # seconds

# Image sizes to benchmark (width x height)
DEFAULT_IMAGE_SIZES = [
    "448x448",
    "512x512",
    "768x768",
    "1024x1024",
    "1536x1536",
    "2048x2048",
    "3072x3072",
    "4032x4032",
]


def run_size_benchmark(
    size: str,
    model_path: str,
    source_image: str,
    output_json: str,
    prompt: str,
    max_tokens: int,
    max_length: int,
    num_iterations: int,
    warmup_iterations: int,
    verbose: bool = False
) -> dict:
    """Run benchmark for a single image size in a subprocess."""
    w, h = size.split("x")

    cmd = [
        sys.executable, "vlm_benchmark.py",
        "-m", model_path,
        "-i", source_image,
        "--image_size", w, h,
        "--max_tokens", str(max_tokens),
        "--max_length", str(max_length),
        "-n", str(num_iterations),
        "-w", str(warmup_iterations),
        "-p", prompt,
        "-o", output_json,
    ]

    if verbose:
        cmd.append("-v")

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        return {"error": f"Process exited with code {result.returncode}"}

    if os.path.exists(output_json):
        with open(output_json, "r") as f:
            return json.load(f)

    return {"error": "No output JSON produced"}


def generate_summary_table(all_results: dict, image_sizes: list, config: dict, compact: bool = False) -> str:
    """Generate a markdown summary table of all benchmark results."""
    lines = []

    # Extract text token count (constant across sizes) from the first valid result
    text_tokens = None
    for size in image_sizes:
        if size in all_results and "error" not in all_results[size]:
            text_tokens = all_results[size]["summary"].get("token_breakdown", {}).get("text_tokens")
            break

    lines.append("# Benchmark Summary")
    lines.append("")
    lines.append(f"- **Model:** {config['model_path']}")
    lines.append(f"- **Image:** {config['source_image']}")
    lines.append(f"- **Prompt:** \"{config['prompt']}\"")
    if text_tokens is not None:
        lines.append(f"- **Text tokens (constant):** {text_tokens}")
    lines.append("")

    if compact:
        # Compact table: Size, Total Tokens, Image Tokens, TTFT (s), Decode TPS
        lines.append("| Size | Total Tok | Img Tok | TTFT (s) | TPS (tok/s) |")
        lines.append("|------|-----------|---------|----------|-------------|")
    else:
        # Full table header
        lines.append("| Size | Img Tok | Total Tok | Preprocess (ms) | TTFT (s) | Prefill (tok/s) | Decode (tok/s) | Decode Lat (ms/tok) |")
        lines.append("|------|---------|-----------|-----------------|----------|-----------------|----------------|---------------------|")

    for size in image_sizes:
        if size not in all_results or "error" in all_results[size]:
            error_msg = all_results.get(size, {}).get("error", "unknown error")
            if compact:
                lines.append(f"| {size} | ERROR | {error_msg} | | |")
            else:
                lines.append(f"| {size} | ERROR | {error_msg} | | | | | |")
            continue

        summary = all_results[size]["summary"]
        tb = summary.get("token_breakdown", {})
        img_tokens = tb.get("image_tokens", "N/A")
        total_tokens = tb.get("prompt_tokens", "N/A")
        preprocess = summary["preprocessing_ms"]["avg"]
        ttft_s = summary["ttft_ms"]["avg"] / 1000.0
        prefill_tps = summary.get("prefill_throughput_tps", {}).get("avg", 0)
        decode_tps = summary["token_generation_ms"]["tps_avg"]
        decode_latency = summary["token_generation_ms"]["avg"]

        if compact:
            lines.append(
                f"| {size} | {total_tokens} | {img_tokens} | {ttft_s:.2f} | {decode_tps:.2f} |"
            )
        else:
            lines.append(
                f"| {size} | {img_tokens} | {total_tokens} | {preprocess:.2f} | {ttft_s:.2f} | {prefill_tps:.2f} | {decode_tps:.2f} | {decode_latency:.2f} |"
            )

    return "\n".join(lines)


def print_summary_table(all_results: dict, image_sizes: list, config: dict, compact: bool = False):
    """Print a formatted summary table of all benchmark results to console."""

    # Extract text token count (constant across sizes) from the first valid result
    text_tokens = None
    for size in image_sizes:
        if size in all_results and "error" not in all_results[size]:
            text_tokens = all_results[size]["summary"].get("token_breakdown", {}).get("text_tokens")
            break

    width = 60 if compact else 110
    print(f"\n{'='*width}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*width}")
    print(f"Model:  {config['model_path']}")
    print(f"Image:  {config['source_image']}")
    print(f"Prompt: \"{config['prompt']}\"")
    if text_tokens is not None:
        print(f"Text tokens (constant across sizes): {text_tokens}")
    print()

    if compact:
        # Compact header: Size, Total Tokens, Image Tokens, TTFT (incl preprocessing), Decode TPS
        header = (
            f"{'Size':<12} "
            f"{'Total Tok':<11} "
            f"{'Img Tok':<10} "
            f"{'TTFT (s)':<12} "
            f"{'TPS':<12}"
        )
        units = (
            f"{'':<12} "
            f"{'':<11} "
            f"{'':<10} "
            f"{'':<12} "
            f"{'(tok/s)':<12}"
        )
        print(header)
        print(units)
        print("-" * width)
    else:
        header = (
            f"{'Size':<12} "
            f"{'Img Tok':<10} "
            f"{'Total Tok':<11} "
            f"{'Preproc (ms)':<14} "
            f"{'TTFT (s)':<12} "
            f"{'Prefill':<14} "
            f"{'Decode':<14} "
            f"{'Decode Lat':<12}"
        )
        units = (
            f"{'':<12} "
            f"{'':<10} "
            f"{'':<11} "
            f"{'':<14} "
            f"{'':<12} "
            f"{'(tokens/s)':<14} "
            f"{'(tokens/s)':<14} "
            f"{'(ms/token)':<12}"
        )
        print(header)
        print(units)
        print("-" * width)

    for size in image_sizes:
        if size not in all_results or "error" in all_results[size]:
            error_msg = all_results.get(size, {}).get("error", "unknown error")
            print(f"{size:<12} ERROR: {error_msg}")
            continue

        summary = all_results[size]["summary"]
        tb = summary.get("token_breakdown", {})
        img_tokens = tb.get("image_tokens", "N/A")
        total_tokens = tb.get("prompt_tokens", "N/A")
        preprocess = summary["preprocessing_ms"]["avg"]
        decode_tps = summary["token_generation_ms"]["tps_avg"]

        if compact:
            ttft_s = summary["ttft_ms"]["avg"] / 1000.0
            print(
                f"{size:<12} "
                f"{total_tokens:<11} "
                f"{img_tokens:<10} "
                f"{ttft_s:<12.2f} "
                f"{decode_tps:<12.2f}"
            )
        else:
            ttft_s = summary["ttft_ms"]["avg"] / 1000.0
            prefill_tps = summary.get("prefill_throughput_tps", {}).get("avg", 0)
            decode_latency = summary["token_generation_ms"]["avg"]
            print(
                f"{size:<12} "
                f"{img_tokens:<10} "
                f"{total_tokens:<11} "
                f"{preprocess:<14.2f} "
                f"{ttft_s:<12.2f} "
                f"{prefill_tps:<14.2f} "
                f"{decode_tps:<14.2f} "
                f"{decode_latency:<12.2f}"
            )

    print(f"{'='*width}")


def main():
    parser = argparse.ArgumentParser(
        description="Run VLM benchmarks across multiple image sizes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_all_benchmarks.py -m ./phi-4-mm-onnx
  python run_all_benchmarks.py -m ./model -i image.jpg -o ./results
  python run_all_benchmarks.py -m ./model --sizes 1024x1024 2048x2048 -v
        """
    )

    # Required arguments
    parser.add_argument(
        "-m", "--model_path",
        type=str,
        required=True,
        help="Path to the ONNX model folder"
    )

    # Optional arguments with defaults
    parser.add_argument(
        "-i", "--image",
        type=str,
        default=DEFAULT_SOURCE_IMAGE,
        help=f"Source image path (default: {DEFAULT_SOURCE_IMAGE})"
    )
    parser.add_argument(
        "-o", "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for results (default: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "-p", "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help=f"Prompt text (default: \"{DEFAULT_PROMPT}\")"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"Max tokens to generate (default: {DEFAULT_MAX_TOKENS})"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=DEFAULT_MAX_LENGTH,
        help=f"Max context length for KV cache (default: {DEFAULT_MAX_LENGTH})"
    )
    parser.add_argument(
        "-n", "--num_iterations",
        type=int,
        default=DEFAULT_NUM_ITERATIONS,
        help=f"Number of benchmark iterations (default: {DEFAULT_NUM_ITERATIONS})"
    )
    parser.add_argument(
        "-w", "--warmup_iterations",
        type=int,
        default=DEFAULT_WARMUP_ITERATIONS,
        help=f"Number of warmup iterations (default: {DEFAULT_WARMUP_ITERATIONS})"
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=str,
        help="Specific sizes to benchmark (e.g., 1024x1024 2048x2048)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output (shows generated text)"
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Print compact summary (Total Tokens, Image Tokens, TTFT incl. preprocessing, Decode TPS)"
    )

    args = parser.parse_args()

    # Determine image sizes to benchmark
    image_sizes = args.sizes if args.sizes else DEFAULT_IMAGE_SIZES

    # Validate model path
    if not os.path.exists(args.model_path):
        print(f"ERROR: Model path not found: {args.model_path}")
        sys.exit(1)

    # Validate source image
    if not os.path.exists(args.image):
        print(f"ERROR: Source image not found: {args.image}")
        print(f"Generate one with: python generate_test_image.py -o {args.image}")
        sys.exit(1)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Store config for summary
    config = {
        "model_path": args.model_path,
        "source_image": args.image,
        "prompt": args.prompt,
        "max_tokens": args.max_tokens,
        "max_length": args.max_length,
        "num_iterations": args.num_iterations,
        "warmup_iterations": args.warmup_iterations,
    }

    all_results = {}

    for i, size in enumerate(image_sizes):
        output_json = f"{args.output_dir}/benchmark_{size}.json"

        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(image_sizes)}] Running benchmark for {size}")
        print(f"{'='*60}")

        try:
            data = run_size_benchmark(
                size=size,
                model_path=args.model_path,
                source_image=args.image,
                output_json=output_json,
                prompt=args.prompt,
                max_tokens=args.max_tokens,
                max_length=args.max_length,
                num_iterations=args.num_iterations,
                warmup_iterations=args.warmup_iterations,
                verbose=args.verbose
            )

            if "error" in data:
                print(f"  ERROR: {data['error']}")
                all_results[size] = data
            else:
                all_results[size] = data
                s = data["summary"]
                tb = s.get("token_breakdown", {})
                print(f"  Tokens: total={tb.get('prompt_tokens', '?')}, "
                      f"image={tb.get('image_tokens', '?')}, "
                      f"text={tb.get('text_tokens', '?')}")
                print(f"  Preprocess: {s['preprocessing_ms']['avg']:.2f} ms")
                print(f"  TTFT:   {s['ttft_ms']['avg']:.2f} ms")
                print(f"  Decode: {s['token_generation_ms']['tps_avg']:.2f} tokens/s")
                peak_mem_mb = s.get('peak_memory_mb')
                if peak_mem_mb:
                    print(f"  Peak Memory: {peak_mem_mb:.2f} MB")

        except Exception as e:
            print(f"  EXCEPTION: {e}")
            all_results[size] = {"error": str(e)}

        # Sleep between runs to let device release resources
        if i < len(image_sizes) - 1:
            time.sleep(DEFAULT_SLEEP_BETWEEN_RUNS)

    # Save combined results
    combined_output = f"{args.output_dir}/all_benchmarks_combined.json"

    combined_data = {
        "config": config,
        "results_by_size": {},
    }

    for size in image_sizes:
        if size in all_results and "error" not in all_results[size]:
            combined_data["results_by_size"][size] = all_results[size]["summary"]
        elif size in all_results:
            combined_data["results_by_size"][size] = all_results[size]

    with open(combined_output, "w") as f:
        json.dump(combined_data, f, indent=2)

    # Save markdown summary table
    summary_md = generate_summary_table(all_results, image_sizes, config, compact=args.compact)
    summary_md_path = f"{args.output_dir}/summary.md"
    with open(summary_md_path, "w") as f:
        f.write(summary_md)

    # Print summary table
    print_summary_table(all_results, image_sizes, config, compact=args.compact)
    print(f"\nPer-size results:  {args.output_dir}/benchmark_<size>.json")
    print(f"Combined results:  {combined_output}")
    print(f"Summary table:     {summary_md_path}")


if __name__ == "__main__":
    main()