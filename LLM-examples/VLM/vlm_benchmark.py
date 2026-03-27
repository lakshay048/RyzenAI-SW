# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
VLM Benchmark Script for onnxruntime-genai

Measures:
- TTFT (Time To First Token): Time from inputs ready to first token generated
- TPS (Tokens Per Second): Token generation throughput
- Preprocessing Time: Time spent in multimodal processor (image preprocessing)
- Image Token Count: Number of tokens generated from the image by the vision encoder
"""

import argparse
import json
import os
import statistics
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import psutil
from PIL import Image

import onnxruntime_genai as og


def get_peak_memory_bytes() -> int:
    """Get peak memory usage (working set) in bytes."""
    process = psutil.Process()
    return process.memory_info().rss


@dataclass
class BenchmarkResult:
    """Stores benchmark measurements for a single run."""
    preprocessing_time_ms: float = 0.0
    ttft_ms: float = 0.0
    token_gen_times_ms: list = field(default_factory=list)
    total_tokens_generated: int = 0
    prompt_tokens: int = 0
    image_tokens: int = 0
    text_tokens: int = 0


@dataclass
class BenchmarkStats:
    """Aggregated statistics across multiple runs."""
    avg: float = 0.0
    std: float = 0.0
    p50: float = 0.0
    p90: float = 0.0
    p99: float = 0.0
    min_val: float = 0.0
    max_val: float = 0.0
    n: int = 0


def compute_stats(measurements: list) -> BenchmarkStats:
    """Compute statistics from a list of measurements."""
    if not measurements:
        return BenchmarkStats()

    sorted_m = sorted(measurements)
    n = len(sorted_m)

    stats = BenchmarkStats(
        avg=statistics.mean(sorted_m),
        std=statistics.stdev(sorted_m) if n > 1 else 0.0,
        p50=sorted_m[int(n * 0.5)] if n > 0 else 0.0,
        p90=sorted_m[int(n * 0.9)] if n > 0 else 0.0,
        p99=sorted_m[int(n * 0.99)] if n > 0 else 0.0,
        min_val=min(sorted_m),
        max_val=max(sorted_m),
        n=n
    )
    return stats


def resize_image(image_path: str, size: tuple) -> str:
    """Resize image to target size and save to temp file."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize(size)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    img.save(tmp.name, format="JPEG")
    return tmp.name


def get_image_size(image_path: str) -> tuple:
    """Get original image dimensions."""
    with Image.open(image_path) as img:
        return img.size


def build_prompt_phi4(image_paths: list, text: str) -> str:
    """Build prompt for phi-4 multimodal model."""
    prompt = "<|user|>\n"
    for i in range(len(image_paths)):
        prompt += f"<|image_{i + 1}|>\n"
    prompt += f"{text}<|end|>\n<|assistant|>\n"
    return prompt


def build_prompt_chat_template(model_path: str, tokenizer: og.Tokenizer,
                               image_paths: list, text: str) -> str:
    """Build prompt using chat_template from the model directory.

    Looks for the template in two locations (in order):
      1. "chat_template" field inside tokenizer_config.json (inline)
      2. chat_template.jinja file (common after ONNX/HuggingFace export)
    """
    model_dir = Path(model_path)
    tok_cfg_path = model_dir / "tokenizer_config.json"
    jinja_path = model_dir / "chat_template.jinja"

    template_str = None
    bos = None

    # 1. Try inline chat_template in tokenizer_config.json
    if tok_cfg_path.exists():
        with open(tok_cfg_path, "r", encoding="utf-8") as f:
            tok_cfg = json.load(f)
        template_str = tok_cfg.get("chat_template")
        bos = tok_cfg.get("bos_token")

    # 2. Fall back to chat_template.jinja file
    if not template_str and jinja_path.exists():
        with open(jinja_path, "r", encoding="utf-8") as f:
            template_str = f.read()
        print(f"  Loaded chat template from: {jinja_path}")

    if not template_str:
        raise RuntimeError(
            f"No chat template found at {model_path}. "
            f"Looked in tokenizer_config.json and chat_template.jinja. "
            f"Please add a prompt builder for this model type."
        )

    if not bos:
        template_str = template_str.replace("{{ bos_token }}", "")

    content_list = [{"type": "image"} for _ in image_paths]
    content_list.append({"type": "text", "text": text})
    messages = [{"role": "user", "content": content_list}]

    message_json = json.dumps(messages)
    prompt = tokenizer.apply_chat_template(message_json, template_str=template_str,
                                           add_generation_prompt=True)
    return prompt


def count_text_tokens(tokenizer: og.Tokenizer, prompt: str) -> int:
    """Count text-only tokens in the prompt (without image tokens).

    Encodes the raw prompt string to get the text token count.
    Image tokens are added by the multimodal processor on top of this.
    """
    try:
        text_ids = tokenizer.encode(prompt)
        return len(text_ids)
    except Exception:
        return 0


def run_single_benchmark(
    model: og.Model,
    processor: og.MultiModalProcessor,
    prompt: str,
    images: og.Images,
    max_tokens: int,
    max_length: int,
    tokenizer: Optional[og.Tokenizer] = None,
    verbose: bool = False
) -> tuple:
    """Run a single benchmark iteration and collect timing metrics.

    Returns:
        tuple: (BenchmarkResult, generated_text) where generated_text is only
               populated if verbose=True
    """
    result = BenchmarkResult()
    generated_tokens = []

    # Measure preprocessing time (multimodal processor call)
    preprocess_start = time.perf_counter()
    inputs = processor(prompt, images=images)
    preprocess_end = time.perf_counter()
    result.preprocessing_time_ms = (preprocess_end - preprocess_start) * 1000

    # Get prompt token count from inputs (includes image tokens post-ViT)
    if "input_ids" in inputs.keys():
        input_ids = inputs["input_ids"]
        result.prompt_tokens = input_ids.shape()[1] if len(input_ids.shape()) > 1 else input_ids.shape()[0]

    # Compute image token count = total prompt tokens - text-only tokens
    if tokenizer is not None:
        result.text_tokens = count_text_tokens(tokenizer, prompt)
        result.image_tokens = max(0, result.prompt_tokens - result.text_tokens)

    # Setup generator
    params = og.GeneratorParams(model)
    params.set_search_options(max_length=max_length)

    generator = og.Generator(model, params)

    # Create stream for decoding tokens if verbose
    stream = processor.create_stream() if verbose else None

    # Measure TTFT: includes set_inputs (prefill) + first token generation
    ttft_start = time.perf_counter()
    generator.set_inputs(inputs)
    generator.generate_next_token()
    ttft_end = time.perf_counter()
    result.ttft_ms = (ttft_end - ttft_start) * 1000
    result.total_tokens_generated = 1

    # Capture first token
    if verbose:
        first_token = generator.get_next_tokens()[0]
        generated_tokens.append(stream.decode(first_token))

    if generator.is_done():
        del generator
        generated_text = "".join(generated_tokens) if verbose else ""
        return result, generated_text

    # Measure subsequent token generation times
    while not generator.is_done() and result.total_tokens_generated < max_tokens:
        token_start = time.perf_counter()
        generator.generate_next_token()
        token_end = time.perf_counter()
        result.token_gen_times_ms.append((token_end - token_start) * 1000)
        result.total_tokens_generated += 1

        # Capture token for verbose output
        if verbose:
            new_token = generator.get_next_tokens()[0]
            generated_tokens.append(stream.decode(new_token))

    del generator
    generated_text = "".join(generated_tokens) if verbose else ""
    return result, generated_text


def run_warmup(
    model: og.Model,
    processor: og.MultiModalProcessor,
    prompt: str,
    images: og.Images,
    max_tokens: int,
    max_length: int,
    num_iterations: int,
    verbose: bool = False
):
    """Run warmup iterations."""
    if verbose:
        print(f"Running {num_iterations} warmup iteration(s)...")

    for i in range(num_iterations):
        inputs = processor(prompt, images=images)
        params = og.GeneratorParams(model)
        params.set_search_options(max_length=max_length)

        generator = og.Generator(model, params)
        generator.set_inputs(inputs)

        tokens_generated = 0
        stream = processor.create_stream()

        while not generator.is_done() and tokens_generated < max_tokens:
            generator.generate_next_token()
            tokens_generated += 1

            if verbose and i == 0:
                new_token = generator.get_next_tokens()[0]
                print(stream.decode(new_token), end="", flush=True)

        if verbose and i == 0:
            print("\n")

        del generator


def print_benchmark_results(
    results: list,
    image_size: tuple,
    model_type: str,
    peak_memory_bytes: int
):
    """Print formatted benchmark results."""
    print("\n" + "=" * 60)
    print("VLM BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Model Type: {model_type}")
    print(f"Image Size: {image_size[0]}x{image_size[1]}")
    print(f"Number of runs: {len(results)}")
    print("-" * 60)

    # Token breakdown (from first run)
    if results and results[0].prompt_tokens > 0:
        r0 = results[0]
        print(f"\nToken Breakdown:")
        print(f"  Total Prompt Tokens: {r0.prompt_tokens}")
        print(f"  Text Tokens:         {r0.text_tokens}")
        print(f"  Image Tokens:        {r0.image_tokens}")

    # Preprocessing Time
    preprocess_times = [r.preprocessing_time_ms for r in results]
    preprocess_stats = compute_stats(preprocess_times)
    print(f"\nPreprocessing Time (image processing):")
    print(f"  Average:  {preprocess_stats.avg:>10.2f} ms")
    print(f"  Std Dev:  {preprocess_stats.std:>10.2f} ms")
    print(f"  P50:      {preprocess_stats.p50:>10.2f} ms")
    print(f"  P90:      {preprocess_stats.p90:>10.2f} ms")
    print(f"  P99:      {preprocess_stats.p99:>10.2f} ms")
    print(f"  Min:      {preprocess_stats.min_val:>10.2f} ms")
    print(f"  Max:      {preprocess_stats.max_val:>10.2f} ms")

    # TTFT
    ttft_times = [r.ttft_ms for r in results]
    ttft_stats = compute_stats(ttft_times)
    print(f"\nTime To First Token (TTFT):")
    print(f"  Average:  {ttft_stats.avg:>10.2f} ms")
    print(f"  Std Dev:  {ttft_stats.std:>10.2f} ms")
    print(f"  P50:      {ttft_stats.p50:>10.2f} ms")
    print(f"  P90:      {ttft_stats.p90:>10.2f} ms")
    print(f"  P99:      {ttft_stats.p99:>10.2f} ms")
    print(f"  Min:      {ttft_stats.min_val:>10.2f} ms")
    print(f"  Max:      {ttft_stats.max_val:>10.2f} ms")

    # Prefill throughput (tokens/s)
    if results and results[0].prompt_tokens > 0:
        prefill_tps_list = []
        for r in results:
            if r.ttft_ms > 0:
                prefill_tps_list.append(r.prompt_tokens / (r.ttft_ms / 1000.0))
        if prefill_tps_list:
            prefill_tps_stats = compute_stats(prefill_tps_list)
            print(f"\nPrefill Throughput:")
            print(f"  Average:  {prefill_tps_stats.avg:>10.2f} tokens/sec")
            print(f"  P50:      {prefill_tps_stats.p50:>10.2f} tokens/sec")

    # Token Generation (TPS)
    all_token_times = []
    for r in results:
        all_token_times.extend(r.token_gen_times_ms)

    if all_token_times:
        token_stats = compute_stats(all_token_times)
        avg_tps = 1000.0 / token_stats.avg if token_stats.avg > 0 else 0
        print(f"\nToken Generation (excluding first token):")
        print(f"  Average:  {token_stats.avg:>10.2f} ms/token")
        print(f"  TPS:      {avg_tps:>10.2f} tokens/sec")
        print(f"  Std Dev:  {token_stats.std:>10.2f} ms")
        print(f"  P50:      {token_stats.p50:>10.2f} ms/token ({1000.0/token_stats.p50:.2f} TPS)" if token_stats.p50 > 0 else "  P50:      N/A")
        print(f"  P90:      {token_stats.p90:>10.2f} ms/token ({1000.0/token_stats.p90:.2f} TPS)" if token_stats.p90 > 0 else "  P90:      N/A")
        print(f"  Min:      {token_stats.min_val:>10.2f} ms/token")
        print(f"  Max:      {token_stats.max_val:>10.2f} ms/token")

    # Total tokens
    total_tokens = sum(r.total_tokens_generated for r in results)
    avg_tokens = total_tokens / len(results) if results else 0
    print(f"\nTokens Generated:")
    print(f"  Average per run: {avg_tokens:.1f}")
    print(f"  Total:           {total_tokens}")

    # E2E metrics
    e2e_times = [r.preprocessing_time_ms + r.ttft_ms + sum(r.token_gen_times_ms) for r in results]
    e2e_stats = compute_stats(e2e_times)
    print(f"\nEnd-to-End Time (Preprocessing + TTFT + Token Gen):")
    print(f"  Average:  {e2e_stats.avg:>10.2f} ms")
    print(f"  P50:      {e2e_stats.p50:>10.2f} ms")

    # Peak memory
    peak_memory_mb = peak_memory_bytes / (1024 * 1024)
    peak_memory_gb = peak_memory_bytes / (1024 * 1024 * 1024)
    print(f"\nPeak Memory Usage:")
    print(f"  {peak_memory_mb:>10.2f} MB ({peak_memory_gb:.2f} GB)")

    print("=" * 60 + "\n")


def export_results_json(results: list, output_path: str, config: dict, 
                        peak_memory_bytes: int):
    """Export benchmark results to JSON file."""
    preprocess_times = [r.preprocessing_time_ms for r in results]
    ttft_times = [r.ttft_ms for r in results]
    all_token_times = []
    for r in results:
        all_token_times.extend(r.token_gen_times_ms)

    preprocess_stats = compute_stats(preprocess_times)
    ttft_stats = compute_stats(ttft_times)
    token_stats = compute_stats(all_token_times) if all_token_times else BenchmarkStats()

    # Prefill throughput
    prefill_tps_list = []
    for r in results:
        if r.ttft_ms > 0 and r.prompt_tokens > 0:
            prefill_tps_list.append(r.prompt_tokens / (r.ttft_ms / 1000.0))
    prefill_tps_stats = compute_stats(prefill_tps_list) if prefill_tps_list else BenchmarkStats()

    # Token counts from first run
    first_run = results[0] if results else BenchmarkResult()

    export_data = {
        "config": config,
        "summary": {
            "token_breakdown": {
                "prompt_tokens": first_run.prompt_tokens,
                "text_tokens": first_run.text_tokens,
                "image_tokens": first_run.image_tokens,
            },
            "preprocessing_ms": {
                "avg": preprocess_stats.avg,
                "std": preprocess_stats.std,
                "p50": preprocess_stats.p50,
                "p90": preprocess_stats.p90,
                "p99": preprocess_stats.p99,
                "min": preprocess_stats.min_val,
                "max": preprocess_stats.max_val,
            },
            "ttft_ms": {
                "avg": ttft_stats.avg,
                "std": ttft_stats.std,
                "p50": ttft_stats.p50,
                "p90": ttft_stats.p90,
                "p99": ttft_stats.p99,
                "min": ttft_stats.min_val,
                "max": ttft_stats.max_val,
            },
            "prefill_throughput_tps": {
                "avg": prefill_tps_stats.avg,
                "p50": prefill_tps_stats.p50,
            },
            "token_generation_ms": {
                "avg": token_stats.avg,
                "std": token_stats.std,
                "p50": token_stats.p50,
                "p90": token_stats.p90,
                "tps_avg": 1000.0 / token_stats.avg if token_stats.avg > 0 else 0,
            },
            "total_runs": len(results),
            "avg_tokens_per_run": sum(r.total_tokens_generated for r in results) / len(results) if results else 0,
            "peak_memory_bytes": peak_memory_bytes,
            "peak_memory_mb": peak_memory_bytes / (1024 * 1024),
        },
        "raw_measurements": {
            "preprocessing_times_ms": preprocess_times,
            "ttft_times_ms": ttft_times,
            "token_gen_times_ms": all_token_times,
        }
    }

    with open(output_path, "w") as f:
        json.dump(export_data, f, indent=2)
    print(f"Results exported to: {output_path}")


def run_benchmark(args: argparse.Namespace):
    """Main benchmark function."""
    print("Loading model...")
    config = og.Config(args.model_path)
    if args.execution_provider != "follow_config":
        config.clear_providers()
        if args.execution_provider != "cpu":
            print(f"Setting execution provider to {args.execution_provider}...")
            config.append_provider(args.execution_provider)

    model = og.Model(config)
    model_type = model.type
    print(f"Model loaded: {model_type}")
    print(f"Device: {model.device_type}")

    processor = model.create_multimodal_processor()
    tokenizer = og.Tokenizer(model)

    # Validate image path
    if not args.image_path or not os.path.exists(args.image_path):
        raise FileNotFoundError(f"Image file not found: {args.image_path}")

    # Get original image size
    original_size = get_image_size(args.image_path)
    print(f"Original image size: {original_size[0]}x{original_size[1]}")

    # Determine target image size
    if args.image_size:
        target_size = tuple(args.image_size)
    else:
        target_size = original_size
    print(f"Benchmark image size: {target_size[0]}x{target_size[1]}")

    # Resize image if needed
    if target_size != original_size:
        print(f"Resizing image to {target_size[0]}x{target_size[1]}...")
        resized_path = resize_image(args.image_path, target_size)
        images = og.Images.open(resized_path)
    else:
        images = og.Images.open(args.image_path)

    # Build prompt based on model type
    prompt_text = args.prompt if args.prompt else "Describe this image in detail."

    if model_type in ("phi3v", "phi4mm"):
        prompt = build_prompt_phi4([args.image_path], prompt_text)
    else:
        # All other models: use chat_template from tokenizer_config.json
        # Works for Gemma, Qwen-VL, InternVL, LLaVA-Next, etc.
        print(f"Model type: '{model_type}' — using chat template from tokenizer_config.json")
        prompt = build_prompt_chat_template(args.model_path, tokenizer, [args.image_path], prompt_text)

    if args.verbose:
        print(f"\nPrompt: {prompt[:200]}...")

    # Run warmup
    run_warmup(
        model=model,
        processor=processor,
        prompt=prompt,
        images=images,
        max_tokens=args.max_tokens,
        max_length=args.max_length,
        num_iterations=args.warmup_iterations,
        verbose=args.verbose
    )

    # Run benchmark iterations
    print(f"\nRunning {args.num_iterations} benchmark iteration(s)...")
    results = []

    for i in range(args.num_iterations):
        if args.verbose:
            print(f"  Iteration {i + 1}/{args.num_iterations}...", end=" ")

        result, generated_text = run_single_benchmark(
            model=model,
            processor=processor,
            prompt=prompt,
            images=images,
            max_tokens=args.max_tokens,
            max_length=args.max_length,
            tokenizer=tokenizer,
            verbose=args.verbose
        )
        results.append(result)

        # Print token breakdown on first iteration
        if i == 0:
            print(f"  Token breakdown: total={result.prompt_tokens} "
                  f"(text={result.text_tokens}, image={result.image_tokens})")

        if args.verbose:
            print(f"  Iteration {i + 1}: "
                  f"Preprocess: {result.preprocessing_time_ms:.1f}ms, "
                  f"TTFT: {result.ttft_ms:.1f}ms, "
                  f"Tokens: {result.total_tokens_generated}")
            # Print generated text on first iteration
            if i == 0 and generated_text:
                print(f"\n  Generated output:\n  {'-'*50}")
                print(f"  {generated_text}")
                print(f"  {'-'*50}\n")

    # Get peak memory usage
    peak_memory_bytes = get_peak_memory_bytes()

    # Print results
    print_benchmark_results(results, target_size, model_type, peak_memory_bytes)

    # Export to JSON if requested
    if args.output_json:
        export_config = {
            "model_path": args.model_path,
            "model_type": model_type,
            "device_type": model.device_type,
            "execution_provider": args.execution_provider,
            "image_path": args.image_path,
            "image_size": list(target_size),
            "prompt": prompt_text,
            "max_tokens": args.max_tokens,
            "max_length": args.max_length,
            "num_iterations": args.num_iterations,
            "warmup_iterations": args.warmup_iterations,
        }
        export_results_json(results, args.output_json, export_config, peak_memory_bytes)


def main():
    parser = argparse.ArgumentParser(
        description="VLM Benchmark Script - Measure TTFT, TPS, and preprocessing time"
    )
    parser.add_argument(
        "-m", "--model_path",
        type=str,
        required=True,
        help="Path to the folder containing the ONNX model"
    )
    parser.add_argument(
        "-i", "--image_path",
        type=str,
        required=True,
        help="Path to the input image"
    )
    parser.add_argument(
        "-e", "--execution_provider",
        type=str,
        default="follow_config",
        choices=["cpu", "dml", "follow_config"],
        help="Execution provider (default: follow_config)"
    )
    parser.add_argument(
        "-p", "--prompt",
        type=str,
        default="Describe this image in detail.",
        help="Prompt text for the VLM"
    )
    parser.add_argument(
        "--image_size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        help="Target image size (width height). If not specified, uses original size."
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=100,
        help="Maximum number of tokens to generate (default: 100)"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=16384,
        help="Max context length for KV cache (default: 16384)"
    )
    parser.add_argument(
        "-n", "--num_iterations",
        type=int,
        default=5,
        help="Number of benchmark iterations (default: 5)"
    )
    parser.add_argument(
        "-w", "--warmup_iterations",
        type=int,
        default=2,
        help="Number of warmup iterations (default: 2)"
    )
    parser.add_argument(
        "-o", "--output_json",
        type=str,
        help="Path to export results as JSON"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    args = parser.parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()