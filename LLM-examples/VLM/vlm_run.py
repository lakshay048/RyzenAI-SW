#!/usr/bin/env python
"""
Simple VLM inference script for quick model testing.

Usage:
    python vlm_run.py -m /path/to/model -i image.jpg
    python vlm_run.py -m /path/to/model -i image.jpg -p "What's in this image?"
    python vlm_run.py -m /path/to/model -i image.jpg --image_size 1024 1024
"""

import argparse
import json
import os
import tempfile
from pathlib import Path

from PIL import Image
import onnxruntime_genai as og


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
    """Build prompt using chat_template from the model directory."""
    model_dir = Path(model_path)
    tok_cfg_path = model_dir / "tokenizer_config.json"
    jinja_path = model_dir / "chat_template.jinja"

    template_str = None
    bos = None

    if tok_cfg_path.exists():
        with open(tok_cfg_path, "r", encoding="utf-8") as f:
            tok_cfg = json.load(f)
        template_str = tok_cfg.get("chat_template")
        bos = tok_cfg.get("bos_token")

    if not template_str and jinja_path.exists():
        with open(jinja_path, "r", encoding="utf-8") as f:
            template_str = f.read()

    if not template_str:
        raise RuntimeError(
            f"No chat template found at {model_path}. "
            f"Looked in tokenizer_config.json and chat_template.jinja."
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


def run_inference(args: argparse.Namespace):
    """Run VLM inference."""
    print(f"Loading model: {args.model_path}")
    config = og.Config(args.model_path)
    
    if args.execution_provider != "follow_config":
        config.clear_providers()
        if args.execution_provider != "cpu":
            config.append_provider(args.execution_provider)

    model = og.Model(config)
    model_type = model.type
    print(f"Model type: {model_type}")
    print(f"Device: {model.device_type}")

    processor = model.create_multimodal_processor()
    tokenizer = og.Tokenizer(model)

    # Validate image
    if not os.path.exists(args.image_path):
        raise FileNotFoundError(f"Image not found: {args.image_path}")

    # Handle image sizing
    original_size = get_image_size(args.image_path)
    print(f"Original image size: {original_size[0]}x{original_size[1]}")

    if args.image_size:
        target_size = tuple(args.image_size)
        print(f"Resizing to: {target_size[0]}x{target_size[1]}")
        image_path = resize_image(args.image_path, target_size)
    else:
        image_path = args.image_path

    images = og.Images.open(image_path)

    # Build prompt
    if model_type in ("phi3v", "phi4mm"):
        prompt = build_prompt_phi4([args.image_path], args.prompt)
    else:
        print(f"Using chat template from tokenizer_config.json")
        prompt = build_prompt_chat_template(args.model_path, tokenizer, [args.image_path], args.prompt)

    if args.show_prompt:
        print(f"\nPrompt:\n{prompt}\n")

    # Process inputs
    print("Processing inputs...")
    inputs = processor(prompt, images=images)

    # Setup generator
    params = og.GeneratorParams(model)
    params.set_search_options(max_length=args.max_length)

    generator = og.Generator(model, params)
    generator.set_inputs(inputs)

    # Generate
    print(f"\n{'='*60}")
    print("Output:")
    print(f"{'='*60}")

    stream = processor.create_stream()
    tokens_generated = 0

    while not generator.is_done() and tokens_generated < args.max_tokens:
        generator.generate_next_token()
        token = generator.get_next_tokens()[0]
        print(stream.decode(token), end="", flush=True)
        tokens_generated += 1

    print(f"\n{'='*60}")
    print(f"Tokens generated: {tokens_generated}")

    del generator


def main():
    parser = argparse.ArgumentParser(
        description="Simple VLM inference script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python vlm_run.py -m ./model -i image.jpg
  python vlm_run.py -m ./model -i image.jpg -p "What's in this image?"
  python vlm_run.py -m ./model -i image.jpg --image_size 1024 1024
        """
    )
    parser.add_argument(
        "-m", "--model_path",
        type=str,
        required=True,
        help="Path to the ONNX model folder"
    )
    parser.add_argument(
        "-i", "--image_path",
        type=str,
        required=True,
        help="Path to input image"
    )
    parser.add_argument(
        "-p", "--prompt",
        type=str,
        default="Describe this image in detail.",
        help="Prompt text (default: 'Describe this image in detail.')"
    )
    parser.add_argument(
        "-e", "--execution_provider",
        type=str,
        default="follow_config",
        choices=["cpu", "dml", "follow_config"],
        help="Execution provider (default: follow_config)"
    )
    parser.add_argument(
        "--image_size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        help="Resize image to this size (width height)"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=256,
        help="Max tokens to generate (default: 256)"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=16384,
        help="Max context length for KV cache (default: 16384)"
    )
    parser.add_argument(
        "--show_prompt",
        action="store_true",
        help="Print the full prompt before generation"
    )

    args = parser.parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()