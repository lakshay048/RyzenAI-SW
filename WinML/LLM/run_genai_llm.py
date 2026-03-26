#!/usr/bin/env python3
"""
LLM Runner script using Windows ML and ONNX GenAI on AMD NPU via VitisAI EP

This script runs Olive-converted LLM models using Windows ML and ONNX Runtime GenAI.
It's designed to work with AMD NPU devices and includes WinML integration.

Usage:
    python run_llm.py --model <path_to_model> --prompt "Your prompt"
    python run_llm.py --model <path_to_model> --interactive
    python run_llm.py --help

Example:
    python run_llm.py --model ./llama3_21binstruct --prompt "What is Python?"
    python run_llm.py --model ./llama3_21binstruct --interactive
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import onnxruntime_genai as og


def register_vitisai_ep():
    """
    Register VitisAI Execution Provider via WinML.

    Returns:
        bool: True if registration successful, False otherwise
    """
    try:
        # Import WinML components using the correct bootstrap module
        from winui3.microsoft.windows.applicationmodel.dynamicdependency.bootstrap import (
            InitializeOptions,
            initialize
        )
        import winui3.microsoft.windows.ai.machinelearning as winml

        # Initialize WinAppSDK
        # Use version with experimental suffix to match installed package name
        # Runtime version: 0.738.2207.0 (experimental4) or 0.770.2319.0 (experimental5)
        print("[INFO] Initializing WinAppSDK 2.0-experimental5...")
        wasdk_handle = initialize(
            version="2.0-experimental5",
            min_version="0.738.2207.0",
            options=InitializeOptions.ON_NO_MATCH_SHOW_UI
        )
        wasdk_handle.__enter__()
        print("[INFO] WinAppSDK 2.0-experimental5 initialized")

        # Get and ensure VitisAI EP
        catalog = winml.ExecutionProviderCatalog.get_default()
        providers = catalog.find_all_providers()

        # Print all available execution providers from WinML catalog
        print("\n[INFO] Available Execution Providers in WinML catalog:")
        print("       (Note: CPU EP is built-in to ONNX Runtime and not shown here)")
        for i, provider in enumerate(providers, 1):
            ready_status = provider.ready_state.name if hasattr(provider.ready_state, 'name') else str(provider.ready_state)
            print(f"  {i}. {provider.name} (Status: {ready_status})")
        print()

        vitisai_provider = None
        for provider in providers:
            if provider.name == "VitisAIExecutionProvider":
                vitisai_provider = provider
                break

        if not vitisai_provider:
            print("[WARNING] VitisAIExecutionProvider not found in catalog")
            return False

        # Ensure the provider is ready
        if vitisai_provider.ready_state != winml.ExecutionProviderReadyState.READY:
            print(f"[INFO] Ensuring VitisAIExecutionProvider (state: {vitisai_provider.ready_state})...")
            result = vitisai_provider.ensure_ready_async().get()
            if result.status != winml.ExecutionProviderReadyResultState.SUCCESS:
                print(f"[ERROR] Failed to ensure VitisAIExecutionProvider: {result.status}")
                return False

        print("[INFO] VitisAIExecutionProvider is ready")

        # Register to ONNX Runtime GenAI
        if vitisai_provider.library_path:
            og.register_execution_provider_library(vitisai_provider.name, vitisai_provider.library_path)
            print(f"[INFO] Registered {vitisai_provider.name} to ONNX GenAI")
            print(f"[INFO] Library path: {vitisai_provider.library_path}")
            return True
        else:
            print("[WARNING] VitisAIExecutionProvider has no library path")
            return False

    except ImportError as e:
        print(f"[ERROR] Failed to import WinML components: {e}")
        print("[INFO] Make sure you're running on Windows with WinML support")
        return False
    except Exception as e:
        print(f"[ERROR] Failed to register VitisAI EP: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_inference(
    model_path: Path,
    prompt: str,
    use_vitisai: bool = True,
    max_length: int = 512
) -> None:
    """
    Run single inference on the given prompt.

    Args:
        model_path: Path to the ONNX model directory
        prompt: Text prompt for the model
        use_vitisai: Whether to use VitisAI EP (default: True)
        max_length: Maximum generation length
    """
    print(f"\n{'='*60}")
    print(f"Loading model from: {model_path}")
    print(f"{'='*60}")

    # Create model configuration
    try:
        config = og.Config(str(model_path))

        if not use_vitisai:
            # Force CPU EP
            config.clear_providers()
            config.append_provider("CPU")
            print("[INFO] Using CPU Execution Provider")
        else:
            print("[INFO] Using VitisAI Execution Provider (configured in genai_config.json)")

        # Load model
        print("[INFO] Loading model (this may take a minute)...")
        model = og.Model(config)
        print("[INFO] ✓ Model loaded successfully")

        # Create tokenizer
        print("[INFO] Creating tokenizer...")
        tokenizer = og.Tokenizer(model)
        print("[INFO] ✓ Tokenizer created")

        # Create generator params
        params = og.GeneratorParams(model)
        params.set_search_options(max_length=max_length)

        # Tokenize input
        input_tokens = tokenizer.encode(prompt)
        print(f"[INFO] Tokenized prompt to {len(input_tokens)} tokens")

        # Run inference with streaming
        print(f"\n{'='*60}")
        print(f"Prompt: {prompt}")
        print(f"{'='*60}")
        print("Response: ", end="", flush=True)

        generator = og.Generator(model, params)
        generator.append_tokens(input_tokens)

        tokenizer_stream = tokenizer.create_stream()

        while not generator.is_done():
            generator.generate_next_token()
            new_tokens = generator.get_next_tokens()
            for token in new_tokens:
                text_chunk = tokenizer_stream.decode(token)
                print(text_chunk, end="", flush=True)

        print(f"\n{'='*60}\n")

    except Exception as e:
        print(f"\n[ERROR] Inference failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def interactive_mode(
    model_path: Path,
    use_vitisai: bool = True,
    max_length: int = 512
) -> None:
    """
    Run model in interactive mode, allowing multiple prompts.

    Args:
        model_path: Path to the ONNX model directory
        use_vitisai: Whether to use VitisAI EP (default: True)
        max_length: Maximum generation length
    """
    print(f"\n{'='*60}")
    print(f"Loading model from: {model_path}")
    print(f"{'='*60}")

    # Load model once
    try:
        config = og.Config(str(model_path))

        if not use_vitisai:
            config.clear_providers()
            config.append_provider("CPU")
            print("[INFO] Using CPU Execution Provider")
        else:
            print("[INFO] Using VitisAI Execution Provider (configured in genai_config.json)")

        print("[INFO] Loading model (this may take a minute)...")
        model = og.Model(config)
        print("[INFO] ✓ Model loaded successfully")

        print("[INFO] Creating tokenizer...")
        tokenizer = og.Tokenizer(model)
        tokenizer_stream = tokenizer.create_stream()
        print("[INFO] ✓ Tokenizer created")

    except Exception as e:
        print(f"\n[ERROR] Model initialization failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Interactive loop
    print(f"\n{'='*60}")
    print("Interactive Mode")
    print(f"{'='*60}")
    print("Type your prompt and press Enter")
    print("Type 'quit', 'exit', or 'q' to exit")
    print("Press Enter without text for default prompt")
    print(f"{'='*60}\n")

    while True:
        try:
            prompt = input("Prompt: ").strip()

            if prompt.lower() in ['quit', 'exit', 'q']:
                print("\nExiting...")
                break

            if not prompt:
                prompt = "What is an AI accelerator?"
                print(f"  (Using default: {prompt})")

            # Create fresh generator for each prompt
            params = og.GeneratorParams(model)
            params.set_search_options(max_length=max_length)

            input_tokens = tokenizer.encode(prompt)
            generator = og.Generator(model, params)
            generator.append_tokens(input_tokens)

            print("\nResponse: ", end="", flush=True)

            while not generator.is_done():
                generator.generate_next_token()
                new_tokens = generator.get_next_tokens()
                for token in new_tokens:
                    text_chunk = tokenizer_stream.decode(token)
                    print(text_chunk, end="", flush=True)

            print(f"\n{'-'*60}\n")

        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"\n[ERROR] Inference failed: {e}")
            import traceback
            traceback.print_exc()
            print()


def main():
    parser = argparse.ArgumentParser(
        description="Run ONNX GenAI LLM models on AMD NPU using VitisAI EP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single prompt inference (uses VitisAI EP / AMD NPU by default)
  python run_llm.py --model ./llama3_21binstruct --prompt "What is AI?"

  # Interactive mode
  python run_llm.py --model ./llama3_21binstruct --interactive

  # Use CPU instead of NPU
  python run_llm.py --model ./llama3_21binstruct --cpu --prompt "Hello"

  # Longer generation
  python run_llm.py --model ./llama3_21binstruct --max-length 1024 --interactive
"""
    )

    parser.add_argument(
        "--model", "-m",
        type=Path,
        required=True,
        help="Path to the ONNX model directory"
    )

    parser.add_argument(
        "--prompt", "-p",
        type=str,
        help="Text prompt to send to the model"
    )

    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode (ignores --prompt)"
    )

    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Use CPU instead of VitisAI/NPU"
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum generation length (default: 512)"
    )

    args = parser.parse_args()

    # Validate model path
    if not args.model.exists():
        print(f"[ERROR] Model path does not exist: {args.model}")
        sys.exit(1)

    # Check for genai_config.json
    genai_config = args.model / "genai_config.json"
    if not genai_config.exists():
        print(f"[ERROR] genai_config.json not found in model directory: {args.model}")
        sys.exit(1)

    use_vitisai = not args.cpu

    # Register VitisAI EP once if requested
    if use_vitisai:
        print(f"\n{'='*60}")
        print("Registering Execution Providers")
        print(f"{'='*60}")
        if not register_vitisai_ep():
            print("[WARNING] Failed to register VitisAI EP, falling back to CPU")
            use_vitisai = False

    # Run in appropriate mode
    if args.interactive:
        interactive_mode(args.model, use_vitisai=use_vitisai, max_length=args.max_length)
    elif args.prompt:
        run_inference(args.model, args.prompt, use_vitisai=use_vitisai, max_length=args.max_length)
    else:
        # Default: run with a demo prompt
        default_prompt = "Explain what an AMD NPU is in one sentence."
        print("[INFO] No prompt provided, using demo prompt")
        run_inference(args.model, default_prompt, use_vitisai=use_vitisai, max_length=args.max_length)


if __name__ == "__main__":
    main()
