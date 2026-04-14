#!/usr/bin/env python3
"""
Parakeet ASR Server - OpenAI Whisper-compatible API.

A speech-to-text server using NVIDIA's Parakeet TDT 0.6B model (ONNX)
with support for AMD Ryzen AI NPU acceleration via VitisAI EP.

Usage:
    # CPU mode
    python server.py --device cpu --port 5092

    # NPU mode (requires ryzen-ai conda environment)
    conda activate ryzen-ai-1.7.0
    python server.py --device npu --port 5092
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse

from inference import Transcriber

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("parakeet")

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Parakeet ASR Server",
    description=(
        "OpenAI Whisper-compatible speech recognition API using "
        "NVIDIA Parakeet TDT 0.6B (ONNX). Supports AMD Ryzen AI NPU acceleration."
    ),
    version="1.0.0",
)

# Global transcriber instance (set during startup)
transcriber: Transcriber = None


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/v1/models")
async def list_models():
    """
    List available models.
    Returns parakeet-tdt-0.6b and whisper-1 (alias for compatibility).
    """
    info = transcriber.get_info() if transcriber else {}
    return {
        "object": "list",
        "data": [
            {
                "id": "parakeet-tdt-0.6b",
                "object": "model",
                "created": 1700000000,
                "owned_by": "nvidia",
                "device": info.get("device", "unknown"),
                "providers": info.get("encoder_providers", []),
            },
            {
                "id": "whisper-1",
                "object": "model",
                "created": 1700000000,
                "owned_by": "nvidia",
                "description": "Alias for parakeet-tdt-0.6b (Whisper API compatibility)",
                "device": info.get("device", "unknown"),
            },
        ],
    }


@app.get("/v1/info")
async def info():
    """Return detailed information about the transcriber configuration."""
    if transcriber is None:
        raise HTTPException(status_code=503, detail="Transcriber not initialized")
    return transcriber.get_info()


@app.post("/v1/audio/transcriptions")
async def transcribe_audio(
    file: UploadFile = File(..., description="Audio file (WAV format, max 25MB)"),
    model: str = Form(default="parakeet-tdt-0.6b", description="Model name (accepted but ignored)"),
    language: str = Form(default="en", description="ISO-639-1 language code"),
    response_format: str = Form(
        default="json",
        description="Output format: json, text, srt, vtt, verbose_json",
    ),
    prompt: str = Form(default="", description="Optional prompt (accepted but ignored)"),
    temperature: float = Form(default=0.0, description="Temperature (accepted but ignored)"),
):
    """
    Transcribe audio to text. Compatible with OpenAI's Whisper API.

    Accepts WAV audio files and returns transcribed text in the requested format.
    """
    if transcriber is None:
        raise HTTPException(status_code=503, detail="Transcriber not initialized")

    # Read uploaded file
    try:
        audio_data = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read uploaded file: {e}")

    if len(audio_data) > 25 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large (max 25MB)")

    if len(audio_data) == 0:
        raise HTTPException(status_code=400, detail="Empty file")

    # Determine format from filename
    filename = file.filename or "audio.wav"
    ext = Path(filename).suffix.lower()
    if not ext:
        ext = ".wav"

    # Transcribe
    try:
        start_time = time.perf_counter()
        text = transcriber.transcribe(audio_data, audio_format=ext, language=language)
        elapsed = time.perf_counter() - start_time
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Transcription failed")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")

    # Estimate audio duration from file size (rough for WAV)
    # Better: parse from WAV header, but this is sufficient for the API response
    duration_estimate = len(audio_data) / (16000 * 2)  # assume 16kHz 16-bit mono

    # Format response
    if response_format == "text":
        return PlainTextResponse(content=text)

    elif response_format == "srt":
        srt = _format_srt(text, duration_estimate)
        return PlainTextResponse(content=srt, media_type="text/plain")

    elif response_format == "vtt":
        vtt = _format_vtt(text, duration_estimate)
        return PlainTextResponse(content=vtt, media_type="text/vtt")

    elif response_format == "verbose_json":
        return JSONResponse(
            content={
                "task": "transcribe",
                "language": language,
                "duration": round(duration_estimate, 2),
                "text": text,
                "segments": [
                    {
                        "id": 0,
                        "start": 0.0,
                        "end": round(duration_estimate, 2),
                        "text": text,
                    }
                ],
                "processing_time": round(elapsed, 3),
                "device": transcriber.device,
            }
        )

    else:  # json (default)
        return JSONResponse(content={"text": text})


# ---------------------------------------------------------------------------
# Subtitle format helpers
# ---------------------------------------------------------------------------
def _format_timestamp_srt(seconds: float) -> str:
    """Format seconds as SRT timestamp: HH:MM:SS,mmm"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _format_timestamp_vtt(seconds: float) -> str:
    """Format seconds as VTT timestamp: HH:MM:SS.mmm"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def _format_srt(text: str, duration: float) -> str:
    """Format text as SRT subtitles."""
    return (
        f"1\n"
        f"{_format_timestamp_srt(0)} --> {_format_timestamp_srt(duration)}\n"
        f"{text}\n"
    )


def _format_vtt(text: str, duration: float) -> str:
    """Format text as WebVTT subtitles."""
    return (
        f"WEBVTT\n\n"
        f"{_format_timestamp_vtt(0)} --> {_format_timestamp_vtt(duration)}\n"
        f"{text}\n"
    )


# ---------------------------------------------------------------------------
# CLI & Startup
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Parakeet ASR Server - OpenAI Whisper-compatible API with Ryzen AI NPU support"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5092,
        help="Server port (default: 5092)",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Server host (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--models-dir",
        default="./models",
        help="Path to models directory (default: ./models)",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "npu", "gpu"],
        default="cpu",
        help="Execution device: cpu or npu (default: cpu)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


def main():
    global transcriber

    args = parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Print banner
    print()
    print("=" * 60)
    print("  Parakeet ASR Server")
    print("  NVIDIA Parakeet TDT 0.6B (ONNX)")
    print(f"  Device: {args.device.upper()}")
    print("=" * 60)
    print()

    # Check models directory
    models_path = Path(args.models_dir)
    if not models_path.exists():
        logger.error(
            "Models directory not found: %s\n"
            "Run: python download_models.py --output-dir %s",
            models_path,
            models_path,
        )
        sys.exit(1)

    # Initialize transcriber
    logger.info("Initializing transcriber (device=%s)...", args.device)
    try:
        transcriber = Transcriber(
            models_dir=args.models_dir,
            device=args.device,
            debug=args.debug,
        )
    except Exception as e:
        logger.exception("Failed to initialize transcriber")
        sys.exit(1)

    info = transcriber.get_info()
    logger.info("ONNX Runtime version: %s", info.get("onnxruntime_version", "unknown"))
    logger.info("Available providers: %s", info.get("available_providers", []))
    logger.info("Encoder providers: %s", info.get("encoder_providers", []))
    logger.info("Decoder providers: %s", info.get("decoder_providers", []))

    # Start server
    logger.info("Starting server on %s:%d", args.host, args.port)
    print()
    print(f"  API: http://localhost:{args.port}/v1/audio/transcriptions")
    print(f"  Health: http://localhost:{args.port}/health")
    print(f"  Models: http://localhost:{args.port}/v1/models")
    print(f"  Info: http://localhost:{args.port}/v1/info")
    print()

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
