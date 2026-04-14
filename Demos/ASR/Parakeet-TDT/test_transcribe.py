#!/usr/bin/env python3
"""
Quick test / benchmark script for the Parakeet transcriber.

Usage:
    # Test with a WAV file on CPU
    python test_transcribe.py audio.wav --device cpu

    # Test on NPU
    conda activate ryzen-ai-1.7.0
    python test_transcribe.py audio.wav --device npu

    # Benchmark: compare CPU vs NPU (run twice with different --device)
    python test_transcribe.py audio.wav --device cpu
    python test_transcribe.py audio.wav --device npu

    # Generate a test WAV file (requires numpy only)
    python test_transcribe.py --generate-test-wav
"""

import argparse
import logging
import struct
import sys
import time

import numpy as np


def generate_test_wav(path: str = "test_silence.wav", duration_sec: float = 3.0):
    """Generate a silent WAV file for smoke-testing the pipeline."""
    sample_rate = 16000
    num_samples = int(sample_rate * duration_sec)
    # Generate very quiet noise (not pure silence, which would be trivial)
    samples = (np.random.randn(num_samples) * 0.001).astype(np.float32)

    # Write WAV file
    data_size = num_samples * 2  # 16-bit
    with open(path, "wb") as f:
        # RIFF header
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + data_size))
        f.write(b"WAVE")
        # fmt chunk
        f.write(b"fmt ")
        f.write(struct.pack("<I", 16))  # chunk size
        f.write(struct.pack("<H", 1))  # PCM
        f.write(struct.pack("<H", 1))  # mono
        f.write(struct.pack("<I", sample_rate))
        f.write(struct.pack("<I", sample_rate * 2))  # byte rate
        f.write(struct.pack("<H", 2))  # block align
        f.write(struct.pack("<H", 16))  # bits per sample
        # data chunk
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        pcm = (samples * 32767).astype(np.int16)
        f.write(pcm.tobytes())

    print(f"Generated test WAV: {path} ({duration_sec}s, {sample_rate}Hz, mono)")
    return path


def main():
    parser = argparse.ArgumentParser(description="Test Parakeet transcriber")
    parser.add_argument("audio_file", nargs="?", help="Path to WAV audio file")
    parser.add_argument("--device", choices=["cpu", "npu", "gpu"], default="cpu",
                        help="Execution device (default: cpu)")
    parser.add_argument("--models-dir", default="./models",
                        help="Path to models directory (default: ./models)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--generate-test-wav", action="store_true",
                        help="Generate a test WAV file and exit")
    parser.add_argument("--runs", type=int, default=1,
                        help="Number of transcription runs for benchmarking")
    parser.add_argument("--decoder-device", choices=["auto", "cpu", "gpu"], default="auto",
                        help="Decoder device: auto=CPU, gpu=DirectML iGPU (default: auto)")
    args = parser.parse_args()

    if args.generate_test_wav:
        generate_test_wav()
        return

    if not args.audio_file:
        parser.print_help()
        print("\nError: audio_file is required (or use --generate-test-wav)")
        sys.exit(1)

    # Setup logging
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Read audio file
    with open(args.audio_file, "rb") as f:
        audio_data = f.read()
    print(f"Audio file: {args.audio_file} ({len(audio_data)} bytes)")

    # Initialize transcriber
    print(f"Initializing transcriber (device={args.device})...")
    t0 = time.perf_counter()

    from inference import Transcriber
    transcriber = Transcriber(
        models_dir=args.models_dir,
        device=args.device,
        decoder_device=args.decoder_device,
        debug=args.debug,
    )
    init_time = time.perf_counter() - t0
    print(f"Initialization: {init_time:.2f}s")

    # Print provider info
    info = transcriber.get_info()
    print(f"ONNX Runtime: {info['onnxruntime_version']}")
    print(f"Available providers: {info['available_providers']}")
    print(f"Encoder providers: {info['encoder_providers']}")
    print(f"Decoder providers: {info['decoder_providers']}")
    print()

    # Calculate audio duration from WAV header
    import struct
    audio_duration = len(audio_data) / (16000 * 2)  # rough estimate
    # Try to get actual duration from WAV
    try:
        from inference.audio import parse_wav
        samples = parse_wav(audio_data)
        audio_duration = len(samples) / 16000.0
    except Exception:
        pass

    # Run transcription
    times = []
    all_timings = []
    for i in range(args.runs):
        t0 = time.perf_counter()
        text = transcriber.transcribe(audio_data, audio_format=".wav")
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        timings = transcriber.get_last_timings()
        if timings:
            all_timings.append(timings.copy())

        rtf = elapsed / audio_duration if audio_duration > 0 else 0
        speed = audio_duration / elapsed if elapsed > 0 else 0

        if args.runs == 1:
            print(f"\nTranscription:")
            print(f"  \"{text}\"")
        else:
            print(f"  Run {i+1}/{args.runs}: {elapsed:.3f}s ({speed:.1f}x real-time)")

    # Print results summary
    print()
    print("=" * 60)
    print(f"  RESULTS ({args.device.upper()})")
    print("=" * 60)
    print(f"  Audio duration : {audio_duration:.1f}s ({audio_duration/60:.1f} min)")
    print(f"  Device         : {args.device.upper()}")
    print(f"  Encoder        : {info['encoder_providers'][0] if info['encoder_providers'] else 'N/A'}")
    print(f"  Decoder        : {info['decoder_providers'][0] if info['decoder_providers'] else 'N/A'}")

    if args.runs == 1:
        elapsed = times[0]
        rtf = elapsed / audio_duration if audio_duration > 0 else 0
        speed = audio_duration / elapsed if elapsed > 0 else 0
        print(f"  Processing time: {elapsed:.3f}s")
        print(f"  RTF            : {rtf:.4f}")
        print(f"  Speed          : {speed:.1f}x real-time")
    else:
        avg = sum(times) / len(times)
        best = min(times)
        worst = max(times)
        avg_rtf = avg / audio_duration if audio_duration > 0 else 0
        best_rtf = best / audio_duration if audio_duration > 0 else 0
        avg_speed = audio_duration / avg if avg > 0 else 0
        best_speed = audio_duration / best if best > 0 else 0
        print(f"  Runs           : {args.runs}")
        print(f"  Average time   : {avg:.3f}s  (RTF={avg_rtf:.4f}, {avg_speed:.1f}x real-time)")
        print(f"  Best time      : {best:.3f}s  (RTF={best_rtf:.4f}, {best_speed:.1f}x real-time)")
        print(f"  Worst time     : {worst:.3f}s")

    # Per-stage timing breakdown
    if all_timings:
        print()
        print("  --- Stage Breakdown (last run) ---")
        t = all_timings[-1]
        total_stages = t["mel"] + t["encoder"] + t["decoder"]
        overhead = times[-1] - total_stages
        print(f"  Mel features   : {t['mel']*1000:7.1f}ms  ({t['mel']/times[-1]*100:4.1f}%)")
        print(f"  Encoder (NPU)  : {t['encoder']*1000:7.1f}ms  ({t['encoder']/times[-1]*100:4.1f}%)")
        print(f"  Decoder (CPU)  : {t['decoder']*1000:7.1f}ms  ({t['decoder']/times[-1]*100:4.1f}%)")
        print(f"  Other overhead : {overhead*1000:7.1f}ms  ({overhead/times[-1]*100:4.1f}%)")
        print(f"  Decoder steps  : {t['decoder_steps']}  ({t['tokens']} tokens emitted)")
        if t['decoder_steps'] > 0:
            print(f"  Avg per step   : {t['decoder']/t['decoder_steps']*1000:.2f}ms/step")

        if len(all_timings) > 1:
            print()
            print("  --- Stage Breakdown (average over runs) ---")
            n = len(all_timings)
            avg_mel = sum(r["mel"] for r in all_timings) / n
            avg_enc = sum(r["encoder"] for r in all_timings) / n
            avg_dec = sum(r["decoder"] for r in all_timings) / n
            avg_total = avg_mel + avg_enc + avg_dec
            print(f"  Mel features   : {avg_mel*1000:7.1f}ms  ({avg_mel/avg*100:4.1f}%)")
            print(f"  Encoder        : {avg_enc*1000:7.1f}ms  ({avg_enc/avg*100:4.1f}%)")
            print(f"  Decoder        : {avg_dec*1000:7.1f}ms  ({avg_dec/avg*100:4.1f}%)")

    print("=" * 60)

    # Print full transcript
    print()
    print("  TRANSCRIPT")
    print("-" * 60)
    # Word-wrap the text to fit in the terminal
    import textwrap
    for line in textwrap.wrap(text, width=56):
        print(f"  {line}")
    print("-" * 60)
    print()

    transcriber.close()


if __name__ == "__main__":
    main()
