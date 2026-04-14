#!/usr/bin/env python3
"""
Real-time streaming microphone transcription using Parakeet TDT 0.6B.

Captures audio from the default microphone, detects speech using simple
energy-based VAD, and transcribes in real-time on CPU or NPU.

Usage:
    conda activate ryzen-ai-1.7.0
    python live_transcribe.py --device npu         # NPU mode (20x+ real-time)
    python live_transcribe.py --device cpu          # CPU mode
    python live_transcribe.py --test-mic            # Test microphone levels
    python live_transcribe.py --list-devices        # Show audio devices

Controls:
    Ctrl+C to stop.
"""

import argparse
import logging
import os
import struct
import sys
import threading
import time
from collections import deque

import numpy as np

# Suppress ONNX Runtime warnings during live transcription
os.environ.setdefault("ORT_LOG_LEVEL", "ERROR")


# ─── Audio Constants ────────────────────────────────────────────────────────

SAMPLE_RATE = 16000
DTYPE = np.float32
BLOCK_DURATION_MS = 100  # 100ms audio blocks from mic
BLOCK_SIZE = int(SAMPLE_RATE * BLOCK_DURATION_MS / 1000)  # 1600 samples

# ─── VAD (Voice Activity Detection) ────────────────────────────────────────

SILENCE_THRESHOLD_MULTIPLIER = 1.5  # Speech must be Nx louder than ambient
SILENCE_THRESHOLD_MIN = 0.005  # Absolute minimum threshold
MAX_SILENCE_AFTER_SPEECH_S = 1.2  # Silence duration to end a speech segment
MAX_SEGMENT_DURATION_S = 15.0  # Force transcription after this long (matches NPU chunk)
MIN_SEGMENT_DURATION_S = 0.8  # Don't transcribe segments shorter than this


def _make_level_bar(energy, threshold, width=40):
    """Create an ASCII level meter bar."""
    # Scale: 0.0 to 0.3 covers most mic ranges
    max_display = 0.3
    level = min(energy / max_display, 1.0)
    thresh_pos = min(threshold / max_display, 1.0)

    filled = int(level * width)
    thresh_mark = int(thresh_pos * width)

    bar = list("." * width)
    for i in range(filled):
        bar[i] = "|" if energy > threshold else ":"
    if 0 <= thresh_mark < width:
        bar[thresh_mark] = "T"

    return "".join(bar)


class LiveTranscriber:
    """Real-time microphone transcription with energy-based VAD."""

    def __init__(self, device="cpu", models_dir="./models"):
        import sounddevice as sd
        self.sd = sd

        # Query default input device to get its native channel count
        dev_info = sd.query_devices(sd.default.device[0], "input")
        self.input_channels = min(int(dev_info["max_input_channels"]), 4)
        self.device_name = dev_info["name"]
        print(f"Microphone: {self.device_name} ({self.input_channels}ch)")

        # Initialize the transcriber (loads models)
        print(f"Loading Parakeet TDT 0.6B ({device.upper()})...")
        t0 = time.perf_counter()
        from inference import Transcriber
        self.transcriber = Transcriber(models_dir=models_dir, device=device)
        load_time = time.perf_counter() - t0
        print(f"Model loaded in {load_time:.1f}s")

        info = self.transcriber.get_info()
        print(f"Encoder: {info['encoder_providers'][0] if info['encoder_providers'] else 'N/A'}")
        print(f"Decoder: {info['decoder_providers'][0] if info['decoder_providers'] else 'N/A'}")

        # State
        self.audio_buffer = []  # Accumulated speech samples
        self.is_speaking = False
        self.speech_start_time = 0.0
        self.last_speech_time = 0.0
        self.ambient_energy = 0.0  # Calibrated ambient noise level
        self.energy_threshold = 0.01  # Will be auto-calibrated
        self.running = False
        self.transcript_lines = []  # Full conversation transcript
        self.show_levels = False  # Show live level meter

        # Thread-safe queue for audio -> transcription
        self._pending_segments = deque()
        self._transcribe_thread = None

    def _to_mono(self, indata):
        """Convert multi-channel input to mono by averaging all channels."""
        if indata.ndim == 1:
            return indata.copy()
        if indata.shape[1] == 1:
            return indata[:, 0].copy()
        # Average all channels for mic array
        return np.mean(indata, axis=1)

    def calibrate(self, duration_s=2.0):
        """Record ambient noise to auto-set the speech energy threshold."""
        print(f"\n  Calibrating ambient noise ({duration_s}s) -- stay quiet...")
        samples = self.sd.rec(
            int(SAMPLE_RATE * duration_s),
            samplerate=SAMPLE_RATE,
            channels=self.input_channels,
            dtype=DTYPE,
        )
        self.sd.wait()
        mono = self._to_mono(samples)
        energy = np.sqrt(np.mean(mono ** 2))
        self.ambient_energy = energy

        # Set threshold: just above ambient, with an absolute floor
        self.energy_threshold = max(
            energy * SILENCE_THRESHOLD_MULTIPLIER,
            energy + 0.01,  # At least 0.01 above ambient
            SILENCE_THRESHOLD_MIN,
        )
        print(f"  Ambient RMS : {energy:.5f}")
        print(f"  Threshold   : {self.energy_threshold:.5f}")
        print(f"  Headroom    : {self.energy_threshold / energy:.1f}x ambient")

    def _audio_callback(self, indata, frames, time_info, status):
        """Called by sounddevice for each audio block."""
        if status:
            pass  # Ignore overflow warnings during live capture

        samples = self._to_mono(indata)
        energy = np.sqrt(np.mean(samples ** 2))
        now = time.monotonic()

        # Show live level meter
        if self.show_levels or not self.is_speaking:
            bar = _make_level_bar(energy, self.energy_threshold)
            state = "SPEECH" if energy > self.energy_threshold else "     "
            speaking_dur = f" {now - self.speech_start_time:.1f}s" if self.is_speaking else ""
            self._print_status(f"[{bar}] {energy:.4f} {state}{speaking_dur}")

        if energy > self.energy_threshold:
            # Speech detected
            if not self.is_speaking:
                self.is_speaking = True
                self.speech_start_time = now
                self.audio_buffer = []

            self.last_speech_time = now
            self.audio_buffer.append(samples)

        elif self.is_speaking:
            # Below threshold but we were speaking -- keep buffering
            self.audio_buffer.append(samples)
            silence_duration = now - self.last_speech_time
            speech_duration = now - self.speech_start_time

            # End segment: enough silence after speech
            if silence_duration > MAX_SILENCE_AFTER_SPEECH_S:
                if speech_duration >= MIN_SEGMENT_DURATION_S:
                    self._finalize_segment()
                else:
                    # Too short, discard
                    self.is_speaking = False
                    self.audio_buffer = []

        # Force-finalize if segment is too long (approaching model limit)
        if self.is_speaking:
            speech_duration = now - self.speech_start_time
            if speech_duration > MAX_SEGMENT_DURATION_S:
                self._finalize_segment()

    def _finalize_segment(self):
        """Package the current speech segment for transcription."""
        if not self.audio_buffer:
            self.is_speaking = False
            return

        segment = np.concatenate(self.audio_buffer)
        duration = len(segment) / SAMPLE_RATE
        self.is_speaking = False
        self.audio_buffer = []

        self._print_status(f"Transcribing {duration:.1f}s...")
        self._pending_segments.append(segment)

    def _transcription_worker(self):
        """Background thread that processes audio segments."""
        while self.running or self._pending_segments:
            if self._pending_segments:
                segment = self._pending_segments.popleft()
                duration = len(segment) / SAMPLE_RATE

                # Convert to WAV bytes for the transcriber
                wav_bytes = self._to_wav_bytes(segment)

                t0 = time.perf_counter()
                text = self.transcriber.transcribe(wav_bytes, audio_format=".wav")
                elapsed = time.perf_counter() - t0

                if text.strip():
                    speed = duration / elapsed if elapsed > 0 else 0
                    self.transcript_lines.append(text.strip())
                    self._print_transcript(text.strip(), duration, elapsed, speed)
                else:
                    self._print_status("(no speech detected in segment)")
            else:
                time.sleep(0.05)

    @staticmethod
    def _to_wav_bytes(samples: np.ndarray) -> bytes:
        """Convert float32 samples to WAV file bytes."""
        pcm = (samples * 32767).clip(-32768, 32767).astype(np.int16)
        data_size = len(pcm) * 2
        header = struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF",
            36 + data_size,
            b"WAVE",
            b"fmt ",
            16,  # chunk size
            1,   # PCM
            1,   # mono
            SAMPLE_RATE,
            SAMPLE_RATE * 2,  # byte rate
            2,   # block align
            16,  # bits per sample
            b"data",
            data_size,
        )
        return header + pcm.tobytes()

    def _print_status(self, msg):
        """Print a status line that gets overwritten."""
        sys.stdout.write(f"\r\033[K  {msg}")
        sys.stdout.flush()

    def _print_transcript(self, text, duration, elapsed, speed):
        """Print a transcribed line with timing info."""
        n = len(self.transcript_lines)
        # Move to new line, clear, then print
        sys.stdout.write(f"\r\033[K")
        print(f"  [{n:3d}] ({duration:.1f}s -> {elapsed:.2f}s = {speed:.0f}x) {text}")
        sys.stdout.flush()

    def run(self):
        """Start live transcription. Press Ctrl+C to stop."""
        self.calibrate()

        print()
        print("=" * 60)
        print("  LIVE TRANSCRIPTION -- speak into your microphone!")
        print("  Level meter: [:::T....] T=threshold, :|=level")
        print("  Press Ctrl+C to stop.")
        print("=" * 60)
        print()

        self.running = True

        # Start transcription worker thread
        self._transcribe_thread = threading.Thread(
            target=self._transcription_worker, daemon=True
        )
        self._transcribe_thread.start()

        # Start audio stream -- request native channel count
        try:
            with self.sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=self.input_channels,
                dtype=DTYPE,
                blocksize=BLOCK_SIZE,
                callback=self._audio_callback,
            ):
                while True:
                    time.sleep(0.1)

        except KeyboardInterrupt:
            pass
        finally:
            self.running = False
            # Transcribe any remaining audio
            if self.is_speaking and self.audio_buffer:
                self._finalize_segment()

            # Wait for pending transcriptions
            if self._pending_segments:
                print("\n  Finishing pending transcriptions...")
            if self._transcribe_thread:
                self._transcribe_thread.join(timeout=30)

            print()
            print()
            print("=" * 60)
            print("  SESSION TRANSCRIPT")
            print("=" * 60)
            if self.transcript_lines:
                for i, line in enumerate(self.transcript_lines, 1):
                    print(f"  {i:3d}. {line}")
            else:
                print("  (no speech detected)")
            print("=" * 60)
            print()

            self.transcriber.close()


def test_microphone():
    """Show live microphone levels for 15 seconds to verify input."""
    import sounddevice as sd

    dev_info = sd.query_devices(sd.default.device[0], "input")
    channels = min(int(dev_info["max_input_channels"]), 4)
    print(f"\nMicrophone: {dev_info['name']} ({channels}ch)")
    print("Recording for 15s -- speak to test levels. Ctrl+C to stop.\n")

    peak = 0.0

    def callback(indata, frames, time_info, status):
        nonlocal peak
        if indata.ndim > 1 and indata.shape[1] > 1:
            mono = np.mean(indata, axis=1)
        else:
            mono = indata[:, 0] if indata.ndim > 1 else indata
        energy = np.sqrt(np.mean(mono ** 2))
        peak = max(peak, energy)
        bar = _make_level_bar(energy, 0.02, width=50)
        sys.stdout.write(f"\r  [{bar}] RMS={energy:.4f} peak={peak:.4f}")
        sys.stdout.flush()

    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=channels,
                            dtype=DTYPE, blocksize=BLOCK_SIZE, callback=callback):
            for _ in range(150):  # 15 seconds
                time.sleep(0.1)
    except KeyboardInterrupt:
        pass

    print(f"\n\n  Peak RMS: {peak:.4f}")
    if peak < 0.01:
        print("  WARNING: Very low levels. Check your microphone is working.")
    elif peak < 0.05:
        print("  Levels look low but usable. Try speaking closer to the mic.")
    else:
        print("  Levels look good!")
    print()


def list_audio_devices():
    """Print available audio input devices."""
    import sounddevice as sd
    print("\nAudio devices:")
    print(sd.query_devices())
    print(f"\nDefault input: {sd.default.device[0]}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Live microphone transcription")
    parser.add_argument("--device", choices=["cpu", "npu", "gpu"], default="cpu",
                        help="Execution device (default: cpu)")
    parser.add_argument("--models-dir", default="./models",
                        help="Path to models directory")
    parser.add_argument("--list-devices", action="store_true",
                        help="List audio devices and exit")
    parser.add_argument("--test-mic", action="store_true",
                        help="Test microphone levels and exit")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.list_devices:
        list_audio_devices()
        return

    if args.test_mic:
        test_microphone()
        return

    live = LiveTranscriber(device=args.device, models_dir=args.models_dir)
    live.run()


if __name__ == "__main__":
    main()
