"""Audio -> VAD -> ASR processing pipeline.

This module captures microphone audio, chunks speech with ONNX Silero VAD,
passes chunks through a low-overhead ASR queue, and returns merged text.
"""

from __future__ import annotations

import queue
import threading
import time
from itertools import count

import numpy as np

from core.model_manager import ensure_required_models


class SileroVADChunker:
    """Chunk speech segments from float32 mono audio using ONNX Silero VAD."""

    def __init__(self, model_path: str, sample_rate: int = 16000, threshold: float = 0.5):
        self._sample_rate = sample_rate
        self._model_path = model_path
        self._threshold = max(0.05, min(0.95, float(threshold)))
        self._model = None
        self._get_speech_timestamps = None
        self._load()

    def _load(self) -> None:
        try:
            from silero_vad import get_speech_timestamps, load_silero_vad

            self._get_speech_timestamps = get_speech_timestamps
            try:
                self._model = load_silero_vad(onnx=True, model_path=self._model_path)
            except TypeError:
                self._model = load_silero_vad(onnx=True)
        except Exception as exc:  # pragma: no cover - runtime environment dependent
            print(f"[Betaal][vad] Failed to load Silero VAD, fallback mode: {exc}")
            self._model = None
            self._get_speech_timestamps = None

    def chunk(self, audio: np.ndarray) -> list[np.ndarray]:
        if audio.size == 0:
            return []

        if self._model is None or self._get_speech_timestamps is None:
            return self._energy_fallback(audio)

        try:
            timestamps = self._get_speech_timestamps(
                audio,
                self._model,
                sampling_rate=self._sample_rate,
                threshold=self._threshold,
                return_seconds=False,
            )
            chunks = [audio[item["start"]:item["end"]] for item in timestamps]
            return [chunk for chunk in chunks if chunk.size > 0]
        except Exception as exc:  # pragma: no cover - model behavior dependent
            print(f"[Betaal][vad] Silero chunking failed, fallback mode: {exc}")
            return self._energy_fallback(audio)

    def _energy_fallback(self, audio: np.ndarray) -> list[np.ndarray]:
        frame = int(self._sample_rate * 0.03)
        hop = frame
        threshold = max(0.002, self._threshold * 0.03)
        min_frames = 8
        silence_frames = 6

        segments = []
        active_start = None
        silent = 0
        voiced = 0

        for idx in range(0, max(len(audio) - frame, 1), hop):
            window = audio[idx:idx + frame]
            rms = float(np.sqrt(np.mean(window * window) + 1e-9))
            if rms >= threshold:
                if active_start is None:
                    active_start = idx
                voiced += 1
                silent = 0
            elif active_start is not None:
                silent += 1
                if silent >= silence_frames:
                    end = idx
                    if voiced >= min_frames:
                        segments.append(audio[active_start:end])
                    active_start = None
                    silent = 0
                    voiced = 0

        if active_start is not None and voiced >= min_frames:
            segments.append(audio[active_start:])
        return [seg for seg in segments if seg.size > 0]


class OVASRBackend:
    """OpenVINO-backed ASR wrapper, preferring GPU execution."""

    def __init__(self, model_dir: str, device: str = "GPU"):
        self._model_dir = model_dir
        self._device = device
        self._pipeline = None
        self._load()

    def _load(self) -> None:
        try:
            import openvino_genai as ov_genai

            self._pipeline = ov_genai.WhisperPipeline(self._model_dir, self._device)
            print(f"[Betaal][asr] Loaded OpenVINO Whisper pipeline on {self._device}")
        except Exception as first_exc:  # pragma: no cover - runtime environment dependent
            print(f"[Betaal][asr] GPU init failed: {first_exc}")
            try:
                import openvino_genai as ov_genai

                self._pipeline = ov_genai.WhisperPipeline(self._model_dir, "CPU")
                print("[Betaal][asr] Falling back to CPU")
            except Exception as second_exc:
                print(f"[Betaal][asr] ASR backend unavailable: {second_exc}")
                self._pipeline = None

    def transcribe(self, audio_chunk: np.ndarray) -> str:
        if audio_chunk.size == 0:
            return ""
        if self._pipeline is None:
            return ""

        try:
            result = self._pipeline.generate(audio_chunk)
            if isinstance(result, str):
                return result.strip()
            if hasattr(result, "texts") and result.texts:
                return str(result.texts[0]).strip()
            if hasattr(result, "text"):
                return str(result.text).strip()
            return str(result).strip()
        except Exception as exc:  # pragma: no cover - model behavior dependent
            print(f"[Betaal][asr] Transcription failed: {exc}")
            return ""


class AudioCapture:
    """Capture fixed-window mono PCM audio with minimal overhead."""

    def __init__(self, sample_rate: int = 16000, block_seconds: float = 0.25):
        self._sample_rate = sample_rate
        self._block_size = int(sample_rate * block_seconds)

    def capture(self, seconds: float = 3.0, stop_event: threading.Event | None = None) -> np.ndarray:
        frame_queue: queue.SimpleQueue[np.ndarray] = queue.SimpleQueue()

        def _callback(indata, frames, _time_info, _status) -> None:
            frame_queue.put(indata[:, 0].copy())

        total_samples = int(self._sample_rate * seconds)
        collected = []
        captured = 0
        try:
            import sounddevice as sd

            with sd.InputStream(
                samplerate=self._sample_rate,
                channels=1,
                dtype="float32",
                blocksize=self._block_size,
                callback=_callback,
            ):
                start = time.time()
                while (
                    captured < total_samples
                    and (time.time() - start) < (seconds + 2.0)
                    and (stop_event is None or not stop_event.is_set())
                ):
                    try:
                        block = frame_queue.get(timeout=0.2)
                    except Exception:
                        continue
                    collected.append(block)
                    captured += len(block)
        except Exception as exc:
            print(f"[Betaal][audio] Capture failed: {exc}")
            return np.empty((0,), dtype=np.float32)

        if not collected:
            return np.empty((0,), dtype=np.float32)
        return np.concatenate(collected).astype(np.float32, copy=False)


class TextPipeline:
    """Queue-based low-overhead speech pipeline.

    Flow:
        mic stream -> silero vad chunks -> asr queue -> merged text
    """

    def __init__(self, model_display_name: str, vad_threshold: float = 0.5):
        self._model_display_name = model_display_name
        model_paths = ensure_required_models(model_display_name)

        self._capture = AudioCapture(sample_rate=16000)
        self._vad = SileroVADChunker(
            model_paths["vad_model_path"],
            sample_rate=16000,
            threshold=vad_threshold,
        )
        self._asr = OVASRBackend(model_paths["asr_model_dir"], device="GPU")

        self._asr_in: queue.SimpleQueue[tuple[int, np.ndarray | None]] = queue.SimpleQueue()
        self._asr_out: queue.SimpleQueue[tuple[int, str | None]] = queue.SimpleQueue()
        self._job_ids = count(1)

        self._worker = threading.Thread(target=self._asr_worker, daemon=True)
        self._worker.start()

    def _asr_worker(self) -> None:
        while True:
            job_id, chunk = self._asr_in.get()
            if chunk is None:
                self._asr_out.put((job_id, None))
                continue
            text = self._asr.transcribe(chunk)
            self._asr_out.put((job_id, text))

    def process(
        self,
        capture_seconds: float = 3.0,
        stop_event: threading.Event | None = None,
    ) -> tuple[str, float]:
        """Capture audio, chunk by VAD, run ASR, return merged text + elapsed sec."""
        start = time.time()
        audio = self._capture.capture(seconds=capture_seconds, stop_event=stop_event)
        chunks = self._vad.chunk(audio)
        if not chunks:
            return "", time.time() - start

        job_id = next(self._job_ids)
        for chunk in chunks:
            self._asr_in.put((job_id, chunk))
        self._asr_in.put((job_id, None))

        parts = []
        while True:
            out_job_id, payload = self._asr_out.get()
            if out_job_id != job_id:
                continue
            if payload is None:
                break
            if payload:
                parts.append(payload)

        return " ".join(parts).strip(), time.time() - start
