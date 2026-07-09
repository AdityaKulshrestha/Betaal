"""Audio -> VAD -> ASR processing pipeline.

This module captures microphone audio, chunks speech with ONNX Silero VAD,
passes chunks through a low-overhead ASR queue, and returns merged text.
"""

from __future__ import annotations

import json
from pathlib import Path
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
    """Cohere OpenVINO IR ASR wrapper (KV-cache decode), preferring GPU."""

    def __init__(self, model_dir: str, device: str = "GPU"):
        self._model_dir = model_dir
        self._device = device
        self._ctx = None
        self._sample_rate = 16000
        self._max_new_tokens = 96
        self._load()

    def _load_ctx(self, device: str) -> bool:
        """Load Cohere OpenVINO IR graphs for direct KV-cache decoding."""
        model_dir = Path(self._model_dir)
        meta_path = model_dir / "ov_cohere_transcribe_kvcache.json"
        if not meta_path.is_file():
            nested_dir = model_dir / "models"
            nested_meta = nested_dir / "ov_cohere_transcribe_kvcache.json"
            if nested_meta.is_file():
                model_dir = nested_dir
                meta_path = nested_meta
            else:
                return False

        try:
            import openvino as ov
            from transformers import AutoProcessor
        except Exception as exc:
            print(f"[Betaal][asr] Fallback dependencies unavailable: {exc}")
            return False

        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            processor = AutoProcessor.from_pretrained(str(model_dir))
            core = ov.Core()
            encoder = core.compile_model(model_dir / meta["encoder_ir"], device)
            prefill = core.compile_model(model_dir / meta["decoder_ir"], device)
            decode = core.compile_model(model_dir / meta["decoder_with_past_ir"], device)
            eos = meta["eos_token_id"]
            eos_set = set(eos) if isinstance(eos, (list, tuple)) else {eos}
            num_layers = int(meta["num_layers"])

            self._ctx = {
                "processor": processor,
                "encoder": encoder,
                "prefill": prefill,
                "decode": decode,
                "eos_set": eos_set,
                "num_layers": num_layers,
            }
            print(f"[Betaal][asr] Loaded KV-cache ASR pipeline on {device}")
            return True
        except Exception as exc:
            print(f"[Betaal][asr] KV-cache init failed on {device}: {exc}")
            return False

    @staticmethod
    def _to_named_outputs(result: dict) -> dict:
        named = {}
        for key, value in result.items():
            if hasattr(key, "get_any_name"):
                named[key.get_any_name()] = value
            else:
                named[str(key)] = value
        return named

    @staticmethod
    def _np(x, dtype):
        return np.asarray(x.cpu() if hasattr(x, "cpu") else x).astype(dtype)

    def _transcribe(self, audio_chunk: np.ndarray) -> str:
        ctx = self._ctx
        if not ctx:
            return ""

        processor = ctx["processor"]
        encoder = ctx["encoder"]
        prefill = ctx["prefill"]
        decode = ctx["decode"]
        eos_set = ctx["eos_set"]
        num_layers = ctx["num_layers"]

        inputs = processor(audio_chunk, sampling_rate=self._sample_rate, language="en", return_tensors="np")
        feat = self._np(inputs["input_features"], np.float32)
        amask = self._np(inputs["attention_mask"], bool)
        prompt = self._np(inputs["decoder_input_ids"], np.int64)

        enc_res = encoder({"input_features": feat, "attention_mask": amask})
        ehs = enc_res["encoder_hidden_states"]
        emask = enc_res["encoder_attention_mask"]

        pf = prefill(
            {
                "decoder_input_ids": prompt,
                "encoder_hidden_states": ehs,
                "encoder_attention_mask": emask,
            }
        )
        pf = self._to_named_outputs(pf)

        self_kv = [
            (pf[f"present.{i}.self.key"], pf[f"present.{i}.self.value"])
            for i in range(num_layers)
        ]
        cross_kv = [
            (pf[f"present.{i}.cross.key"], pf[f"present.{i}.cross.value"])
            for i in range(num_layers)
        ]

        next_id = int(pf["logits"][0, -1].argmax())
        generated = [next_id]

        for _ in range(self._max_new_tokens):
            if next_id in eos_set:
                break

            seq_len = self_kv[0][0].shape[2] if self_kv else 0
            self_mask = np.ones((1, seq_len + 1), dtype=np.int64)
            feed = {
                "decoder_input_ids": np.array([[next_id]], dtype=np.int64),
                "encoder_hidden_states": ehs,
                "encoder_attention_mask": emask,
                "self_attention_mask": self_mask,
            }
            for i in range(num_layers):
                feed[f"past.{i}.self.key"] = self_kv[i][0]
                feed[f"past.{i}.self.value"] = self_kv[i][1]
                feed[f"past.{i}.cross.key"] = cross_kv[i][0]
                feed[f"past.{i}.cross.value"] = cross_kv[i][1]

            step = decode(feed)
            step = self._to_named_outputs(step)
            self_kv = [
                (step[f"present.{i}.self.key"], step[f"present.{i}.self.value"])
                for i in range(num_layers)
            ]

            next_id = int(step["logits"][0, -1].argmax())
            generated.append(next_id)

        text = processor.batch_decode([generated], skip_special_tokens=True)[0]
        return str(text).strip()

    def _load(self) -> None:
        if self._load_ctx(self._device):
            return
        if self._device != "CPU" and self._load_ctx("CPU"):
            return
        print("[Betaal][asr] ASR backend unavailable after KV-cache init attempts")

    def transcribe(self, audio_chunk: np.ndarray) -> str:
        if audio_chunk.size == 0:
            return ""
        if self._ctx is None:
            return ""
        try:
            return self._transcribe(audio_chunk)
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

    def __init__(
        self,
        model_display_name: str,
        vad_threshold: float = 0.5,
        log_transcript: bool = False,
    ):
        self._model_display_name = model_display_name
        self._log_transcript = log_transcript
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
                if self._log_transcript:
                    print(f"[Betaal][asr] chunk: {payload}")

        merged = " ".join(parts).strip()
        if self._log_transcript and merged:
            print(f"[Betaal][asr] transcript: {merged}")
        return merged, time.time() - start
