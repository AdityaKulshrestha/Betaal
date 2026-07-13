"""Audio -> VAD -> ASR processing pipeline.

This module captures microphone audio, chunks speech with ONNX Silero VAD,
passes chunks through a low-overhead ASR queue, and returns merged text.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import queue
import threading
import time

import numpy as np

from core.model_manager import ensure_required_models
from core.model_registry import resolve_backend


def _ov_cache_dir() -> str | None:
    """Local (non-OneDrive) directory for OpenVINO compiled-model caching.

    Keeping the multi-GB compiled blob on LOCAL storage (LOCALAPPDATA) avoids
    OneDrive uploading / dehydrating it, which would otherwise force a slow
    recompile on the next boot.
    """
    try:
        root = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
        cache_dir = os.path.join(root, "Betaal", "ov_cache")
        os.makedirs(cache_dir, exist_ok=True)
        return cache_dir
    except Exception:  # pragma: no cover - env dependent
        return None


def available_devices() -> list[str]:
    """List selectable OpenVINO compute devices for the UI (AUTO + detected)."""
    devices = ["AUTO"]
    try:
        import openvino as ov

        for name in ov.Core().available_devices:
            base = name.split(".")[0]
            if base not in devices:
                devices.append(base)
    except Exception:  # pragma: no cover - env dependent
        pass
    for fallback in ("GPU", "CPU"):
        if fallback not in devices:
            devices.append(fallback)
    return devices


class SileroVADChunker:
    """Chunk speech segments from float32 mono audio using ONNX Silero VAD."""

    _WINDOW = 512  # Silero v5 requires exactly 512 samples per frame at 16 kHz
    _CONTEXT = 64  # Silero v5 prepends the last 64 samples of the previous frame

    def __init__(self, model_path: str, sample_rate: int = 16000, threshold: float = 0.5):
        self._sample_rate = sample_rate
        self._model_path = model_path
        self._threshold = max(0.05, min(0.95, float(threshold)))
        self._session = None
        self._load()

    def _load(self) -> None:
        # Run the ONNX model directly through onnxruntime -- no torch / silero
        # package needed (which would pull in PyTorch and slow startup).
        try:
            import onnxruntime as ort

            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 1
            opts.inter_op_num_threads = 1
            self._session = ort.InferenceSession(
                self._model_path,
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
        except Exception as exc:  # pragma: no cover - runtime environment dependent
            print(f"[Betaal][vad] Failed to load Silero VAD ONNX, fallback mode: {exc}")
            self._session = None

    def chunk(self, audio: np.ndarray) -> list[np.ndarray]:
        if audio.size == 0:
            return []

        if self._session is None:
            return self._energy_fallback(audio)

        try:
            segments = self._speech_timestamps(audio)
            chunks = [audio[start:end] for start, end in segments]
            return [chunk for chunk in chunks if chunk.size > 0]
        except Exception as exc:  # pragma: no cover - model behavior dependent
            print(f"[Betaal][vad] ONNX VAD failed, fallback mode: {exc}")
            return self._energy_fallback(audio)

    def _frame_probs(self, audio: np.ndarray) -> list[float]:
        """Run Silero VAD per 512-sample window, carrying hidden state + context.

        Silero VAD v5 expects each 512-sample window to be prefixed with the
        final 64 samples of the previous window (576 samples total). Omitting
        this context makes the model score nearly every frame as silence, so no
        speech segments are ever emitted.
        """
        session = self._session
        state = np.zeros((2, 1, 128), dtype=np.float32)
        context = np.zeros(self._CONTEXT, dtype=np.float32)
        sr = np.array(self._sample_rate, dtype=np.int64)
        window = self._WINDOW
        probs: list[float] = []
        for start in range(0, len(audio), window):
            frame = audio[start:start + window]
            if len(frame) < window:
                frame = np.pad(frame, (0, window - len(frame)))
            model_input = np.concatenate((context, frame)).reshape(1, -1).astype(np.float32)
            out, state = session.run(
                ["output", "stateN"],
                {
                    "input": model_input,
                    "state": state,
                    "sr": sr,
                },
            )
            context = frame[-self._CONTEXT:]
            probs.append(float(out[0, 0]))
        return probs

    def _speech_timestamps(self, audio: np.ndarray) -> list[tuple[int, int]]:
        """Reproduce Silero's segmenter (lenient defaults) from frame probs."""
        sr = self._sample_rate
        threshold = self._threshold
        neg_threshold = max(0.15, threshold - 0.15)
        min_speech = int(sr * 0.25)    # 250 ms
        min_silence = int(sr * 0.30)   # 300 ms (lenient)
        pad = int(sr * 0.20)           # 200 ms padding (lenient)
        window = self._WINDOW
        total = len(audio)

        probs = self._frame_probs(audio)
        segments: list[tuple[int, int]] = []
        triggered = False
        current_start = 0
        temp_end = 0
        for i, prob in enumerate(probs):
            sample = window * i
            if prob >= threshold and temp_end:
                temp_end = 0
            if prob >= threshold and not triggered:
                triggered = True
                current_start = sample
                continue
            if prob < neg_threshold and triggered:
                if not temp_end:
                    temp_end = sample
                if sample - temp_end < min_silence:
                    continue
                if temp_end - current_start > min_speech:
                    segments.append((current_start, temp_end))
                temp_end = 0
                triggered = False
        if triggered and (total - current_start) > min_speech:
            segments.append((current_start, total))

        return [(max(0, s - pad), min(total, e + pad)) for s, e in segments]

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

    def __init__(
        self,
        model_dir: str,
        device: str = "GPU",
        max_kv_tokens: int = 256,
        kv_evict_ratio: float = 0.25,
    ):
        self._model_dir = model_dir
        self._device = device
        self._ctx = None
        self._sample_rate = 16000
        self._max_new_tokens = 200
        # Self-attention KV-cache guard: never let the decode cache grow past
        # ``max_kv_tokens``; when it does, evict the oldest tokens so only the
        # most recent ``(1 - kv_evict_ratio)`` fraction is kept (sliding window).
        self._max_kv_tokens = max(1, int(max_kv_tokens))
        self._kv_evict_ratio = min(0.9, max(0.0, float(kv_evict_ratio)))
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
            from core.cohere_processor import CohereNPProcessor
        except Exception as exc:
            print(f"[Betaal][asr] Fallback dependencies unavailable: {exc}")
            return False

        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            # Torch-free processor: the upstream CohereAsrProcessor hard-depends
            # on PyTorch, but we only run the OpenVINO IR graphs, so we
            # reproduce its mel frontend + tokenizer in NumPy/librosa instead.
            processor = CohereNPProcessor(str(model_dir))
            core = ov.Core()
            # Cache compiled kernels to local disk so subsequent boots skip the
            # expensive recompilation (much faster startup, far less CPU).
            try:
                cache_dir = _ov_cache_dir()
                if cache_dir:
                    core.set_property({"CACHE_DIR": cache_dir})
            except Exception as cache_exc:  # pragma: no cover - env dependent
                print(f"[Betaal][asr] Model cache unavailable: {cache_exc}")
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

    def _evict_kv(self, self_kv):
        """Bound the self-attention KV cache; drop oldest tokens past the cap."""
        if not self_kv:
            return self_kv
        seq_len = self_kv[0][0].shape[2]
        if seq_len <= self._max_kv_tokens:
            return self_kv
        keep = max(1, int(self._max_kv_tokens * (1.0 - self._kv_evict_ratio)))
        return [(k[:, :, -keep:, :], v[:, :, -keep:, :]) for k, v in self_kv]

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
            self_kv = self._evict_kv(self_kv)

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


class WhisperOVBackend:
    """Whisper ASR via OpenVINO GenAI WhisperPipeline."""

    def __init__(self, model_dir: str, device: str = "GPU"):
        self._pipe = None
        self._load(str(model_dir), device)

    def _load(self, model_dir: str, device: str) -> None:
        try:
            import openvino_genai as ov_genai
        except Exception as exc:
            print(f"[Betaal][asr] openvino_genai unavailable: {exc}")
            return
        cache_dir = _ov_cache_dir()
        # Try GPU then CPU; for each, prefer the disk cache but fall back to no
        # cache in case the CACHE_DIR property is rejected by this build.
        attempts = []
        for dev in (device, "CPU"):
            if cache_dir:
                attempts.append((dev, {"CACHE_DIR": cache_dir}))
            attempts.append((dev, {}))
        for dev, cfg in attempts:
            try:
                self._pipe = ov_genai.WhisperPipeline(model_dir, dev, **cfg)
                print(f"[Betaal][asr] Loaded Whisper pipeline on {dev}")
                return
            except Exception as exc:
                print(f"[Betaal][asr] Whisper init failed on {dev}: {exc}")
        self._pipe = None

    def transcribe(self, audio_chunk: np.ndarray) -> str:
        if self._pipe is None or audio_chunk.size == 0:
            return ""
        try:
            result = self._pipe.generate(audio_chunk.astype(np.float32, copy=False))
            texts = getattr(result, "texts", None)
            text = texts[0] if texts else str(result)
            return str(text).strip()
        except Exception as exc:  # pragma: no cover - model behavior dependent
            print(f"[Betaal][asr] Whisper transcription failed: {exc}")
            return ""


class AudioCapture:
    """Continuous mono PCM capture with a single persistent input stream.

    A previous design opened a fresh ``InputStream`` for every 3 s window and
    closed it during VAD/ASR -- so anything spoken while the model was busy was
    lost. Here one stream stays open for the whole dictation session and its
    callback keeps filling a queue *during* inference, so no audio is dropped;
    ``capture`` simply drains the next window from that always-running queue.
    """

    def __init__(self, sample_rate: int = 16000, block_seconds: float = 0.25):
        self._sample_rate = sample_rate
        self._block_size = int(sample_rate * block_seconds)
        self._stream = None
        self._frame_queue: queue.SimpleQueue[np.ndarray] = queue.SimpleQueue()

    def start(self) -> None:
        """Open the persistent input stream (idempotent)."""
        if self._stream is not None:
            return
        import sounddevice as sd

        # Drop any stale frames buffered from a previous session.
        try:
            while True:
                self._frame_queue.get_nowait()
        except queue.Empty:
            pass

        def _callback(indata, _frames, _time_info, _status) -> None:
            self._frame_queue.put(indata[:, 0].copy())

        stream = sd.InputStream(
            samplerate=self._sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self._block_size,
            callback=_callback,
        )
        stream.start()
        self._stream = stream

    def stop(self) -> None:
        """Close the stream and release the microphone."""
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:  # pragma: no cover - device teardown guard
                pass

    def capture(self, seconds: float = 3.0, stop_event: threading.Event | None = None) -> np.ndarray:
        """Drain ``seconds`` of audio from the running stream (starts it lazily)."""
        if self._stream is None:
            try:
                self.start()
            except Exception as exc:
                print(f"[Betaal][audio] Capture failed: {exc}")
                return np.empty((0,), dtype=np.float32)

        total_samples = int(self._sample_rate * seconds)
        collected = []
        captured = 0
        start = time.time()
        while (
            captured < total_samples
            and (time.time() - start) < (seconds + 2.0)
            and (stop_event is None or not stop_event.is_set())
        ):
            try:
                block = self._frame_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            collected.append(block)
            captured += len(block)

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
        device: str = "GPU",
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
        if resolve_backend(model_display_name) == "whisper_genai":
            self._asr = WhisperOVBackend(model_paths["asr_model_dir"], device=device)
        else:
            self._asr = OVASRBackend(model_paths["asr_model_dir"], device=device)

    def start_capture(self) -> None:
        """Open the continuous microphone stream for a dictation session."""
        try:
            self._capture.start()
        except Exception as exc:  # pragma: no cover - device dependent
            print(f"[Betaal][audio] Could not start capture stream: {exc}")

    def stop_capture(self) -> None:
        """Close the microphone stream at the end of a dictation session."""
        self._capture.stop()

    def process(
        self,
        capture_seconds: float = 3.0,
        stop_event: threading.Event | None = None,
    ) -> tuple[str, float]:
        """Capture audio, chunk by VAD, run ASR inline, return merged text + elapsed sec."""
        start = time.time()
        audio = self._capture.capture(seconds=capture_seconds, stop_event=stop_event)
        chunks = self._vad.chunk(audio)
        if not chunks:
            return "", time.time() - start

        parts = []
        for chunk in chunks:
            if stop_event is not None and stop_event.is_set():
                break
            text = self._asr.transcribe(chunk)
            if text:
                parts.append(text)
                if self._log_transcript:
                    print(f"[Betaal][asr] chunk: {text}")

        merged = " ".join(parts).strip()
        if self._log_transcript and merged:
            print(f"[Betaal][asr] transcript: {merged}")
        return merged, time.time() - start
