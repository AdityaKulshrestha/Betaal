"""Clipboard reformatter powered by an OpenVINO GenAI LLM.

Takes highlighted / clipboard text, rewrites it with an instruction prompt via
an ``openvino_genai.LLMPipeline``, and returns the cleaned-up text to paste
back. Kept intentionally small: the prompt and model are configurable, the
heavy ``openvino_genai`` import is lazy, and load status is exposed so the UI
can show whether the model is ready.
"""

from __future__ import annotations

import threading

from core.model_manager import ensure_llm_model

# Default instruction. ``{content}`` is replaced with the user's selected text.
DEFAULT_PROMPT = (
    "Reformat and clean up the following text. Fix grammar, spelling and "
    "punctuation, and improve clarity while preserving the original meaning "
    "and language. Return only the reformatted text with no preamble or "
    "explanation.\n\n{content}"
)

# Load / run status values reported to callers (and the UI).
STATUS_UNLOADED = "unloaded"
STATUS_LOADING = "loading"
STATUS_READY = "ready"
STATUS_REFORMATTING = "reformatting"
STATUS_ERROR = "error"


class Reformatter:
    """Lazily-loaded OpenVINO GenAI LLM that rewrites a block of text."""

    def __init__(
        self,
        model_display_name: str,
        device: str = "CPU",
        prompt: str | None = None,
        max_new_tokens: int = 512,
    ):
        self._model_display_name = model_display_name
        self._device = device or "CPU"
        self._prompt = prompt or DEFAULT_PROMPT
        self._max_new_tokens = int(max_new_tokens)
        self._pipe = None
        self._status = STATUS_UNLOADED
        self._error = ""
        self._lock = threading.Lock()

    @property
    def status(self) -> str:
        return self._status

    @property
    def error(self) -> str:
        return self._error

    def load(self) -> None:
        """Download (if needed) and build the LLM pipeline. Idempotent."""
        with self._lock:
            if self._pipe is not None:
                self._status = STATUS_READY
                return
            self._status = STATUS_LOADING
            try:
                import openvino_genai as ov_genai

                model_path = ensure_llm_model(self._model_display_name)
                pipeline_config = {"ATTENTION_BACKEND": "SDPA"}
                self._pipe = ov_genai.LLMPipeline(
                    model_path, self._device, **pipeline_config
                )
                self._status = STATUS_READY
                self._error = ""
            except Exception as exc:  # pragma: no cover - runtime/model dependent
                self._pipe = None
                self._status = STATUS_ERROR
                self._error = str(exc)
                raise

    def reformat(self, content: str) -> str:
        """Rewrite ``content`` using the configured prompt; returns clean text."""
        content = (content or "").strip()
        if not content:
            return ""
        if self._pipe is None:
            self.load()

        import openvino_genai as ov_genai

        prompt = self._prompt.replace("{content}", content)
        gen_config = ov_genai.GenerationConfig()
        gen_config.max_new_tokens = self._max_new_tokens
        # Newer GenAI builds can auto-wrap the prompt in the model's chat
        # template, which markedly improves instruct-model output.
        try:
            gen_config.apply_chat_template = True
        except Exception:  # pragma: no cover - older ov_genai without the field
            pass

        with self._lock:
            self._status = STATUS_REFORMATTING
            try:
                result = self._pipe.generate(prompt, gen_config)
            finally:
                self._status = STATUS_READY
        return str(result).strip()
