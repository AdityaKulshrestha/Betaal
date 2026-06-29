"""Simulated ASR text pipeline for Betaal.

Buffer mode: bypasses real audio capture and VAD. When triggered it mocks a
short processing delay and returns a hardcoded transcription buffer.
"""

import time

# Hardcoded placeholder returned while running in buffer mode.
PLACEHOLDER_TEXT = (
    " [Betaal Test: This is a placeholder for the transcribed system text.] "
)

# Simulated processing latency in seconds.
PROCESS_DELAY = 1.5


class TextPipeline:
    """Mock transcription pipeline.

    Future integration:
        - sounddevice: capture microphone audio into numpy float32 arrays.
        - silero-vad: detect speech segments / endpoint silence.
        - faster-whisper: transcribe the buffered audio segments.
    Replace ``process`` internals to feed the audio array through VAD and the
    ASR model instead of the static placeholder.
    """

    def __init__(self, placeholder=PLACEHOLDER_TEXT, delay=PROCESS_DELAY):
        self._placeholder = placeholder
        self._delay = delay

    def process(self):
        """Simulate ASR execution and return (text, duration_seconds)."""
        start = time.time()
        time.sleep(self._delay)
        # TODO: audio_array = capture_audio(); segments = vad(audio_array);
        #       text = whisper.transcribe(segments)
        duration = time.time() - start
        return self._placeholder, duration
