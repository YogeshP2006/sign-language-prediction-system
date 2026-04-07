"""Text-to-Speech engine for converting predicted gestures to audio."""

from typing import Optional

from src.utils import setup_logger

logger = setup_logger(__name__)


class TTSEngine:
    """Text-to-Speech engine using pyttsx3."""

    def __init__(self, rate: int = 150, volume: float = 0.9, voice_index: int = 0):
        self.rate = rate
        self.volume = volume
        self.voice_index = voice_index
        self._engine = None
        self._available = False
        self._init_engine()

    def _init_engine(self) -> None:
        """Initialize the pyttsx3 engine."""
        try:
            import pyttsx3
            self._engine = pyttsx3.init()
            self._engine.setProperty("rate", self.rate)
            self._engine.setProperty("volume", self.volume)
            voices = self._engine.getProperty("voices")
            if voices and self.voice_index < len(voices):
                self._engine.setProperty("voice", voices[self.voice_index].id)
            self._available = True
            logger.info("TTS engine initialized successfully")
        except Exception as e:
            logger.warning(f"TTS engine initialization failed: {e}. Audio output disabled.")
            self._available = False

    def speak(self, text: str) -> bool:
        """Convert text to speech. Returns True if successful."""
        if not self._available or not text:
            return False
        try:
            self._engine.say(text)
            self._engine.runAndWait()
            return True
        except Exception as e:
            logger.error(f"TTS error: {e}")
            return False

    def is_available(self) -> bool:
        """Check if TTS is available."""
        return self._available

    def close(self) -> None:
        """Clean up TTS resources."""
        if self._engine:
            try:
                self._engine.stop()
            except Exception:
                pass
