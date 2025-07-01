"""voice.py – Unified Text-to-Speech (TTS) module for QuantAI Restaurant

This module encapsulates all TTS functionality for the QuantAI Restaurant
platform.  It provides a single public function `speak()` that will attempt
speech synthesis with Coqui XTTS-v2 first and automatically fall back to
ElevenLabs if Coqui is unavailable or fails at runtime.

Key features
------------
• Coqui XTTS-v2 integration with support for `speaker` or `speaker_wav` voice
  cloning arguments as documented in the official Coqui-TTS API.
• Automatic fallback to ElevenLabs (https://elevenlabs.io) when Coqui fails or
  is not installed.  Requires the environment variable `ELEVENLABS_API_KEY`.
• Device-aware initialisation (CUDA if available, else CPU).
• Lazy, singleton-style initialisation so that expensive model loads happen
  once per process.  Thread-safe for use from FastAPI handlers.
• Streaming support for real-time audio chunk generation.

Usage
-----
```python
import voice
wav_bytes = voice.speak("Kia ora, welcome to QuantAI!", language="en")

# For streaming:
async for chunk in voice.speak_streaming("Kia ora, welcome to QuantAI!", language="en"):
    # Process each audio chunk as it's generated
    process_chunk(chunk)
```
"""
from __future__ import annotations

import io
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional, AsyncGenerator, List
from threading import Lock
import asyncio
import concurrent.futures

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional ElevenLabs dependency – imported lazily inside the class so the
# package is only required when the Coqui fallback is triggered.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# TTS manager implementation
# ---------------------------------------------------------------------------

class _TTSManager:
    """Internal singleton that handles TTS with Coqui XTTS-v2 and fallback.

    The first successful backend that initialises becomes the active engine
    for the lifetime of the process.  Subsequent calls route directly to the
    chosen backend, making `speak()` effectively zero-overhead.
    """

    _instance: Optional["_TTSManager"] = None
    _instance_lock: Lock = Lock()
    _thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)

    # -------------------------- construction ------------------------------
    def __new__(cls, *args, **kwargs):  # noqa: D401 – singleton pattern
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:  # double-check inside the lock
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, speaker_wav: Optional[str] | Path = None):
        # Prevent re-initialisation in the singleton pattern
        if getattr(self, "_initialised", False):
            return
        self._initialised: bool = True
        self._speaker_wav = str(speaker_wav) if speaker_wav else None
        self._device = "cuda" if self._is_cuda_available() else "cpu"
        self._engine: Optional[str] = None  # "coqui" | "elevenlabs" | None

        # Attempt Coqui initialisation first
        if self._try_init_coqui():
            self._engine = "coqui"
            logger.info("Coqui XTTS-v2 backend initialised (device=%s)", self._device)
        elif self._try_init_elevenlabs():
            self._engine = "elevenlabs"
            logger.info("ElevenLabs backend initialised")
        else:
            logger.error("No TTS backend could be initialised – synthesis disabled")

    # --------------------------- helpers ----------------------------------
    @staticmethod
    def _is_cuda_available() -> bool:  # isolated for mocking in tests
        try:
            import torch

            return torch.cuda.is_available()
        except Exception:
            return False

    # ---------------------- Coqui initialisation ---------------------------
    def _try_init_coqui(self) -> bool:
        """Return True if Coqui XTTS-v2 was successfully initialised."""
        try:
            # Local import so that failures are caught cleanly
            import torch.serialization  # type: ignore
            from TTS.api import TTS  # type: ignore
            from TTS.tts.configs.xtts_config import XttsConfig  # type: ignore
            from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs  # type: ignore
            from TTS.config.shared_configs import BaseDatasetConfig  # type: ignore

            # PyTorch 2.6+ safety – allow Coqui custom classes during load
            torch.serialization.add_safe_globals([
                XttsConfig,
                XttsAudioConfig,
                BaseDatasetConfig,
                XttsArgs,
            ])

            self._coqui = TTS(
                model_name="tts_models/multilingual/multi-dataset/xtts_v2",
                progress_bar=False,
            ).to(self._device)
            return True
        except Exception as exc:  # pragma: no cover – handled as fallback
            logger.warning("Coqui XTTS-v2 backend unavailable: %s", exc)
            self._coqui = None  # type: ignore[attr-defined]
            return False

    # ---------------------- ElevenLabs initialisation ----------------------
    def _try_init_elevenlabs(self) -> bool:
        """Return True if ElevenLabs TTS backend could be initialised."""
        api_key = os.getenv("ELEVENLABS_API_KEY")
        if not api_key:
            logger.warning("ELEVENLABS_API_KEY not set – skipping ElevenLabs backend")
            return False
        try:
            from elevenlabs import set_api_key  # type: ignore

            set_api_key(api_key)
            # We intentionally defer the heavy imports (e.g. `generate`) to the
            # call site so that module load remains light-weight.
            return True
        except Exception as exc:  # pragma: no cover – library not installed
            logger.warning("ElevenLabs backend unavailable: %s", exc)
            return False

    # --------------------------- public API --------------------------------
    def speak(
        self,
        text: str,
        *,
        language: str = "en",
        speaker: Optional[str] = None,
        speaker_wav: Optional[str | Path] = None,
    ) -> Optional[bytes]:
        """Synthesise `text` into WAV bytes.

        The method will use the active backend determined at initialisation.
        If synthesis fails with the current backend, it will *once* attempt to
        fall back to the alternative engine (Coqui → ElevenLabs or vice versa).
        """
        if not text:
            return b""

        # First attempt with active engine -------------------------------------------------
        result = self._synth_with_current(text, language, speaker, speaker_wav)
        if result is not None:
            return result

        # Automatic fallback (once) --------------------------------------------------------
        if self._engine == "coqui" and self._try_init_elevenlabs():
            self._engine = "elevenlabs"
            return self._synth_with_current(text, language, speaker, speaker_wav)
        if self._engine == "elevenlabs" and self._try_init_coqui():
            self._engine = "coqui"
            return self._synth_with_current(text, language, speaker, speaker_wav)

        logger.error("All TTS backends failed – returning None")
        return None

    async def speak_streaming(
        self,
        text: str,
        *,
        language: str = "en",
        speaker: Optional[str] = None,
        speaker_wav: Optional[str | Path] = None,
        chunk_size: int = 1024,  # Bytes per chunk
    ) -> AsyncGenerator[bytes, None]:
        """Synthesise `text` into streaming WAV bytes.

        This is an async generator that yields audio chunks as they are generated.
        The method will use the active backend determined at initialisation.
        If streaming synthesis fails, it will fall back to non-streaming synthesis.
        """
        if not text:
            return
            
        # For very short text, use non-streaming for better efficiency
        if len(text) < 20:
            result = self.speak(text, language=language, speaker=speaker, speaker_wav=speaker_wav)
            if result:
                yield result
            return

        # Try streaming synthesis with current engine
        try:
            if self._engine == "coqui":
                async for chunk in self._coqui_synth_streaming(text, language, speaker, speaker_wav or self._speaker_wav, chunk_size):
                    yield chunk
            elif self._engine == "elevenlabs":
                async for chunk in self._elevenlabs_synth_streaming(text, language, speaker, chunk_size):
                    yield chunk
            else:
                # No engine available, try to initialize one
                if self._try_init_coqui():
                    self._engine = "coqui"
                    async for chunk in self._coqui_synth_streaming(text, language, speaker, speaker_wav or self._speaker_wav, chunk_size):
                        yield chunk
                elif self._try_init_elevenlabs():
                    self._engine = "elevenlabs"
                    async for chunk in self._elevenlabs_synth_streaming(text, language, speaker, chunk_size):
                        yield chunk
                else:
                    logger.error("No TTS backend available for streaming")
                    return
        except Exception as e:
            logger.error(f"Streaming synthesis failed: {e}, falling back to non-streaming")
            # Fall back to non-streaming synthesis
            result = self.speak(text, language=language, speaker=speaker, speaker_wav=speaker_wav)
            if result:
                yield result

    # -------------------------- internal logic -----------------------------
    def _synth_with_current(
        self,
        text: str,
        language: str,
        speaker: Optional[str],
        speaker_wav: Optional[str | Path],
    ) -> Optional[bytes]:
        if self._engine == "coqui":
            return self._coqui_synth(text, language, speaker, speaker_wav or self._speaker_wav)
        if self._engine == "elevenlabs":
            return self._elevenlabs_synth(text, language, speaker)
        return None

    # --------------------------- Coqui synth ------------------------------
    def _coqui_synth(
        self,
        text: str,
        language: str,
        speaker: Optional[str],
        speaker_wav: Optional[str | Path],
    ) -> Optional[bytes]:
        if not hasattr(self, "_coqui") or self._coqui is None:  # pragma: no cover
            return None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                out_path = Path(tmp.name)
            self._coqui.tts_to_file(
                text=text,
                language=language.lower(),
                speaker=speaker,
                speaker_wav=str(speaker_wav) if speaker_wav else None,
                file_path=str(out_path),
            )
            audio_bytes = out_path.read_bytes()
            out_path.unlink(missing_ok=True)
            return audio_bytes
        except Exception as exc:  # pragma: no cover – fallback triggers
            logger.error("Coqui synthesis failed: %s", exc)
            return None

    async def _coqui_synth_streaming(
        self,
        text: str,
        language: str,
        speaker: Optional[str],
        speaker_wav: Optional[str | Path],
        chunk_size: int = 1024,
    ) -> AsyncGenerator[bytes, None]:
        """Stream audio chunks from Coqui TTS."""
        if not hasattr(self, "_coqui") or self._coqui is None:
            return
            
        try:
            # Split text into sentences for parallel processing
            sentences = self._split_into_sentences(text)
            audio_futures = []
            
            # Process sentences in parallel using the thread pool
            loop = asyncio.get_event_loop()
            for sentence in sentences:
                if not sentence.strip():
                    continue
                future = loop.run_in_executor(
                    self._thread_pool,
                    self._process_sentence,
                    sentence,
                    language,
                    speaker,
                    speaker_wav
                )
                audio_futures.append(future)
            
            # As each sentence is processed, yield its audio chunks
            for future in asyncio.as_completed(audio_futures):
                audio_bytes = await future
                if audio_bytes:
                    # Yield chunks of the audio
                    for i in range(0, len(audio_bytes), chunk_size):
                        yield audio_bytes[i:i+chunk_size]
                        # Small delay to simulate real-time streaming
                        await asyncio.sleep(0.01)
                        
        except Exception as exc:
            logger.error(f"Coqui streaming synthesis failed: {exc}")
            return

    def _process_sentence(
        self,
        sentence: str,
        language: str,
        speaker: Optional[str],
        speaker_wav: Optional[str | Path],
    ) -> Optional[bytes]:
        """Process a single sentence with Coqui TTS (for thread pool execution)."""
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                out_path = Path(tmp.name)
            self._coqui.tts_to_file(
                text=sentence,
                language=language.lower(),
                speaker=speaker,
                speaker_wav=str(speaker_wav) if speaker_wav else None,
                file_path=str(out_path),
            )
            audio_bytes = out_path.read_bytes()
            out_path.unlink(missing_ok=True)
            return audio_bytes
        except Exception as exc:
            logger.error(f"Sentence processing failed: {exc}")
            return None

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences for parallel processing."""
        import re
        # Simple sentence splitting - can be improved for better results
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return sentences

    # ------------------------- ElevenLabs synth ---------------------------
    def _elevenlabs_synth(
        self,
        text: str,
        language: str,
        speaker: Optional[str],
    ) -> Optional[bytes]:
        try:
            # ElevenLabs recommends language-specific voices; fallback to provided
            # `speaker` name or use a generic one.
            voice_name = speaker or _default_elevenlabs_voice(language)

            from elevenlabs import generate  # type: ignore – heavy import

            audio_bytes: bytes = generate(text=text, voice=voice_name, model="eleven_multilingual_v2")
            return audio_bytes
        except Exception as exc:  # pragma: no cover
            logger.error("ElevenLabs synthesis failed: %s", exc)
            return None

    async def _elevenlabs_synth_streaming(
        self,
        text: str,
        language: str,
        speaker: Optional[str],
        chunk_size: int = 1024,
    ) -> AsyncGenerator[bytes, None]:
        """Stream audio chunks from ElevenLabs TTS."""
        try:
            voice_name = speaker or _default_elevenlabs_voice(language)
            
            # Import here to avoid dependency if not used
            from elevenlabs import generate, stream  # type: ignore
            
            # Split text into sentences for better streaming experience
            sentences = self._split_into_sentences(text)
            
            for sentence in sentences:
                if not sentence.strip():
                    continue
                
                # Use ElevenLabs streaming API
                audio_stream = stream(
                    text=sentence,
                    voice=voice_name,
                    model="eleven_multilingual_v2"
                )
                
                # Process the stream in chunks
                buffer = b""
                for chunk in audio_stream:
                    buffer += chunk
                    while len(buffer) >= chunk_size:
                        yield buffer[:chunk_size]
                        buffer = buffer[chunk_size:]
                        # Small delay to simulate real-time streaming
                        await asyncio.sleep(0.01)
                
                # Yield any remaining data
                if buffer:
                    yield buffer
                    
        except Exception as exc:
            logger.error(f"ElevenLabs streaming synthesis failed: {exc}")
            return

    # ---------------------------- misc ------------------------------------
    @property
    def engine(self) -> Optional[str]:
        return self._engine


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _default_elevenlabs_voice(lang: str) -> str:
    """Return a reasonable ElevenLabs voice name for a language code."""
    mapping = {
        "en": "Rachel",
        "es": "Antonio",
        "fr": "Chloe",
        "de": "Daniel",
        "it": "Giuseppe",
        "pt": "Joana",
        "zh": "Liang",
        "ja": "Akira",
        "ko": "Jisoo",
        "ru": "Dmitry",
        "ar": "Fatima",
        "hi": "Vedant",
    }
    return mapping.get(lang.lower()[:2], "Rachel")


# ---------------------------------------------------------------------------
# Public module-level API
# ---------------------------------------------------------------------------

# Lazy singleton instance ----------------------------------------------------
_default_manager: _TTSManager | None = None
_manager_lock: Lock = Lock()

def _get_manager() -> _TTSManager:
    global _default_manager
    if _default_manager is None:
        with _manager_lock:
            if _default_manager is None:
                _default_manager = _TTSManager()
    return _default_manager


def speak(
    text: str,
    *,
    language: str = "en",
    speaker: Optional[str] = None,
    speaker_wav: Optional[str | Path] = None,
) -> Optional[bytes]:
    """Public helper to synthesise `text` to WAV bytes.

    It defers to the internal `_TTSManager` singleton so that the expensive
    backend initialisation only happens once per process.
    """
    manager = _get_manager()
    return manager.speak(text, language=language, speaker=speaker, speaker_wav=speaker_wav)


async def speak_streaming(
    text: str,
    *,
    language: str = "en",
    speaker: Optional[str] = None,
    speaker_wav: Optional[str | Path] = None,
    chunk_size: int = 1024,
) -> AsyncGenerator[bytes, None]:
    """Public helper to synthesise `text` to streaming WAV bytes.

    This is an async generator that yields audio chunks as they are generated.
    It defers to the internal `_TTSManager` singleton for efficient processing.
    """
    manager = _get_manager()
    async for chunk in manager.speak_streaming(
        text, language=language, speaker=speaker, speaker_wav=speaker_wav, chunk_size=chunk_size
    ):
        yield chunk


def engine() -> Optional[str]:
    """Return the currently active TTS engine ("coqui", "elevenlabs", or None)."""
    return _get_manager().engine 