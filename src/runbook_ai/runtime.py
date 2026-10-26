"""Small helpers for checking the local Ollama installation."""

from dataclasses import dataclass

import ollama

from .answer import CHAT_MODEL
from .search import EMBED_MODEL


@dataclass(frozen=True)
class OllamaStatus:
    online: bool
    models: frozenset[str] = frozenset()
    error: str | None = None

    @property
    def chat_ready(self) -> bool:
        return _model_available(self.models, CHAT_MODEL)

    @property
    def embedding_ready(self) -> bool:
        return _model_available(self.models, EMBED_MODEL)

    @property
    def ready(self) -> bool:
        return self.online and self.chat_ready and self.embedding_ready


def _field(value, name, default=None):
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _model_available(installed: frozenset[str], required: str) -> bool:
    if required in installed:
        return True
    # Some Ollama versions include the full quantization suffix in list output.
    base, _, tag = required.partition(":")
    return any(name.startswith(f"{base}:{tag}") for name in installed)


def inspect_ollama() -> OllamaStatus:
    """Read local model names. A stopped or missing Ollama service is non-fatal."""
    try:
        # Keep the welcome screen responsive when Ollama is stopped or unreachable.
        response = ollama.Client(timeout=2).list()
        models = _field(response, "models", []) or []
        names = frozenset(
            name
            for item in models
            if (name := _field(item, "name") or _field(item, "model"))
        )
        return OllamaStatus(online=True, models=names)
    except Exception as exc:  # Ollama is an optional local service at app startup.
        return OllamaStatus(online=False, error=exc.__class__.__name__)
