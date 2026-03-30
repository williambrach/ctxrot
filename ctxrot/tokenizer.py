import logging
from typing import Protocol

logger = logging.getLogger(__name__)


class _TokenCounter(Protocol):
    def count_tokens(self, text: str) -> int: ...


_tokenizer: _TokenCounter | None = None
_fallback: bool = False

_MODEL_ID = "tokiers/gpt2"


def _get_tokenizer() -> _TokenCounter | None:
    global _tokenizer, _fallback  # noqa: PLW0603
    if _fallback:
        return None
    if _tokenizer is not None:
        return _tokenizer
    try:
        import tokie  # noqa: PLC0415

        _tokenizer = tokie.Tokenizer.from_pretrained(_MODEL_ID)
        return _tokenizer
    except Exception:
        logger.warning(
            "Failed to load tokie tokenizer (%s), falling back to len//4",
            _MODEL_ID,
        )
        _fallback = True
        return None


def count_tokens(text: str) -> int:
    """Count tokens in *text*. Falls back to len(text)//4 on error."""
    tok = _get_tokenizer()
    if tok is None:
        return len(text) // 4
    try:
        return tok.count_tokens(text)
    except Exception:
        logger.warning("tokie.count_tokens failed, falling back to len//4")
        return len(text) // 4
