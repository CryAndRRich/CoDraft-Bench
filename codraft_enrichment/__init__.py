from .prompt import ENRICHMENT_PROMPTS
from .schemas import SCHEMA_MAP

# The original Gemini pipeline needs instructor, which the vLLM environment of
# scripts/llm.ipynb does not install, so its functions are imported on first use.
_LAZY = {"get_client", "process_batch", "run_enrichment"}


def __getattr__(name):
    if name in _LAZY:
        from . import main_enrichment
        return getattr(main_enrichment, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "get_client",
    "process_batch",
    "run_enrichment",
    "ENRICHMENT_PROMPTS",
    "SCHEMA_MAP"
]
