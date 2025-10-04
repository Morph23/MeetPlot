from __future__ import annotations

import functools
import importlib
import logging
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import nltk
from matplotlib.figure import Figure

LOGGER = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def ensure_nltk_resources() -> None:
    resources = [
        ("punkt", "tokenizers/punkt"),
        ("stopwords", "corpora/stopwords"),
        ("vader_lexicon", "sentiment/vader_lexicon"),
        ("wordnet", "corpora/wordnet"),
        ("omw-1.4", "corpora/omw-1.4"),
    ]
    for resource_name, path in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            LOGGER.info("Downloading NLTK resource %s", resource_name)
            nltk.download(resource_name)


@functools.lru_cache(maxsize=1)
def ensure_spacy_model(model: str = "en_core_web_sm"):
    try:
        return importlib.import_module(model)
    except ImportError:  # pragma: no cover - depends on external network
        from spacy.cli import download  # type: ignore[import]

        LOGGER.info("Downloading spaCy model %s", model)
        download(model)
        return importlib.import_module(model)


@functools.lru_cache(maxsize=1)
def get_cache_dir() -> Path:
    cache_dir = Path(".meetplot_artifacts")
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def figure_to_png_bytes(fig: Figure, dpi: int = 200) -> BytesIO:
    buffer = BytesIO()
    fig.savefig(buffer, format="PNG", dpi=dpi, bbox_inches="tight")
    buffer.seek(0)
    return buffer
