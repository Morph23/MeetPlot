from __future__ import annotations

import functools
from collections import defaultdict
from typing import Dict, Iterable, List

import spacy  # type: ignore[import]

from .data_models import TranscriptSegment
from .utils import ensure_spacy_model

DEFAULT_ENTITY_LABELS = ("PERSON", "ORG", "GPE", "PRODUCT", "EVENT", "LOC")


@functools.lru_cache(maxsize=1)
def _load_model(model_name: str = "en_core_web_sm"):
    ensure_spacy_model(model_name)
    return spacy.load(model_name)


def extract_entities(
    segments: Iterable[TranscriptSegment],
    labels: Iterable[str] = DEFAULT_ENTITY_LABELS,
) -> Dict[str, List[str]]:
    doc = _load_model()(" ".join(segment.text for segment in segments))
    allowed = set(labels)
    entities: Dict[str, List[str]] = defaultdict(list)
    for ent in doc.ents:
        if ent.label_ in allowed:
            entities[ent.label_].append(ent.text)
    return entities
