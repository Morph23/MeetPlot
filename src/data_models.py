from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from io import BytesIO
from typing import Dict, List, Tuple, Union

import networkx as nx
from matplotlib.figure import Figure
from PIL import Image


@dataclass
class TranscriptSegment:
    """Single utterance extracted from the VTT transcript."""

    index: int
    start: timedelta
    end: timedelta
    speaker: str
    text: str

    @property
    def duration(self) -> timedelta:
        return self.end - self.start


@dataclass
class SpeakerStats:
    speaker: str
    total_time: timedelta
    total_words: int
    utterance_count: int
    question_count: int

    @property
    def average_utterance_duration(self) -> timedelta:
        if self.utterance_count == 0:
            return timedelta(0)
        return self.total_time / self.utterance_count

    @property
    def average_utterance_length(self) -> float:
        if self.utterance_count == 0:
            return 0.0
        return self.total_words / self.utterance_count


@dataclass
class NLPAnalysisResult:
    compact_transcript: str
    overall_sentiment: Dict[str, float]
    speaker_sentiments: Dict[str, Dict[str, float]]
    top_trigrams: List[Tuple[str, str, str]]
    word_frequencies: List[Tuple[str, int]]
    frequency_figure: Figure
    wordcloud_image: Image.Image
    wordcloud_buffer: BytesIO
    sentiment_timeline: List[Dict[str, Union[float, str]]]


@dataclass
class GraphAnalysisResult:
    interaction_graph: nx.DiGraph
    interaction_figure: Figure
    topic_graph: nx.Graph
    topic_figure: Figure
    question_counts: Dict[str, int]
    back_and_forth_pairs: List[Tuple[Tuple[str, str], int]]
    most_inquisitive_speakers: List[Tuple[str, int]]


@dataclass
class AnalysisBundle:
    segments: List[TranscriptSegment]
    speaker_stats: Dict[str, SpeakerStats]
    nlp: NLPAnalysisResult
    graph: GraphAnalysisResult
    entities: Dict[str, List[str]]
    pdf_buffer: BytesIO | None = None
    pdf_name: str = "meetplot_report.pdf"
    artifact_gallery: Dict[str, BytesIO] = field(default_factory=dict)
