from __future__ import annotations

from collections import Counter
from io import BytesIO
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from nltk import word_tokenize
from nltk.collocations import TrigramAssocMeasures, TrigramCollocationFinder
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud

from .data_models import NLPAnalysisResult, TranscriptSegment
from .utils import ensure_nltk_resources


def _prepare_tokens(text: str) -> List[str]:
    ensure_nltk_resources()
    tokens = [token.lower() for token in word_tokenize(text)]
    stop_words = set(stopwords.words("english"))
    return [token for token in tokens if token.isalpha() and token not in stop_words]


def _sentiment_for_text(text: str) -> Dict[str, float]:
    ensure_nltk_resources()
    analyzer = SentimentIntensityAnalyzer()
    return analyzer.polarity_scores(text)


def analyze_transcript(
    segments: Iterable[TranscriptSegment],
    compact_text: str,
    top_n_terms: int = 25,
    top_n_trigrams: int = 10,
) -> NLPAnalysisResult:
    segments = list(segments)
    tokens = _prepare_tokens(compact_text)
    word_counts = Counter(tokens)
    top_frequencies = word_counts.most_common(top_n_terms)

    if tokens:
        trigram_measures = TrigramAssocMeasures()
        finder = TrigramCollocationFinder.from_words(tokens)
        finder.apply_freq_filter(2)
        scored = finder.score_ngrams(trigram_measures.pmi)
        top_trigrams = [trigram for trigram, _ in scored[:top_n_trigrams]]
    else:
        top_trigrams = []

    speaker_texts: Dict[str, str] = {}
    sentiment_timeline: List[Dict[str, float | str]] = []
    for segment in segments:
        speaker_texts.setdefault(segment.speaker, "")
        speaker_texts[segment.speaker] += f" {segment.text}"
        if segment.text.strip():
            scores = _sentiment_for_text(segment.text)
            sentiment_timeline.append(
                {
                    "speaker": segment.speaker,
                    "start_seconds": segment.start.total_seconds(),
                    "end_seconds": segment.end.total_seconds(),
                    "time_seconds": segment.start.total_seconds() + (segment.duration.total_seconds() / 2.0),
                    "compound": scores["compound"],
                    "positive": scores["pos"],
                    "negative": scores["neg"],
                    "neutral": scores["neu"],
                }
            )

    sentiment_timeline.sort(key=lambda entry: entry["time_seconds"])  # type: ignore[arg-type]

    speaker_sentiments = {
        speaker: _sentiment_for_text(text)
        for speaker, text in speaker_texts.items()
        if text.strip()
    }

    overall_sentiment = _sentiment_for_text(compact_text) if compact_text.strip() else {
        "neg": 0.0,
        "neu": 0.0,
        "pos": 0.0,
        "compound": 0.0,
    }

    fig, ax = plt.subplots(figsize=(10, 5))
    if top_frequencies:
        terms, counts = zip(*top_frequencies)
        indices = np.arange(len(terms))
        ax.bar(indices, counts, color="#1f77b4")
        ax.set_xticks(indices)
        ax.set_xticklabels(terms, rotation=45, ha="right")
        ax.set_ylabel("Frequency")
        ax.set_title("Top Word Frequencies")
    else:
        ax.text(0.5, 0.5, "No meaningful tokens", ha="center", va="center")
        ax.axis("off")
    fig.tight_layout()

    if tokens:
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(
            " ".join(tokens)
        )
    else:
        wordcloud = WordCloud(width=800, height=400, background_color="white")
        wordcloud.generate("No data")

    image_buffer = BytesIO()
    wordcloud_image = wordcloud.to_image()
    wordcloud_image.save(image_buffer, format="PNG")
    image_buffer.seek(0)

    return NLPAnalysisResult(
        compact_transcript=compact_text,
        overall_sentiment=overall_sentiment,
        speaker_sentiments=speaker_sentiments,
        top_trigrams=top_trigrams,
        word_frequencies=top_frequencies,
        frequency_figure=fig,
        wordcloud_image=wordcloud_image,
        wordcloud_buffer=image_buffer,
        sentiment_timeline=sentiment_timeline,
    )
