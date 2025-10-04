from __future__ import annotations

import re
from datetime import datetime, timedelta
from io import StringIO
from typing import Dict, Iterable, List, Tuple

import pandas as pd
import webvtt

from .data_models import SpeakerStats, TranscriptSegment

TIMESTAMP_FORMATS = ["%H:%M:%S.%f", "%M:%S.%f", "%H:%M:%S", "%M:%S"]
SPEAKER_PATTERN = re.compile(r"^(?P<speaker>[^:]+):\s*(?P<text>.+)$")
WORD_PATTERN = re.compile(r"[\w']+")


def _parse_timestamp(value: str) -> timedelta:
    for fmt in TIMESTAMP_FORMATS:
        try:
            dt = datetime.strptime(value, fmt)
            base = datetime(1900, 1, 1)
            return dt - base
        except ValueError:
            continue
    raise ValueError(f"Unrecognized timestamp format: {value}")


def _extract_speaker(text: str) -> Tuple[str, str]:
    match = SPEAKER_PATTERN.match(text)
    if match:
        return match.group("speaker").strip(), match.group("text").strip()
    return "Unknown", text.strip()


def parse_vtt(file_text: str | bytes) -> List[TranscriptSegment]:
    """Parse a VTT transcript into structured segments."""

    if isinstance(file_text, bytes):
        buffer = StringIO(file_text.decode("utf-8", errors="ignore"))
    else:
        buffer = StringIO(file_text)

    buffer.seek(0)
    vtt = webvtt.from_buffer(buffer)
    segments: List[TranscriptSegment] = []
    for idx, caption in enumerate(vtt):
        text = caption.text.strip()
        if not text:
            continue
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        normalized = " ".join(lines)
        speaker, utterance = _extract_speaker(normalized)
        segment = TranscriptSegment(
            index=idx,
            start=_parse_timestamp(caption.start),
            end=_parse_timestamp(caption.end),
            speaker=speaker,
            text=utterance,
        )
        segments.append(segment)
    return segments


def _count_questions(text: str) -> int:
    sentences = re.split(r"(?<=[?])\s+", text)
    return sum(1 for sentence in sentences if sentence.strip().endswith("?"))


def compute_speaker_stats(segments: Iterable[TranscriptSegment]) -> Dict[str, SpeakerStats]:
    stats: Dict[str, SpeakerStats] = {}
    for segment in segments:
        words = WORD_PATTERN.findall(segment.text)
        entry = stats.setdefault(
            segment.speaker,
            SpeakerStats(
                speaker=segment.speaker,
                total_time=timedelta(0),
                total_words=0,
                utterance_count=0,
                question_count=0,
            ),
        )
        entry.total_time += segment.duration
        entry.total_words += len(words)
        entry.utterance_count += 1
        entry.question_count += _count_questions(segment.text)
    return stats


def compact_transcript(segments: Iterable[TranscriptSegment]) -> str:
    return " ".join(segment.text.strip() for segment in segments if segment.text)


def to_dataframe(segments: Iterable[TranscriptSegment]) -> pd.DataFrame:
    data = [
        {
            "index": segment.index,
            "start": segment.start.total_seconds(),
            "end": segment.end.total_seconds(),
            "speaker": segment.speaker,
            "text": segment.text,
            "duration": segment.duration.total_seconds(),
            "word_count": len(WORD_PATTERN.findall(segment.text)),
            "question_count": _count_questions(segment.text),
        }
        for segment in segments
    ]
    return pd.DataFrame(data)
