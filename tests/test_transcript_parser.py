from pathlib import Path

from src.transcript_parser import compact_transcript, compute_speaker_stats, parse_vtt

FIXTURE = Path(__file__).parent.parent / "examples" / "sample_zoom.vtt"


def test_parse_vtt_segments():
    text = FIXTURE.read_text()
    segments = parse_vtt(text)
    assert len(segments) == 6
    assert segments[0].speaker == "Speaker 1"
    assert segments[1].speaker == "Speaker 2"


def test_compute_speaker_stats():
    segments = parse_vtt(FIXTURE.read_text())
    stats = compute_speaker_stats(segments)
    assert stats["Speaker 1"].utterance_count == 2
    assert stats["Speaker 2"].question_count == 2


def test_compact_transcript_contains_keywords():
    segments = parse_vtt(FIXTURE.read_text())
    compact = compact_transcript(segments)
    assert "launch is scheduled" in compact
