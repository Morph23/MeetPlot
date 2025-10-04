from __future__ import annotations

from datetime import datetime
from io import BytesIO
from typing import Iterable, List

from reportlab.lib import colors  # type: ignore[import]
from reportlab.lib.pagesizes import letter  # type: ignore[import]
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # type: ignore[import]
from reportlab.lib.units import inch  # type: ignore[import]
from reportlab.platypus import (  # type: ignore[import]
    Image,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from .data_models import AnalysisBundle, SpeakerStats
from .utils import figure_to_png_bytes


def _format_seconds(value) -> str:
    return f"{float(value):.2f}s"


def _speaker_table(stats: Iterable[SpeakerStats]) -> Table:
    data: List[List[str]] = [
        [
            "Speaker",
            "Total Time",
            "Total Words",
            "Questions",
            "Avg Utterance (s)",
            "Avg Words",
        ]
    ]
    for item in stats:
        data.append(
            [
                item.speaker,
                _format_seconds(item.total_time.total_seconds()),
                str(item.total_words),
                str(item.question_count),
                _format_seconds(item.average_utterance_duration.total_seconds()),
                f"{item.average_utterance_length:.1f}",
            ]
        )

    table = Table(data, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4a90e2")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("ALIGN", (1, 1), (-1, -1), "CENTER"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 10),
                ("BACKGROUND", (0, 1), (-1, -1), colors.whitesmoke),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ]
        )
    )
    return table


def _build_image(figure) -> Image:
    buffer = figure_to_png_bytes(figure)
    return Image(buffer, width=6.5 * inch, height=3.5 * inch)


def generate_pdf_report(bundle: AnalysisBundle) -> BytesIO:
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.75 * inch, bottomMargin=0.75 * inch)
    styles = getSampleStyleSheet()
    title_style = styles["Title"]
    subtitle_style = styles["Heading2"]
    body_style = styles["BodyText"]
    bullet_style = ParagraphStyle("Bullets", parent=body_style, bulletIndent=12, leftIndent=18)

    story: List = []
    story.append(Paragraph("MeetPlot Transcript Analysis Report", title_style))
    story.append(Paragraph(f"Generated on {datetime.utcnow():%Y-%m-%d %H:%M UTC}", body_style))
    story.append(Spacer(1, 0.2 * inch))

    speakers = list(bundle.speaker_stats.keys())
    story.append(Paragraph("Conversation Overview", subtitle_style))
    story.append(
        Paragraph(
            f"Participants: {len(speakers)} | Segments: {len(bundle.segments)} | PDF Filename: {bundle.pdf_name}",
            body_style,
        )
    )
    story.append(Spacer(1, 0.1 * inch))

    story.append(Paragraph("Speaker Participation", subtitle_style))
    story.append(_speaker_table(bundle.speaker_stats.values()))
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("Sentiment Summary", subtitle_style))
    sentiment_lines = [
        Paragraph(
            f"Overall sentiment — Positive: {bundle.nlp.overall_sentiment['pos']:.2f}, Neutral: {bundle.nlp.overall_sentiment['neu']:.2f}, Negative: {bundle.nlp.overall_sentiment['neg']:.2f}, Compound: {bundle.nlp.overall_sentiment['compound']:.2f}",
            body_style,
        )
    ]
    story.extend(sentiment_lines)

    if bundle.nlp.speaker_sentiments:
        bullets = ListFlowable(
            [
                ListItem(
                    Paragraph(
                        f"{speaker}: pos {scores['pos']:.2f}, neu {scores['neu']:.2f}, neg {scores['neg']:.2f}, compound {scores['compound']:.2f}",
                        bullet_style,
                    )
                )
                for speaker, scores in bundle.nlp.speaker_sentiments.items()
            ],
            bulletType="bullet",
        )
        story.append(bullets)
    story.append(Spacer(1, 0.2 * inch))

    if bundle.nlp.word_frequencies:
        story.append(Paragraph("Top Word Frequencies", subtitle_style))
        story.append(_build_image(bundle.nlp.frequency_figure))
        story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("Word Cloud", subtitle_style))
    story.append(Image(bundle.nlp.wordcloud_buffer, width=6.5 * inch, height=3.0 * inch))
    story.append(Spacer(1, 0.2 * inch))

    if bundle.graph.back_and_forth_pairs:
        top_pair, top_weight = bundle.graph.back_and_forth_pairs[0]
        story.append(Paragraph("Conversation Dynamics", subtitle_style))
        story.append(
            Paragraph(
                f"Most back-and-forth: {top_pair[0]} → {top_pair[1]} ({top_weight} exchanges)",
                body_style,
            )
        )
    else:
        story.append(Paragraph("Conversation Dynamics", subtitle_style))
        story.append(Paragraph("Not enough interaction data to compute exchanges.", body_style))
    story.append(Spacer(1, 0.2 * inch))

    if bundle.graph.question_counts:
        story.append(Paragraph("Questions Asked", subtitle_style))
        question_items = sorted(bundle.graph.question_counts.items(), key=lambda item: item[1], reverse=True)
        story.append(
            ListFlowable(
                [
                    ListItem(Paragraph(f"{speaker}: {count} questions", bullet_style))
                    for speaker, count in question_items
                ],
                bulletType="bullet",
            )
        )
        story.append(Spacer(1, 0.2 * inch))

    if bundle.graph.interaction_figure:
        story.append(Paragraph("Speaker Interaction Graph", subtitle_style))
        story.append(_build_image(bundle.graph.interaction_figure))
        story.append(Spacer(1, 0.2 * inch))

    if bundle.graph.topic_figure:
        story.append(Paragraph("Semantic Topic Graph", subtitle_style))
        story.append(_build_image(bundle.graph.topic_figure))
        story.append(Spacer(1, 0.2 * inch))

    if bundle.entities:
        story.append(Paragraph("Named Entities", subtitle_style))
        entity_items = []
        for label, values in bundle.entities.items():
            unique_values = sorted(set(values))
            if unique_values:
                entity_items.append(
                    ListItem(
                        Paragraph(f"{label}: {', '.join(unique_values)}", bullet_style),
                        bulletColor=colors.HexColor("#4a90e2"),
                    )
                )
        if entity_items:
            story.append(ListFlowable(entity_items, bulletType="bullet"))
        story.append(Spacer(1, 0.2 * inch))

    story.append(PageBreak())
    story.append(Paragraph("Top Trigrams", subtitle_style))
    if bundle.nlp.top_trigrams:
        trigram_items = [" ".join(trigram) for trigram in bundle.nlp.top_trigrams]
        story.append(
            ListFlowable(
                [ListItem(Paragraph(item, bullet_style)) for item in trigram_items],
                bulletType="bullet",
            )
        )
    else:
        story.append(Paragraph("No trigram collocations identified.", body_style))

    doc.build(story)
    buffer.seek(0)
    return buffer
