from __future__ import annotations

from collections import OrderedDict
from datetime import timedelta

import altair as alt
import pandas as pd
import streamlit as st  # type: ignore[import]

from src.data_models import AnalysisBundle
from src.graph_analysis import analyze_graphs
from src.ner_analysis import extract_entities
from src.nlp_analysis import analyze_transcript
from src.pdf_report import generate_pdf_report
from src.transcript_parser import compact_transcript, compute_speaker_stats, parse_vtt, to_dataframe


def _format_timedelta(value: timedelta) -> str:
    total_seconds = value.total_seconds()
    minutes, seconds = divmod(total_seconds, 60)
    return f"{int(minutes)}m {int(seconds)}s"


def _render_speaker_metrics(bundle: AnalysisBundle) -> None:
    stats_list = list(bundle.speaker_stats.values())
    if not stats_list:
        return

    cols = st.columns(min(4, len(stats_list)))
    for idx, stats in enumerate(stats_list):
        col = cols[idx % len(cols)]
        col.metric(
            label=stats.speaker,
            value=_format_timedelta(stats.total_time),
            delta=f"{stats.total_words} words | {stats.question_count} questions",
        )

    longest = max(stats_list, key=lambda item: item.total_time, default=None)
    shortest = min(stats_list, key=lambda item: item.total_time, default=None)
    if longest and shortest:
        st.write(
            f"**Most talkative:** {longest.speaker} ({_format_timedelta(longest.total_time)}) | "
            f"**Least talkative:** {shortest.speaker} ({_format_timedelta(shortest.total_time)})"
        )


def _render_graph_insights(bundle: AnalysisBundle) -> None:
    st.subheader("Conversation Graph Insights")
    col_left, col_right = st.columns(2)

    with col_left:
        st.write("**Interaction Graph**")
        st.pyplot(bundle.graph.interaction_figure)
    with col_right:
        st.write("**Semantic Topic Graph**")
        st.pyplot(bundle.graph.topic_figure)

    st.write("**Question Leaders**")
    for speaker, count in bundle.graph.most_inquisitive_speakers[:5]:
        st.write(f"- {speaker}: {count} questions")

    if bundle.graph.back_and_forth_pairs:
        top_pair, weight = bundle.graph.back_and_forth_pairs[0]
        st.write(f"**Most interactive pair:** {top_pair[0]} ↔ {top_pair[1]} ({weight} exchanges)")
    else:
        st.info("Not enough exchanges to build interaction pairs.")


def _render_nlp_insights(bundle: AnalysisBundle) -> None:
    st.subheader("NLP Insights")
    st.write("**Overall Sentiment**")
    sentiment = bundle.nlp.overall_sentiment
    st.metric("Compound", f"{sentiment['compound']:.2f}", delta=f"Pos {sentiment['pos']:.2f} / Neg {sentiment['neg']:.2f}")

    if bundle.nlp.speaker_sentiments:
        st.write("**Sentiment by speaker**")
        sentiment_df = (
            pd.DataFrame(bundle.nlp.speaker_sentiments)
            .T[["pos", "neu", "neg", "compound"]]
            .reset_index()
            .rename(columns={"index": "Speaker", "pos": "Positive", "neu": "Neutral", "neg": "Negative", "compound": "Compound"})
        )
        melted = sentiment_df.melt(id_vars="Speaker", var_name="Metric", value_name="Score")
        chart = (
            alt.Chart(melted)
            .mark_bar()
            .encode(
                x=alt.X("Score", title="Sentiment Score", scale=alt.Scale(domain=[-1, 1])),
                y=alt.Y("Speaker", sort="-x"),
                color=alt.Color(
                    "Metric",
                    legend=alt.Legend(title="Metric", orient="bottom", direction="horizontal"),
                ),
                tooltip=["Speaker", "Metric", alt.Tooltip("Score", format=".2f")],
            )
            .properties(height=300)
        )
        st.altair_chart(chart, use_container_width=True)
        with st.expander("Sentiment table"):
            st.dataframe(sentiment_df, use_container_width=True)
        if bundle.nlp.sentiment_timeline:
            timeline_df = pd.DataFrame(bundle.nlp.sentiment_timeline)
            timeline_df["time_minutes"] = timeline_df["time_seconds"] / 60
            st.write("**Sentiment Trend Over Time**")
            trend_chart = (
                alt.Chart(timeline_df)
                .mark_line(point=True)
                .encode(
                    x=alt.X("time_minutes", title="Time (minutes)"),
                    y=alt.Y("compound", title="Compound Sentiment", scale=alt.Scale(domain=[-1, 1])),
                    color=alt.Color("speaker", title="Speaker"),
                    tooltip=[
                        alt.Tooltip("speaker", title="Speaker"),
                        alt.Tooltip("time_seconds", title="Time (s)", format=".1f"),
                        alt.Tooltip("compound", title="Compound", format=".2f"),
                        alt.Tooltip("positive", title="Positive", format=".2f"),
                        alt.Tooltip("negative", title="Negative", format=".2f"),
                    ],
                )
                .properties(height=250)
            )
            st.altair_chart(trend_chart, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(bundle.nlp.frequency_figure)
    with col2:
        st.image(bundle.nlp.wordcloud_buffer, caption="Word Cloud", use_container_width=True)

    if bundle.nlp.top_trigrams:
        st.write("**Top Trigram Collocations**")
        st.write(", ".join([" ".join(trigram) for trigram in bundle.nlp.top_trigrams]))


def _render_entities(bundle: AnalysisBundle) -> None:
    if not bundle.entities:
        return
    st.subheader("Named Entities")
    for label, values in bundle.entities.items():
        unique = sorted(set(values))
        if unique:
            st.write(f"**{label}**: {', '.join(unique)}")


def _prepare_bundle(vtt_text: str) -> AnalysisBundle:
    segments = parse_vtt(vtt_text)
    if not segments:
        st.warning("No transcript content detected.")
        st.stop()

    stats = compute_speaker_stats(segments)
    compact = compact_transcript(segments)
    nlp = analyze_transcript(segments, compact)
    graph = analyze_graphs(segments, stats)
    entities = extract_entities(segments)

    return AnalysisBundle(
        segments=segments,
        speaker_stats=OrderedDict(sorted(stats.items(), key=lambda item: item[0])),
        nlp=nlp,
        graph=graph,
        entities=entities,
    )


def run_app() -> None:
    st.set_page_config(page_title="MeetPlot", layout="wide")
    st.title("MeetPlot Transcript Analyzer")
    st.caption("Upload a Zoom .vtt transcript to explore conversational insights and export a PDF summary.")

    uploaded_file = st.file_uploader("Upload Zoom VTT", type=["vtt"])
    if not uploaded_file:
        st.info("Upload a transcript to begin analysis.")
        return

    vtt_bytes = uploaded_file.getvalue()
    text = vtt_bytes.decode("utf-8", errors="ignore")
    bundle = _prepare_bundle(text)

    st.success(f"Detected {len(bundle.segments)} transcript segments across {len(bundle.speaker_stats)} speakers.")

    _render_speaker_metrics(bundle)

    st.subheader("Transcript Timeline")
    df = to_dataframe(bundle.segments)
    st.dataframe(df)

    _render_nlp_insights(bundle)
    _render_graph_insights(bundle)
    _render_entities(bundle)

    if "pdf_bytes" not in st.session_state:
        st.session_state["pdf_bytes"] = None

    if st.button("Generate PDF Report"):
        with st.spinner("Generating PDF report..."):
            pdf_buffer = generate_pdf_report(bundle)
            bundle.pdf_buffer = pdf_buffer
            st.session_state["pdf_bytes"] = pdf_buffer.getvalue()

    if st.session_state.get("pdf_bytes"):
        st.download_button(
            label="Download MeetPlot Report",
            data=st.session_state["pdf_bytes"],
            file_name=bundle.pdf_name,
            mime="application/pdf",
        )


if __name__ == "__main__":
    run_app()
