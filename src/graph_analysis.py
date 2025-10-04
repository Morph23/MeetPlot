from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer

from .data_models import GraphAnalysisResult, SpeakerStats, TranscriptSegment


def _build_interaction_graph(segments: Iterable[TranscriptSegment]) -> nx.DiGraph:
    graph = nx.DiGraph()
    previous_speaker: str | None = None

    for segment in segments:
        speaker = segment.speaker
        if speaker not in graph:
            graph.add_node(speaker)
        if previous_speaker and previous_speaker != speaker:
            weight = graph[previous_speaker][speaker]["weight"] + 1 if graph.has_edge(previous_speaker, speaker) else 1
            graph.add_edge(previous_speaker, speaker, weight=weight)
        previous_speaker = speaker

    return graph


def _rank_back_and_forth_pairs(graph: nx.DiGraph) -> List[Tuple[Tuple[str, str], int]]:
    pair_weights: Counter[Tuple[str, str]] = Counter()
    for u, v, data in graph.edges(data=True):
        pair = tuple(sorted((u, v)))
        pair_weights[pair] += data.get("weight", 1)
    return pair_weights.most_common()


def _draw_interaction_graph(graph: nx.DiGraph) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 6))
    if not graph.nodes():
        ax.text(0.5, 0.5, "No interactions", ha="center", va="center")
        ax.axis("off")
        return fig

    pos = nx.spring_layout(graph, seed=42)
    weights = [graph[u][v]["weight"] for u, v in graph.edges]
    nx.draw_networkx_nodes(graph, pos, node_color="#f0ad4e", ax=ax)
    nx.draw_networkx_labels(graph, pos, font_size=10, ax=ax)
    if weights:
        nx.draw_networkx_edges(
            graph,
            pos,
            ax=ax,
            width=[0.5 + w for w in weights],
            arrows=True,
            arrowstyle="-|>",
            arrowsize=12,
            edge_color="#5bc0de",
        )
    ax.set_title("Speaker Interaction Graph")
    ax.axis("off")
    fig.tight_layout()
    return fig


def _build_topic_graph(speaker_texts: Dict[str, str], top_terms: int = 5) -> Tuple[nx.Graph, Dict[str, List[str]]]:
    topics: Dict[str, List[str]] = {}
    graph = nx.Graph()

    valid_items = {speaker: text for speaker, text in speaker_texts.items() if text.strip()}
    if not valid_items:
        return graph, topics

    vectorizer = TfidfVectorizer(max_features=100, stop_words="english", ngram_range=(1, 2))
    texts = list(valid_items.values())
    speakers = list(valid_items.keys())
    matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()

    feature_list = feature_names.tolist()

    for idx, speaker in enumerate(speakers):
        graph.add_node(speaker, bipartite="speaker")
        row = matrix[idx].toarray().ravel()
        if row.sum() == 0:
            continue
        sorted_indices = row.argsort()[::-1]
        keywords: List[str] = []
        for feature_idx in sorted_indices:
            weight = float(row[feature_idx])
            if weight <= 0:
                break
            keyword = feature_list[feature_idx]
            if keyword in keywords:
                continue
            keywords.append(keyword)
            graph.add_node(keyword, bipartite="topic")
            graph.add_edge(speaker, keyword, weight=weight)
            if len(keywords) >= top_terms:
                break
        if keywords:
            topics[speaker] = keywords

    return graph, topics


def _draw_topic_graph(graph: nx.Graph) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 6))
    if not graph.nodes():
        ax.text(0.5, 0.5, "No topics identified", ha="center", va="center")
        ax.axis("off")
        return fig

    speakers = [node for node, data in graph.nodes(data=True) if data.get("bipartite") == "speaker"]
    topics = [node for node in graph if node not in speakers]

    pos = nx.spring_layout(graph, seed=24)
    nx.draw_networkx_nodes(graph, pos, nodelist=speakers, node_color="#5cb85c", node_shape="s", ax=ax)
    nx.draw_networkx_nodes(graph, pos, nodelist=topics, node_color="#d9534f", node_shape="o", ax=ax)
    nx.draw_networkx_labels(graph, pos, font_size=9, ax=ax)
    nx.draw_networkx_edges(graph, pos, ax=ax, edge_color="#999999")

    ax.set_title("Speaker-Topic Semantic Graph")
    ax.axis("off")
    fig.tight_layout()
    return fig


def analyze_graphs(
    segments: Iterable[TranscriptSegment],
    speaker_stats: Dict[str, SpeakerStats],
) -> GraphAnalysisResult:
    graph = _build_interaction_graph(segments)
    back_and_forth_pairs = _rank_back_and_forth_pairs(graph)

    question_counts = {speaker: stats.question_count for speaker, stats in speaker_stats.items()}
    inquisitive = sorted(question_counts.items(), key=lambda item: item[1], reverse=True)

    speaker_texts = defaultdict(str)
    for segment in segments:
        speaker_texts[segment.speaker] += f" {segment.text}"

    topic_graph, _ = _build_topic_graph(dict(speaker_texts))

    return GraphAnalysisResult(
        interaction_graph=graph,
        interaction_figure=_draw_interaction_graph(graph),
        topic_graph=topic_graph,
        topic_figure=_draw_topic_graph(topic_graph),
        question_counts=question_counts,
        back_and_forth_pairs=back_and_forth_pairs,
        most_inquisitive_speakers=inquisitive,
    )
