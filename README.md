# MeetPlot

MeetPlot ingests Zoom `.vtt` transcripts and produces NLP, graph, and PDF analytics through a Streamlit interface. The pipeline extracts speaker-level metrics, computes sentiment and trigram statistics, derives interaction graphs, performs spaCy named-entity recognition, and assembles a multi-page PDF report with all visual artefacts.

<figure>
   <img src="assets/Screenshot 2025-10-04 141835.png" alt="MeetPlot overview" style="max-width:100%;" />
   <figcaption><em>Figure 1.</em> MeetPlot overview — dashboard showing transcript upload, timeline, and key panels.</figcaption>
</figure>

<figure>
   <img src="assets/Screenshot 2025-10-04 141850.png" alt="Sentiment panels and trend" style="max-width:100%;" />
   <figcaption><em>Figure 2.</em> Sentiment panels and trend — per-speaker sentiment and timeline.</figcaption>
</figure>

## Capabilities

- Parse `.vtt` captions into structured segments with timing, speaker attribution, questions, and word counts.
- Generate NLP outputs using NLTK (overall sentiment, per-speaker sentiment, trigram collocations, frequency histogram, word cloud, sentiment timeline).
- Analyse speaker dynamics via NetworkX (interaction graph, question leaders, back-and-forth pairs, semantic topic map built from TF-IDF keywords).
- Recognise entities (PERSON, ORG, GPE, PRODUCT, EVENT, LOC) with spaCy.
- Export a ReportLab PDF that embeds charts, graphs, word cloud, key metrics, and entity summaries.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If automated downloads are blocked, manually fetch the required models:

```powershell
python -m nltk.downloader punkt stopwords vader_lexicon wordnet omw-1.4
python -m spacy download en_core_web_sm
```

## Usage

1. Start the dashboard:
   ```powershell
   streamlit run streamlit_app.py
   ```
2. Upload a Zoom `.vtt` transcript. Review the transcript timeline, NLP panels, graph analytics, named entities, and sentiment trend. A sample real meeting from the nyc.gov website is included in examples. 
3. Click **Generate PDF Report** to produce a downloadable briefing.
4. Run automated checks any time with:
   ```powershell
   python -m pytest
   ```

## Key Modules

- `app/main.py` – Streamlit UI and orchestration.
- `src/transcript_parser.py` – VTT parsing + speaker statistics.
- `src/nlp_analysis.py` – tokenisation, sentiment, trigrams, frequency plots, word cloud, sentiment timeline.
- `src/graph_analysis.py` – interaction graph, topic graph, question metrics.
- `src/ner_analysis.py` – spaCy entity extraction.
- `src/pdf_report.py` – ReportLab PDF composition.
- `examples/sample_zoom.vtt` – sample transcript for testing.
- `tests/` – pytest coverage for the parser.

## Operational Notes

- VTT captions must include speaker prefixes (e.g. `Speaker 1:`) to avoid `Unknown` assignments.
- Graph metrics rely on distinct vocabulary per speaker; short transcripts may yield sparse topic graphs.
- Report generation renders figures in-memory; large transcripts can take several seconds while matplotlib charts finish drawing.
