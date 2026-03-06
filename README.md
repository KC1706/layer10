# Layer10: Organizational Memory Graph Pipeline

This is a complete end-to-end Python system that extracts organizational knowledge from an unstructured public corpus (like GitHub issues/PRs), and builds a structured, grounded long-term memory graph optimized for strict evidence grounding and reversibility.

## Project Structure

- `requirements.txt`: Project dependencies.
- `ingest.py` (Phase 1): Fetches and deduplicates GitHub issues into an unstructured JSONL corpus.
- `extractor.py` (Phase 2): Uses the LiteLLM/Instructor stack to map unstructured text into structured Pydantic objects (Entity, Claim, Evidence). Ensures exact-substring grounding character-for-character.
- `dedup.py` (Phase 3): Reconciles entities using `thefuzz` name-matching and semantically deduplicates relationships utilizing `sentence-transformers` embeddings. Records merge events in an audit log.
- `graph_store.py` (Phase 4): Ingests the canonicalized components into SQLite and models the network inside `networkx`.
- `retriever.py` (Phase 5): FastAPI backend endpoint `get_context_pack` demonstrating semantic retrieval.
- `app.py` (Phase 6): Streamlit interactive Visualization layer (with PyVis) for manual graph deep-diving.

## Quickstart Execution

### 1. Environment Setup

Ensure you have Python 3.10+ installed.

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure LLM
The extraction phase requires an LLM. By default, it uses Gemini Flash via `litellm`.
You MUST export your key:
```bash
export GEMINI_API_KEY="your-api-key-here"
```

### 3. Run Pipeline End-to-End

**Phase 1: Ingestion**
Fetches 100 recent issues/PRs from a public repo (default: `tiangolo/fastapi`) and saves to `data/corpus.jsonl`
```bash
python ingest.py --repo "tiangolo/fastapi" --limit 100
```

**Phase 2: LLM Extraction**
Extracts entities and relationships into `data/extracted.jsonl`.
*(Note: Because of API rate limits, run a small test first using --limit)*
```bash
python extractor.py --limit 10
```

**Phase 3: Deduplication**
Merges matching nodes and semantically-identical claims, auditing merges. Saves to `data/canonical.json`.
```bash
python dedup.py
```

**Phase 4: Persistence**
Loads the canonical data into an SQLite Database (`data/graph.db`) and exports the multi-edge graph to `data/graph_export.json`
```bash
python graph_store.py
```

### 4. Running the Interfaces

**Phase 5: Retrieval API Server (FastAPI)**
Runs on `http://localhost:8000`. You can query `/retrieve` via POST for local subgraphs.
```bash
python retriever.py
```

**Phase 6: Visualization App (Streamlit)**
Run the front-end dashboard to explore the global map and inspect evidence/merges.
```bash
streamlit run app.py
```
