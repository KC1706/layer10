# Layer10: End-to-End Testing Guide

This guide walks you through testing the entire Layer10 AI Knowledge Graph pipeline from end-to-end. Layer10 is composed of several sequential data processing stages culminating in a graph-backed Retrieval Augmented Generation (GraphRAG) visualization and search UI.

## Prerequisites

### 1. Environment Setup
Activate your Python environment and install the required dependencies (if you haven't already):
```bash
pip install -r requirements.txt
```

### 2. API Keys
The pipeline relies heavily on LLM calls (via `litellm`) for extraction, deduplication resolution, and community summarization. You must export your API key before running the scripts.

For Gemini:
```bash
export GEMINI_API_KEY="your-api-key-here"
```

For OpenAI (if you configured the models to use OpenAI):
```bash
export OPENAI_API_KEY="your-api-key-here"
```

---

## The Pipeline Execution

Run these commands in sequential order from the root `layer10` directory.

### Step 1: Data Ingestion
Fetches your source documents, normalizes them, and chunks them using the overlapping text chunker to prevent context loss.

```bash
python ingest.py --repo "your/target_repo" --output data/corpus.jsonl
```
*(Note: `ingest.py` defaults to pulling GitHub Issues from `tiangolo/fastapi` if no `--repo` flag is provided).*

### Step 2: Information Extraction
Uses the LLM to structurally extract Entities, Claims, and Evidence from the chunked text via the async LiteLLM + Instructor pipeline.

```bash
python extractor.py --input data/corpus.jsonl --output data/extracted.jsonl
```

### Step 3: Deduplication & Canonicalization
Runs fuzzy matching for entities and semantic similarity matching for claims. Resolves high-confidence conflicts automatically using an LLM to decide whether to `ADD`, `UPDATE`, `DELETE`, or do `NONE`.

```bash
python dedup.py --input data/extracted.jsonl --output data/canonical.json
```

### Step 4: Graph Database Construction
Ingests the canonicalized data into the SQLite Bi-Temporal graph schema (`graph.db`) and exports a snapshot for visualizations.

```bash
python graph_store.py --input data/canonical.json --db data/graph.db --export data/graph_export.json
```

### Step 5: Hierarchical Community Summarization (GraphRAG)
Runs the Louvain community detection algorithm over your newly formed graph to cluster related entities, then prompts the LLM to generate macro-summaries of each community.

```bash
python cluster.py --db data/graph.db
```

---

## Running the Servers (Testing the Output)

Once the database (`data/graph.db`) is fully built and clustered, you need to spin up the API and the UI. These should be run in **two separate terminal windows**.

### Terminal Window 1: Start the Retrieval API
The `retriever.py` FastAPI server handles the complex backend query logic, hybrid RRF search, and Map-Reduce global evaluation.

```bash
# Ensure your API key is exported in this terminal too!
export GEMINI_API_KEY="your-api-key-here"

# Start the server
uvicorn retriever:app --host 0.0.0.0 --port 8000 --reload
```
*You can verify the API is running by visiting `http://localhost:8000/docs` in your browser to see the interactive Swagger UI.*

### Terminal Window 2: Start the Streamlit UI
The Streamlit app connects to your local FastAPI server to execute searches and visualize the graph.

```bash
# Start the Streamlit application
streamlit run app.py
```

---

## How to Test the Application

Once the Streamlit UI opens in your browser (`http://localhost:8501`), perform the following tests to verify all the optimizations:

### Test 1: Local Search (Graph Traversal + CoT)
1. In the sidebar, select **Search Mode: Local Search (Graph Traversal)**.
2. Ask a specific question about an entity in your dataset (e.g., *"What features did John Smith develop?"*).
3. **Verify:** Look at the retrieved answers and the PyVis network graph. You should see a highly localized cluster of nodes explicitly relating to your query.
4. **Verify Temporal Data (Memento Optimization):** Hover over connections/edges or look in the "Evidence" expander payload to verify that claims possess `valid_from`, `valid_to`, `version`, and `confidence_score` dimensions. 

### Test 2: Global Search (Map-Reduce Communities)
1. In the sidebar, toggle **Search Mode: Global Search (Map-Reduce)**.
2. Ask a broad, macroscopic question about the whole dataset (e.g., *"What are the overarching themes, major architectural decisions, or technical debt patterns documented?"*).
3. **Verify:** The system will dispatch concurrent "Map" prompts to evaluate the community summaries generated in Step 5.
4. **Verify Evidence:** In the UI underneath the answer, you should see the raw, LLM-generated `Community Reports` that the Map-Reduce pipeline used to synthesize the answer.

### Test 3: System Resilience & Deduplication (Advanced)
To test the active LLM deduplication and Memento Confidence Decay:
1. Run `ingest.py`, `extractor.py`, and `dedup.py` again on a slightly modified dataset containing contradictory information.
2. **Verify:** In your `graph.db`, old edges will not be deleted; instead, their `valid_to` column will be stamped with a timestamp, while the new conflicting edge is inserted with an incremented `version`.
3. Over time, or by running multiple duplicate ingestions, observe how canonical claims have their `last_observed_at` values updated, reinforcing their `confidence_score` during retrieval queries.
