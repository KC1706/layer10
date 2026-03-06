# Graphiti Optimization Analysis for Layer10

This document outlines an analysis of the [Graphiti codebase](https://github.com/getzep/graphiti), an open-source framework by Zep for building temporally-aware knowledge graphs. The goal is to identify architectural patterns and optimization techniques from Graphiti that can be applied to improve the existing Layer10 Python system.

## 1. Architectural Overview & Key Differences

### 1.1 Bi-Temporal Data Model
- **Graphiti:** Implements an explicit bi-temporal model tracking both event occurrence times (`valid_at`, `invalid_at`) and graph ingestion times (`created_at`, `expired_at`). This allows for precise point-in-time queries ("What was true on Jan 1st?") and handles contradicting facts elegantly by expiring old edges rather than immediately deleting them.
- **Layer10:** Currently uses a simpler `temporal_validity` string field (e.g., "Always") per claim in SQLite.
- **Optimization:** Adopt Graphiti's bi-temporal approach by replacing the single string field with concrete `valid_at` and `invalid_at` ISO-8601 timestamps on SQLite `claims`. Add `expired_at` for audit logging instead of just the `AuditLog` JSON file.

### 1.2 Episodic Memory & Source Grounding
- **Graphiti:** Introduces the concept of `EpisodicNode` (representing the raw source document/message) and explicitly links it to extracted `EntityNode`s via `EpisodicEdge`s (MENTIONS). This provides native, strong provenance.
- **Layer10:** Uses an `evidences` table linked to `claims`.
- **Optimization:** Layer10's grounding is already strong (character-offset level), but we can adopt Graphiti's mental model by formally treating GitHub issues/PRs as "Episodes" and tracking their temporal validity window to understand *when* knowledge was introduced versus *when* it applies.

### 1.3 Driver-Agnostic Pluggable Backend
- **Graphiti:** Uses a robust `GraphDriver` abstract base class with concrete implementations for Neo4j, Kuzu, FalkorDB, and Amazon Neptune. It completely decouples the graph logic from the database dialect.
- **Layer10:** Hardcoded to SQLite (`sqlite3` module directly mixed in `graph_store.py`).
- **Optimization:** Refactor `graph_store.py` to use a driver interface. While we may stick with SQLite for simplicity, an abstract `GraphDatabaseProvider` would allow seamless future migration to Neo4j or Kuzu as the dataset grows.

### 1.4 Hybrid Search & Retrieval
- **Graphiti:** Combines semantic embedding search (cosine similarity), keyword search (BM25), and graph traversal (distance re-ranking) using Reciprocal Rank Fusion (RRF).
- **Layer10:** Relies purely on semantic `sentence-transformers` search over claim text in `retriever.py`.
- **Optimization:** Implement RRF in `retriever.py`. We can add a simple BM25 indexing pass (e.g., using `rank_bm25` library) and combine its scores with the existing semantic embeddings to drastically improve retrieval recall for exact entity names or IDs.

### 1.5 Batch Processing & Concurrency
- **Graphiti:** Uses `asyncio.Semaphore` (`semaphore_gather`) to heavily parallelize LLM extraction and Neo4j database insertions, significantly speeding up ingestion. It also features dedicated bulk endpoints (`extract_nodes_and_edges_bulk`).
- **Layer10:** Ingestion (`extractor.py`, `dedup.py`) runs synchronously in a single-threaded loop.
- **Optimization:** Convert the pipeline in `ingest.py`, `extractor.py`, and `dedup.py` to use `asyncio` and `aiohttp` (or `instructor`'s async client). Batching LLM calls will yield the highest performance gain for the entire system.

### 1.6 Deduplication Resolution Flow
- **Graphiti:** Passes candidate duplicates directly into the LLM prompt to let the model decide if a new fact contradicts or duplicates an existing one, returning structured indices.
- **Layer10:** Uses `sentence-transformers` cosine similarity with a hardcoded threshold (`0.85`), automatically merging if above the threshold.
- **Optimization:** Layer10's approach is faster and cheaper (no extra LLM calls), but Graphiti's approach is more accurate. We can implement a hybrid: use Sentence Transformers to find candidates > 0.70, and if they fall between 0.70 - 0.95, use a small LLM (like `gpt-4o-mini` or `gemini-2.5-flash`) to make the final determination.

## 2. Proposed Implementation Plan for Layer10

Based on the analysis above, here are the actionable steps to optimize the Layer10 codebase:

### Phase 8.1: Async Pipeline Conversion
- **Goal:** Dramatically reduce extraction time by parallelizing LLM calls.
- **Changes:**
  - Update `extractor.py`: Change `LitellmExtractor.extract_knowledge` to be `async` using `instructor.from_litellm(client=AsyncOpenAI())`.
  - Update `dedup.py`: Parallelize the embedding generation in `ClaimDeduplicator`.
  - Add `asyncio.gather` with a semaphore in the main ingest script to process multiple GitHub issues concurrently.

### Phase 8.2: Hybrid Retrieval (Semantic + Keyword)
- **Goal:** Improve search accuracy for specific code tokens or user names.
- **Changes:**
  - Update `requirements.txt`: Add `rank_bm25`.
  - Update `retriever.py`: When loading the graph, build a BM25 index over the claim texts alongside the embeddings.
  - Implement Reciprocal Rank Fusion (RRF) in the `get_context_pack` endpoint to combine semantic and keyword scores before returning results.

### Phase 8.3: Bi-Temporal Schema Upgrade
- **Goal:** Better handle contradictory facts and graph updates.
- **Changes:**
  - Update `graph_store.py`: Modify the `claims` SQLite table schema to replace `temporal_validity` with `valid_at`, `invalid_at`, and `expired_at` timestamp columns.
  - Update `app.py`: Reflect the new temporal fields in the Streamlit UI.

### Phase 8.4: Pluggable Storage Driver
- **Goal:** Prepare for scaling beyond SQLite.
- **Changes:**
  - Refactor `graph_store.py` to define a `StorageDriver` protocol.
  - Move the current SQLite logic into an `SQLiteDriver` class.
  - (Optional) Provide a stub `Neo4jDriver` class demonstrating how the interface works.
