# Cognee Optimization Analysis for Layer10

This document contains an architectural analysis of the [Cognee codebase](https://github.com/topoteretes/cognee), an open-source framework optimized for turning documents into connected AI memory through GraphRAG. The objective is to identify techniques from Cognee that can be leveraged to further optimize the Layer10 knowledge graph system.

## 1. Architectural Overview & Key Differences

### 1.1 Text Chunker with Overlap
- **Cognee:** Cognee implements a `TextChunkerWithOverlap` in `cognee/modules/chunking/text_chunker_with_overlap.py`. Instead of splitting text using hard stops or generic Langchain Recursive splitters, it builds chunks paragraph by paragraph and maintains a configurable `chunk_overlap_ratio`. If a relationship spans two paragraphs that get separated by a hard split, a non-overlapping chunker will miss it entirely. The overlap ensures the LLM sees the complete context of an edge relation.
- **Layer10:** `extractor.py` currently assumes we pass the whole text document in one go or splits it externally without any overlap guarantees.
- **Optimization Strategy:** Introduce an `overlap_chunker.py` utility. Before sending text to `LitellmExtractor` in Layer10, we should process large text blobs into chunks bounded by a maximum character limit (e.g., 2000 chars) with a fixed 15% overlap. This ensures entities that exist across the boundary of a chunk are correctly mapped into the final Knowledge Graph.

### 1.2 Chain-of-Thought (CoT) Graph Completion Retriever
- **Cognee:** Implements a heavily optimized `GraphCompletionCotRetriever` residing in `cognee/modules/retrieval/graph_completion_cot_retriever.py`. This isn't just a standard semantic search. It is an agentic _iterative_ retriever. It performs an initial search, asks the LLM to generate an answer along with "follow-up questions" (validation prompts), and then dynamically queries the graph _again_ based on those follow-up questions to pull deeper multi-hop triplets before generating the final answer.
- **Layer10:** Currently has a static 1-hop expansion in `retriever.py` (added during Phase 10). While good, it fetches the direct 1-hop neighborhood unconditionally, regardless of whether that neighborhood actually helps answer the specific user question.
- **Optimization Strategy:** Enhance `retriever.py` to support `multi_hop_query` functionality. Instead of just grabbing the 1-hop neighborhood blindly, we implement a lightweight 2-iteration loop in the retrieval endpoint.
  - Phase A: Fetch initial facts via BM25/Semantic.
  - Phase B: Pass facts to an LLM explicitly asking: "Based on these facts, what follow-up entity relationships do you need to answer the user's question?".
  - Phase C: Run a fast SQL lookup for the exact entities requested by the LLM, and yield the final `ContextPack`.

### 1.3 Telemetry & Search Logging
- **Cognee:** Their central `search.py` entrypoint actively logs all queries and telemetry (e.g., query latency, hit rates) against a `tenant_id` and `user.id`.
- **Layer10:** Currently completely lacks observability in `app.py` or `retriever.py`. 
- **Optimization Strategy:** Add basic search logging to `retriever.py`, storing the `question` and `retrieved_claim_ids` in a new SQLite table called `search_logs`. This allows administrators to see what users are searching for and if the graph had the answers.

## 2. Proposed Implementation Plan for Layer10

Based on the analysis, we can plan the following optimizations for Phase 11:

### Phase 11.1: Overlapping Text Chunker
- Create a new utility file `chunker.py` that splits text based on a max character limit and maintains an overlap percentage.
- Update `app.py` or the ingestion script to use this chunker *before* sending pieces to `extractor.py`.
- **Why:** Massive increase in edge extraction stability near chunk boundaries.

### Phase 11.2: CoT Multi-Hop Retrieval
- Redesign `get_context_pack()` inside `retriever.py`.
- Keep the current Hybrid Search (Semantic + keyword via RRF).
- Add an intermediate LLM call using `instructor` that reviews the Top 5 hits and returns a structured Pydantic object: `FollowupRequest(entities_to_explore=["entityA", "entityB"])`.
- Perform the 1-hop SQL expansion *only* against the smartly selected `entities_to_explore`.
- **Why:** Prevent context-window bloat by intelligently pulling only the sub-graph paths the LLM specifically asks for.

### Phase 11.3: Retrieval Observability
- Add a new table `search_logs(id, timestamp, query, hits_json)` to `graph_store.py`.
- Update `retriever.py` to `INSERT` into this table at the end of every `/retrieve` API call.
- **Why:** Brings production-level observability to the knowledge engine.

## Summary
Cognee excels in iterative reasoning and robust ingestion handling. By porting their overlapping chunk bounds and multi-hop Chain-of-Thought retrieval strategies, Layer10 will be able to handle significantly larger text inputs without losing edge relations, and answer far more complex cross-document user queries.
