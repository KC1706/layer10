# Mem0 Optimization Analysis for Layer10

This document contains an architectural analysis of the [Mem0 codebase](https://github.com/mem0ai/mem0), an open-source framework optimized for building personalized, long-term memory for AI agents. The objective is to identify techniques from Mem0 that can be adopted to further optimize the Layer10 knowledge graph system.

## 1. Architectural Overview & Key Differences

### 1.1 LLM-Driven Memory Curation (CRUD Operations)
- **Mem0:** Uses a sophisticated `DEFAULT_UPDATE_MEMORY_PROMPT` to manage state. Instead of just embedding strings and keeping them forever, Mem0 runs a continuous LLM compilation step. It retrieves existing memory facts, compares them with new facts, and explicitly tells the LLM to return `ADD`, `UPDATE`, `DELETE`, or `NONE` operations.
- **Layer10:** Currently uses `sentence-transformers` semantic similarity to deduplicate claims (Phase 3). While fast, if a new issue contradicts an old issue (e.g., "Feature X is planned" vs "Feature X is cancelled"), Layer10 will likely store both as separate claims or incorrectly merge them as the same topic if their vectors are close.
- **Optimization Strategy:** Introduce an active LLM-based Conflict Resolution step in `dedup.py`. For claims that have high semantic similarity (e.g., > 0.75), pass them to an LLM to accurately determine if the new claim `UPDATE`s, `DELETE`s, or is simply `NONE` (redundant) compared to the old claim. 

### 1.2 Multi-Level State Segregation (User, Session, Agent)
- **Mem0:** Memory is explicitly partitioned across `user_id`, `agent_id`, and `run_id`. This allows rapid filtering before vector search, ensuring that context boundaries are strictly maintained.
- **Layer10:** Everything goes into a global organizational memory graph.
- **Optimization Strategy:** Our graph operates at an organizational level, but we can adopt a "Scope" or "Project" boundary. We could add a `project_id` or `domain` column to the `claims` and `entities` tables. The `KnowledgeRetriever` can then accept a `scope` parameter to strictly filter out irrelevant domains before applying Vector/BM25 search, improving precision.

### 1.3 Categorized Extraction Prompts
- **Mem0:** Implements highly specialized extraction prompts (`USER_MEMORY_EXTRACTION_PROMPT`, `AGENT_MEMORY_EXTRACTION_PROMPT`) that instruct the LLM on exactly *what* categories of information to look for (e.g., Preferences, Capabilities, Plans).
- **Layer10:** Uses a generic prompt in `extractor.py`: "Extract entities and relationship claims from the provided text."
- **Optimization Strategy:** Update the `LitellmExtractor` system prompt in `extractor.py` to provide a defined taxonomy of organizational knowledge to look for (e.g., "Software Architecture Decisions", "Team Dependencies", "Project Timelines", "Known Bugs"). This will yield much higher quality and more structured claims.

### 1.4 Graph Search with Distance Thresholds
- **Mem0:** The `MemoryGraph` module implements a Neo4j traverser that retrieves nodes based on cosine similarity and then follows the relationships to a predefined depth (usually 1).
- **Layer10:** Our hybrid retriever fetches claims via exact match (BM25) and semantic similarity. However, it currently returns just the exact matched claims.
- **Optimization Strategy:** Enhance `retriever.py` to implement a "1-hop neighborhood" expansion. When top claims or entities are identified by Hybrid Search, query SQLite to retrieve all *connected* claims to those entities, giving the final LLM response generation a much broader and fully contextualized subgraph.

## 2. Proposed Implementation Plan for Layer10

Based on the analysis above, we can plan the following targeted optimizations:

### Phase 10.1: LLM-Driven Deduplication (CRUD Resolution)
- Modify `dedup.py` to upgrade the semantic deduplicator.
- Instead of auto-merging evidence based solely on a cosine score threshold, use an async LLM call (`gemini-1.5-flash` or similar) to evaluate the existing claim vs the new claim, instructing it to evaluate if the new claim should overwrite or void the old claim, or if it's an entirely new fact.
- This will heavily rely on the `valid_at` and `invalid_at` properties we added from the Graphiti analysis!

### Phase 10.2: Taxonomic Extraction Prompting
- Update `extractor.py` `system_prompt`. Let's borrow Mem0's instructional pattern and define explicit categories of engineering knowledge we care about (Architectural Decisions, Feature Requests, Technical Debt).

### Phase 10.3: Graph 1-Hop Expansion Retrieval
- Update `retriever.py`. After finding top `claim_id`s and `entity_id`s using RRF (Reciprocal Rank Fusion), perform an SQLite `SELECT` to fetch all edge relationships where `subject_id` or `object_id` is in the returned set. Add these 1-hop claims to the `ContextPack`.

## Summary
While Graphiti provided the structural blueprint for Bi-Temporal Graphing, Mem0 provides the intelligence blueprint for maintaining that graph dynamically. By integrating Mem0's LLM-driven CRUD memory resolution and taxonomic extraction techniques, Layer10's graph will be highly resilient to conflicting or redundant information.
