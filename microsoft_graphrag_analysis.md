# Microsoft GraphRAG Optimization Analysis for Layer10

This document contains an architectural analysis of the [Microsoft GraphRAG codebase](https://github.com/microsoft/graphrag), an advanced data pipeline suite built to extract and search structured knowledge graphs using LLMs. The goal is to identify patterns and techniques from Microsoft's implementation that can be leveraged to optimize the Layer10 knowledge graph system.

## 1. Architectural Overview & Key Differences

### 1.1 Hierarchical Community Detection (Leiden Algorithm)
- **GraphRAG:** Microsoft's indexing pipeline heavily relies on grouping the extracted entities and relationships into hierarchical communities using the Leiden algorithm (found in `cluster_graph` / `create_communities.py`). Once clustered, it generates an LLM summary ("Community Report") for every single community at every level of the hierarchy.
- **Layer10:** Currently flat. All entities and claims exist in a single global namespace. There is no concept of subsets or higher-level abstractions of the data.
- **Optimization Strategy:** Introduce a `community_builder.py` script. After processing all documents and extracting the graph, we run a clustering algorithm (e.g., using NetworkX's community modules or python-leidenalg) over our graph to detect communities, and then prompt an LLM to generate a summary for each community. Store these summaries in a new `communities` SQLite table.

### 1.2 Dual-Mode Retrieval (Global vs. Local Search)
- **GraphRAG:** Defines two highly distinct search modes:
  - **Local Search (`local_search/search.py`):** Starts from specific entities matched to the user's query and traverses immediate neighbors (similar to our current Phase 10/11 CoT multi-hop search).
  - **Global Search (`global_search/search.py`):** Uses a heavy Map-Reduce pattern. It takes the user's question and maps it across *all* Community Reports generated in the indexing phase asking the LLM to rate the importance of that community to the question. It then concatenates the most valuable points and Reduces them into a final answer. This is explicitly designed for macroscopic questions like "What are the main themes of this entire repository?", which Local Search completely fails at.
- **Layer10:** Currently only has Local Search (Hybrid BM25/Semantic matching -> 1 Hop / CoT Graph traversal). Global queries asking about the entire corpus fail because semantic search forces the retriever into localized pockets.
- **Optimization Strategy:** Implement a `/global-search` endpoint in `retriever.py`.
  - Phase A: Fetch all Community Report summaries from the database.
  - Phase B (Map): Concurrently ask the LLM to score (0-100) how relevant each community report is to the user query and extract key points.
  - Phase C (Reduce): Gather the highest-scored points and pass them to the LLM to generate a comprehensive global answer.

## 2. Proposed Implementation Plan for Layer10

Based on the analysis, we can plan the following optimizations for Phase 12:

### Phase 12.1: Graph Community Detection & Summarization
- Add a script `cluster.py`.
- Read the networkx graph from `data/graph_export.json` (or build it from SQLite).
- Use `networkx.algorithms.community.louvain_communities` (or standard greedy modularity) to partition the nodes into communities.
- For each group of nodes, pull their internal claims and prompt the LLM: "Write a high-level summary report for this community of entities."
- Save the results into a new `community_reports(id, title, summary, node_ids)` table in `graph_store.py`.
- **Why:** Essential prerequisite for answering dataset-wide macroscopic queries.

### Phase 12.2: Map-Reduce Global Search
- Add a new API boundary in `retriever.py`: `get_global_context()`.
- If the user selects "Global Search" in the Streamlit App:
  - The retriever pulls *all* `community_reports`.
  - Uses `asyncio.gather` and `instructor` to run a concurrent "Map" prompt across chunks of the reports: "Evaluate if these community reports help answer the query. Return points and a score."
  - Collects points with score > 0, sorts by score, and feeds the top N tokens to the final Reduce prompt.
- **Why:** unlocks the ability to ask "What are the general themes of the project?" or "Summarize the major feature requests", which graph traversal alone cannot answer.

### Phase 12.3: Streamlit UI Integration
- Update `app.py` to add a radio button "Search Mode: [Local, Global]".
- Based on the mode, call the respective retrieval pipeline.
- Render the `community_reports` used in the Global Search as the "evidence".

## Summary
Microsoft GraphRAG differentiates itself primarily through its heavy reliance on hierarchical summarization. By clustering our flat graph into communities and generating macro-summaries, Layer10 will gain the ability to answer broad, corpus-wide questions using a Map-Reduce global search pattern, completing the graph's capability to operate as a true narrative knowledge engine.
