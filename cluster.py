import os
import json
import sqlite3
import argparse
import asyncio
import instructor
import litellm
import networkx as nx
import networkx.algorithms.community as nx_comm
from typing import List, Dict, Any
from pydantic import BaseModel, Field

class CommunitySummary(BaseModel):
    title: str = Field(description="A short, descriptive title for this community of entities and concepts.")
    summary: str = Field(description="A comprehensive summary explaining the common themes, dependencies, and core insights connecting this group of entities. Write this in a paragraph format as if producing a research report.")

async def generate_community_summary(community_nodes: set, graph: nx.MultiDiGraph, model: str) -> CommunitySummary:
    """
    Given a set of nodes belonging to a specific community, pull all internal 
    graph edges and prompt the LLM to write a comprehensive report summarizing them.
    """
    client = instructor.from_litellm(litellm.completion, mode=instructor.Mode.MD_JSON)
    
    # 1. Gather all internal edges to formulate context
    subgraph = graph.subgraph(community_nodes)
    
    context_lines = []
    # Identify entities to provide rich node descriptions
    for n in subgraph.nodes(data=True):
        node_id, data = n
        name = data.get('name', str(node_id))
        node_type = data.get('type', 'Unknown')
        context_lines.append(f"Entity: {name} ({node_type})")
        
    context_lines.append("\nRelationships within community:")
    for u, v, k, data in subgraph.edges(keys=True, data=True):
        u_name = subgraph.nodes[u].get('name', str(u))
        v_name = subgraph.nodes[v].get('name', str(v))
        pred = data.get('predicate', 'related to')
        context_lines.append(f"- {u_name} {pred} {v_name}")
        
    context_text = "\n".join(context_lines)
    
    # 2. Instruct the LLM to act as a research analyst
    system_prompt = (
        "You are an expert intelligence analyst examining an isolated community within a larger knowledge graph. "
        "Your goal is to understand how these specific entities relate to one another and synthesize "
        "a high-level executive summary report for this community. Discover the implicit macro-themes connecting these explicit micro-facts."
    )
    user_prompt = f"Analyze the following subgraph and produce a community report:\n\n{context_text}"
    
    try:
        # For huge communities, you might hit token windows. Ideally, chunking or LLM choice handles this.
        response, completion_kwargs = await client.chat.completions.create_with_completion(
            model=model,
            response_model=CommunitySummary,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_retries=3
        )
        return response
    except Exception as e:
        print(f"Failed to generate summary for community: {e}")
        return CommunitySummary(title="Unknown Theme", summary="Failed to generate summary due to API error or context length.")

async def process_communities(graph: nx.MultiDiGraph, model: str, db_path: str):
    """
    Detects communities using Louvain partition, generates LLM summaries concurrently,
    and inserts them into the backing SQLite storage.
    """
    # 1. Detection Phase (Louvain greedy modularity maximization)
    print("Running Louvain community detection algorithm...")
    # Convert directed multigraph to undirected simple graph for partitioning
    G_undirected = nx.Graph(graph) 
    communities = nx_comm.louvain_communities(G_undirected)
    print(f"Graph partitioned into {len(communities)} distinct hierarchical communities.")
    
    # Filter out single-node communities to avoid noise
    valid_communities = [c for c in communities if len(c) > 1]
    print(f"Found {len(valid_communities)} communities containing 2+ nodes.")
    
    # 2. Summarization Phase
    db = sqlite3.connect(db_path)
    cursor = db.cursor()
    cursor.execute("DELETE FROM community_reports") # Clears existing
    db.commit()

    semaphore = asyncio.Semaphore(5)  # Limit concurrent LLM calls
    
    async def process_community(community_id, nodes):
        async with semaphore:
            print(f"Generating report for Community {community_id} ({len(nodes)} nodes)...")
            report = await generate_community_summary(nodes, graph, model)
            
            # Persist to local GraphStore SQLite DB
            node_ids_json = json.dumps(list(nodes))
            cursor.execute('''
                INSERT INTO community_reports (id, title, summary, node_ids)
                VALUES (?, ?, ?, ?)
            ''', (str(community_id), report.title, report.summary, node_ids_json))
            print(f" -> Saved Report '{report.title}'")

    tasks = []
    for i, comm in enumerate(valid_communities):
        tasks.append(process_community(i, comm))
        
    await asyncio.gather(*tasks)
    db.commit()
    db.close()
    print("Community Summarization complete.")

def main():
    parser = argparse.ArgumentParser(description="Phase X: GraphRAG Community Detection & Summarization")
    parser.add_argument("--input", type=str, default="data/graph_export.json", help="Exported NetworkX graph json")
    parser.add_argument("--db", type=str, default="data/graph.db", help="SQLite database to insert reports into")
    parser.add_argument("--model", type=str, default="gemini/gemini-1.5-flash", help="LLM for summarization")
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Could not find graph export at {args.input}. Did you run Phase 4?")
        return
        
    print(f"Loading graph from {args.input}...")
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)
        graph = nx.node_link_graph(data)
        
    asyncio.run(process_communities(graph, args.model, args.db))

if __name__ == "__main__":
    main()
