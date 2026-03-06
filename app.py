import streamlit as st
import streamlit as st
import sqlite3
import requests
import pandas as pd
import streamlit.components.v1 as components
import json
from pyvis.network import Network
import os

st.set_page_config(layout="wide", page_title="Organizational Knowledge Graph")

# ---------------------------------------------------------
# Data Loaders
# ---------------------------------------------------------

@st.cache_data
def load_data(db_path="data/graph.db"):
    if not os.path.exists(db_path):
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
    conn = sqlite3.connect(db_path)
    
    # Load Entities
    entities_df = pd.read_sql_query("SELECT * FROM entities", conn)
    
    # Load Claims
    claims_query = """
    SELECT 
        c.id as claim_id, c.predicate, c.valid_at, c.invalid_at, c.expired_at,
        s.id as source_id, s.name as source_name, s.type as source_type,
        t.id as target_id, t.name as target_name, t.type as target_type
    FROM claims c
    JOIN entities s ON c.subject_id = s.id
    JOIN entities t ON c.object_id = t.id
    """
    claims_df = pd.read_sql_query(claims_query, conn)
    
    # Load Evidences
    evidences_df = pd.read_sql_query("SELECT * FROM evidences", conn)
    
    # Load Community Reports
    community_reports_df = pd.DataFrame()
    try:
        community_reports_df = pd.read_sql_query("SELECT * FROM community_reports", conn)
    except:
        pass # Table might not exist yet if Phase 12.1 wasn't run
        
    conn.close()
    return entities_df, claims_df, evidences_df, community_reports_df

# ---------------------------------------------------------
# Graph Rendering Engine
# ---------------------------------------------------------

def render_graph(entities_df, claims_df, max_nodes=100):
    net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="black", directed=True)
    
    # Filter for visualization (prevent massive browser hang)
    display_claims = claims_df.head(max_nodes)
    node_ids = set()
    
    # Add nodes and edges
    for _, row in display_claims.iterrows():
        src = row["source_id"]
        tgt = row["target_id"]
        
        if src not in node_ids:
            net.add_node(src, label=row["source_name"], title=f"Type: {row['source_type']}", color="#add8e6")
            node_ids.add(src)
            
        if tgt not in node_ids:
            net.add_node(tgt, label=row["target_name"], title=f"Type: {row['target_type']}", color="#add8e6")
            node_ids.add(tgt)
            
        edge_title = f"{row['predicate']} (Valid: {row['valid_at']})"
        net.add_edge(src, tgt, title=edge_title, label=row["predicate"])
        
    # Generate network HTML
    net.set_options("""
    var options = {
      "nodes": { "font": { "size": 16 } },
      "edges": { "font": { "size": 12, "align": "middle" }, "arrows": { "to": { "enabled": true } } },
      "physics": { "barnesHut": { "gravitationalConstant": -3000 } }
    }
    """)
    
    html_path = "data/tmp_graph.html"
    os.makedirs("data", exist_ok=True)
    net.save_graph(html_path)
    return html_path

# ---------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------

st.title("🧠 Layer10: Organizational Memory Graph")

entities_df, claims_df, evidences_df, community_reports_df = load_data()

if entities_df.empty:
    st.warning("No data found! Please run the pipeline (ingest, extract, dedup, store) first.")
    st.stop()

# Layout
col_graph, col_panel = st.columns([2, 1])

with col_graph:
    st.subheader("Global Knowledge Map")
    
    limit = st.slider("Max edges to display", 10, 500, 50)
    
    html_file = render_graph(entities_df, claims_df, max_nodes=limit)
    
    with open(html_file, 'r', encoding='utf-8') as f:
         graph_html = f.read()
         
    components.html(graph_html, height=650)
    
with col_panel:
    st.subheader("Deep Dive Panel")
    
    st.markdown("### Ask the Graph")
    query = st.text_input("Enter a query (e.g., 'Who is related to Project X?' or 'What are the main themes of the repository?')", key="query_input")
    
    search_mode = st.radio("Search Mode", ["Local Search (Graph Traversal)", "Global Search (Map-Reduce Communities)"], index=0)
    
    if st.button("Search"):
        if query:
            with st.spinner(f"Running {search_mode}..."):
                try:
                    if "Global" in search_mode:
                        payload = {"question": query, "top_k_entities": 3, "top_k_claims": 5}
                        response = requests.post("http://localhost:8000/global_search", json=payload)
                        if response.status_code == 200:
                            data = response.json()
                            st.success("Global Search Complete!")
                            st.markdown("### Answer")
                            st.write(data["answer"])
                            
                            st.markdown("### Communities Used")
                            if data.get("communities_used"):
                                for c_id in data["communities_used"]:
                                    if not community_reports_df.empty:
                                        report = community_reports_df[community_reports_df["id"] == str(c_id)]
                                        if not report.empty:
                                            with st.expander(f"Community Report {c_id}: {report.iloc[0]['title']}"):
                                                st.write(report.iloc[0]["summary"])
                                    else:
                                        st.write(f"- Community {c_id}")
                            else:
                                st.info("No relevant communities found.")
                                
                        else:
                            st.error(f"Global search failed: {response.text}")

                    else: # Local Search
                        payload = {"question": query, "top_k_entities": 3, "top_k_claims": 5}
                        response = requests.post("http://localhost:8000/retrieve", json=payload)
                        if response.status_code == 200:
                            data = response.json()
                            st.success("Local Search Complete!")
                            
                            st.markdown("### Entities Discovered (via Graph Traversal)")
                            for ent in data.get("entities", []):
                                st.write(f"- **{ent['name']}** ({ent['type']})")

                            st.markdown("### Target Claims")
                            for c in data.get("claims", []):
                                with st.expander(f"{c['subject_name']} -> {c['predicate']} -> {c['object_name']}"):
                                    for ev in c.get('evidence', []):
                                         st.markdown(f"> *{ev['exact_excerpt']}*")
                                         st.caption(f"Source: {ev['source_id']}")

                        else:
                            st.error(f"Local search failed: {response.text}")
                except Exception as e:
                     st.error(f"Failed to connect to Retriever API. Ensure uvicorn is running. Error: {e}")

    st.divider()
    tabs = st.tabs(["Evidence Hub", "Merge Inspector", "Community Reports"])
    
    with tabs[0]:
        st.markdown("**Select a Claim to view grounding:**")
        
        # In a fully interactive pyvis setup, we would capture click events.
        # Since pyvis HTML is isolated, we use a dropdown bound to the claims.
        claim_options = claims_df.apply(
            lambda x: f"{x['source_name']} -> {x['predicate']} -> {x['target_name']}", axis=1
        ).tolist()
        
        selected_idx = st.selectbox("Explore Edge:", range(len(claim_options)), format_func=lambda x: claim_options[x])
        
        if selected_idx is not None:
            selected_claim = claims_df.iloc[selected_idx]
            claim_id = selected_claim["claim_id"]
            
            st.markdown(f"**Predicate:** `{selected_claim['predicate']}`")
            st.markdown(f"**Valid At:** `{selected_claim['valid_at']}` | **Invalid At:** `{selected_claim.get('invalid_at', '')}` | **Expired At:** `{selected_claim.get('expired_at', '')}`")
            
            evs = evidences_df[evidences_df["claim_id"] == claim_id]
            st.markdown(f"**Supporting Evidence:** ({len(evs)} citations)")
            
            for _, ev in evs.iterrows():
                with st.expander(f"Source: {ev['source_id']} ({ev['timestamp']})"):
                    # Use markdown blockquote for the exact excerpt grounding
                    st.markdown(f"> *{ev['exact_excerpt']}*")
                    st.caption(f"Offsets: {ev['character_start_offset']} - {ev['character_end_offset']}")

    with tabs[1]:
        st.markdown("**Audit Canonicalized Entities**")
        if not entities_df.empty:
            merged_entities = entities_df[entities_df["aliases"] != "[]"]
            if merged_entities.empty:
                 st.info("No merged entities found.")
            else:
                 for _, ent in merged_entities.head(10).iterrows():
                     st.markdown(f"**{ent['name']}** ({ent['type']})")
                     try:
                         aliases = json.loads(ent["aliases"])
                         st.write(f"Merged Aliases: `{', '.join(aliases)}`")
                     except:
                         pass
                     st.divider()

         # Check Audit Log file if exists
        audit_path = "data/audit_log.jsonl"
        if os.path.exists(audit_path):
             st.markdown("**Recent Claim Deduplication Merges:**")
             with open(audit_path, "r") as f:
                 logs = [json.loads(line) for line in f.readlines()][-5:] # Last 5
             for log in logs:
                 if log["type"] == "claim":
                      st.caption(f"Confidence: {log['confidence_score']:.2f}")
                      st.write("Merged matching claim evidence into canonical edge.")
                      
    with tabs[2]:
        st.markdown("**Hierarchical Community Summaries**")
        if not community_reports_df.empty:
            st.info(f"Loaded {len(community_reports_df)} macro-summaries generated via Louvain graph partitioning.")
            for _, row in community_reports_df.iterrows():
                with st.expander(f"Community {row['id']}: {row['title']}"):
                    st.write(row['summary'])
                    try:
                         nodes = json.loads(row['node_ids'])
                         st.caption(f"Entities in community: {len(nodes)}")
                    except:
                         pass
        else:
            st.warning("No Community Reports found. Have you run `cluster.py`?")
             
st.sidebar.markdown(f"### Stats\n- Entities: **{len(entities_df)}**\n- Claims: **{len(claims_df)}**\n- Evidence Points: **{len(evidences_df)}**")
if not community_reports_df.empty:
    st.sidebar.markdown(f"- Communities: **{len(community_reports_df)}**")
