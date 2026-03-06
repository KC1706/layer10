import sqlite3
import argparse
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi
import torch
import numpy as np
import uvicorn
import instructor
import litellm
from pydantic import Field

# ---------------------------------------------------------
# API Models
# ---------------------------------------------------------

class QueryRequest(BaseModel):
    question: str
    top_k_entities: int = 3
    top_k_claims: int = 5

class EvidenceResponse(BaseModel):
    exact_excerpt: str
    timestamp: str
    source_id: str

class ClaimResponse(BaseModel):
    claim_id: str
    subject_name: str
    predicate: str
    object_name: str
    valid_at: str
    invalid_at: str
    expired_at: str
    evidence: List[EvidenceResponse]

class ContextPack(BaseModel):
    question: str
    entities: List[Dict[str, str]]
    claims: List[ClaimResponse]

class FollowupRequest(BaseModel):
    entities_to_explore: List[str] = Field(description="List of specific entity names we must lookup to answer the user query")

class GlobalSearchResult(BaseModel):
    score: int = Field(description="How relevant is this community to answering the user query? (0-100)")
    points: List[str] = Field(description="A list of bullet points extracted from the community report that help answer the query.")

class GlobalSearchResponse(BaseModel):
    question: str
    answer: str
    communities_used: List[str]

# ---------------------------------------------------------
# Retriever Engine
# ---------------------------------------------------------

class KnowledgeRetriever:
    def __init__(self, db_path: str, model_name: str = "all-MiniLM-L6-v2"):
        self.db_path = db_path
        print(f"Loading SentenceTransformer: {model_name}...")
        self.model = SentenceTransformer(model_name)
        
        # In a real production system, you would load embeddings into a VectorDB (Qdrant, Milvus, FAISS)
        # For this prototype, we will load text representations into memory and search dynamically.
        self._load_search_index()
        
    def _load_search_index(self):
        """Loads entities and claims into memory for embedding search."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Load Entities
        self.entities = []
        cursor.execute("SELECT id, name, type, aliases FROM entities WHERE valid_to IS NULL")
        for row in cursor.fetchall():
            self.entities.append(dict(row))
            
        # Format entity strings for search (name + type + aliases)
        self.entity_texts = [
            f"{e['name']} {e['type']} {e['aliases']}" for e in self.entities
        ]
        if self.entity_texts:
            self.entity_embs = self.model.encode(self.entity_texts, convert_to_tensor=True)
            tokenized_entities = [text.lower().split() for text in self.entity_texts]
            self.entity_bm25 = BM25Okapi(tokenized_entities)
        else:
            self.entity_embs = None
            self.entity_bm25 = None
        
        # Load Claims (joined with subject/object names)
        self.claims = []
        query = """
            SELECT 
                c.id as claim_id, 
                s.name as subject_name, 
                c.predicate, 
                o.name as object_name, 
                c.valid_at,
                c.invalid_at,
                c.expired_at,
                c.confidence_score,
                c.last_observed_at
            FROM claims c
            JOIN entities s ON c.subject_id = s.id
            JOIN entities o ON c.object_id = o.id
            WHERE c.valid_to IS NULL
        """
        cursor.execute(query)
        for row in cursor.fetchall():
            claim = dict(row)
            # Load evidences for this claim
            cursor.execute("""
                SELECT source_id, exact_excerpt, timestamp 
                FROM evidences WHERE claim_id = ?
                ORDER BY timestamp DESC
            """, (claim["claim_id"],))
            claim["evidence"] = [dict(e) for e in cursor.fetchall()]
            self.claims.append(claim)
            
        # Format claim strings for search
        self.claim_texts = [
            f"{c['subject_name']} {c['predicate']} {c['object_name']}" for c in self.claims
        ]
        if self.claim_texts:
            self.claim_embs = self.model.encode(self.claim_texts, convert_to_tensor=True)
            tokenized_claims = [text.lower().split() for text in self.claim_texts]
            self.claim_bm25 = BM25Okapi(tokenized_claims)
        else:
            self.claim_embs = None
            self.claim_bm25 = None
            
        cursor.execute("SELECT id, title, summary FROM community_reports")
        self.community_reports = [dict(row) for row in cursor.fetchall()]

        conn.close()

    def get_context_pack(self, question: str, top_k_entities: int = 3, top_k_claims: int = 5) -> Dict[str, Any]:
        """
        Retrieves the most semantically relevant subgraph (entities + claims + strict evidence)
        to answer the user's question. Uses Hybrid Search (Semantic + BM25) and RRF.
        """
        question_emb = self.model.encode(question, convert_to_tensor=True)
        tokenized_query = question.lower().split()
        
        def compute_rrf(semantic_scores, bm25_scores, k=60):
            semantic_ranks = {idx.item(): rank for rank, idx in enumerate(torch.argsort(semantic_scores, descending=True))}
            bm25_ranks = {idx: rank for rank, idx in enumerate(np.argsort(bm25_scores)[::-1])}
            
            rrf_scores = {}
            for i in range(len(semantic_scores)):
                rrf_scores[i] = 1.0 / (k + semantic_ranks[i]) + 1.0 / (k + bm25_ranks[i])
            return rrf_scores

        # 1. Search Entities
        top_entities = []
        if self.entity_embs is not None:
             cos_scores = util.cos_sim(question_emb, self.entity_embs)[0]
             bm25_scores = self.entity_bm25.get_scores(tokenized_query)
             rrf_scores = compute_rrf(cos_scores, bm25_scores)
             
             sorted_indices = sorted(rrf_scores.keys(), key=lambda i: rrf_scores[i], reverse=True)[:top_k_entities]
             
             for idx in sorted_indices:
                 if rrf_scores[idx] > 0.001: 
                     top_entities.append({
                         "id": self.entities[idx]["id"],
                         "name": self.entities[idx]["name"],
                         "type": self.entities[idx]["type"],
                         # Cast to string so it matches the response schema (Dict[str, str])
                         "score": str(float(rrf_scores[idx]))
                     })
                     
        # 2. Search Claims
        top_claims = []
        if self.claim_embs is not None:
            cos_scores = util.cos_sim(question_emb, self.claim_embs)[0]
            bm25_scores = self.claim_bm25.get_scores(tokenized_query)
            rrf_scores = compute_rrf(cos_scores, bm25_scores)
            
            import datetime
            import math
            now = datetime.datetime.now()
            decay_half_life_days = 30
            decay_factor = math.log(0.5) / (decay_half_life_days * 24 * 3600)
            
            decayed_rrf_scores = {}
            for idx in rrf_scores:
                claim = self.claims[idx]
                try:
                    last_observed = datetime.datetime.fromisoformat(claim.get("last_observed_at", now.isoformat()))
                except Exception:
                    last_observed = now
                age_seconds = max(0, (now - last_observed).total_seconds())
                decayed_confidence = max(0.1, claim.get("confidence_score", 1.0) * math.exp(decay_factor * age_seconds))
                
                decayed_rrf_scores[idx] = rrf_scores[idx] * decayed_confidence
            
            sorted_indices = sorted(decayed_rrf_scores.keys(), key=lambda i: decayed_rrf_scores[i], reverse=True)[:top_k_claims]
            
            for idx in sorted_indices:
                 if decayed_rrf_scores[idx] > 0.001: 
                     claim = self.claims[idx]
                     top_claims.append({
                         "claim_id": claim["claim_id"],
                         "subject_name": claim["subject_name"],
                         "predicate": claim["predicate"],
                         "object_name": claim["object_name"],
                         "valid_at": claim["valid_at"],
                         "invalid_at": claim["invalid_at"],
                         "expired_at": claim["expired_at"],
                         "score": float(decayed_rrf_scores[idx]),
                         "evidence": claim["evidence"]
                     })

        # 3. CoT Multi-Hop Graph Context Expansion
        client = instructor.from_litellm(litellm.completion, mode=instructor.Mode.MD_JSON)
        
        context_lines = []
        for c in top_claims:
            context_lines.append(f"- {c['subject_name']} {c['predicate']} {c['object_name']}")
            
        system_prompt = "You are an intelligent knowledge graph retrieval reasoning agent. Based on the initial semantic claims retrieved below, what other specific entity names must we look up in the graph to fully answer the user query? If you have enough info, return an empty list."
        user_prompt = f"User Query: {question}\n\nInitial Claims:\n" + "\n".join(context_lines)
        
        try:
            followup = client.chat.completions.create(
                model="gemini/gemini-2.5-flash",
                response_model=FollowupRequest,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_retries=2
            )
            entities_to_explore = set(followup.entities_to_explore)
        except Exception as e:
            print(f"CoT request failed: {e}")
            entities_to_explore = set()

        if not entities_to_explore and (top_claims or top_entities):
            seed_entity_names = set(e["name"] for e in top_entities)
            for c in top_claims:
                seed_entity_names.add(c["subject_name"])
                seed_entity_names.add(c["object_name"])
            entities_to_explore = seed_entity_names
                
        top_claim_ids = set(c["claim_id"] for c in top_claims)
        expanded_claims = []
        
        for claim in self.claims:
            if claim["claim_id"] not in top_claim_ids:
                if claim["subject_name"] in entities_to_explore or claim["object_name"] in entities_to_explore:
                    expanded_claims.append({
                         "claim_id": claim["claim_id"],
                         "subject_name": claim["subject_name"],
                         "predicate": claim["predicate"],
                         "object_name": claim["object_name"],
                         "valid_at": claim["valid_at"],
                         "invalid_at": claim["invalid_at"],
                         "expired_at": claim["expired_at"],
                         "score": 0.0001,  # Minor score to signify graph expansion
                         "evidence": claim["evidence"]
                    })
            
        # Cap the expansion so we don't blow up the LLM token context with a massive dense subgraph node
        top_claims.extend(expanded_claims[:10])
        
        return {
            "question": question,
            "entities": top_entities,
            "claims": top_claims
        }

    async def get_global_context(self, question: str) -> Dict[str, Any]:
        """
        Executes a Map-Reduce global search across all hierarchical community 
        reports to answer broad, corpus-wide questions.
        """
        import asyncio
        client = instructor.from_litellm(litellm.completion, mode=instructor.Mode.MD_JSON)
        model = "gemini/gemini-2.5-flash"

        # --- MAP PHASE ---
        map_system_prompt = (
            "You are an expert intelligence analyst. You will be provided with a specific "
            "community report from a knowledge graph. Determine how relevant this community is "
            "to answering the user's question on a scale of 0-100. If it is relevant (score > 0), "
            "extract the specific key points that help answer the query."
        )

        async def map_report(report: Dict[str, Any]):
            user_prompt = f"User Question: {question}\n\nCommunity Report: {report['title']}\n{report['summary']}"
            try:
                response = await client.chat.completions.create(
                    model=model,
                    response_model=GlobalSearchResult,
                    messages=[
                        {"role": "system", "content": map_system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_retries=2
                )
                return {"report_id": report["id"], "score": response.score, "points": response.points}
            except Exception as e:
                print(f"Map phase failed for report {report['id']}: {e}")
                return {"report_id": report["id"], "score": 0, "points": []}

        # Concurrently map all reports
        map_tasks = [map_report(report) for report in self.community_reports]
        map_results = await asyncio.gather(*map_tasks)

        # --- REDUCE PHASE ---
        # Filter and sort the mapped results
        valid_results = [res for res in map_results if res["score"] > 0 and res["points"]]
        valid_results.sort(key=lambda x: x["score"], reverse=True)
        
        if not valid_results:
             return {
                 "question": question,
                 "answer": "I do not have enough information across any of the graph communities to answer this question.",
                 "communities_used": []
             }

        # Gather the top points into a final payload (protecting against context bloat)
        reduce_context = []
        communities_used = []
        token_estimate = 0
        
        for res in valid_results:
             points_text = f"From Community {res['report_id']} (Score: {res['score']}):\n" + "\n".join(f"- {p}" for p in res["points"])
             
             # Rough token estimation (4 chars ~ 1 token)
             if token_estimate + (len(points_text) / 4) > 15000:
                 break
                 
             reduce_context.append(points_text)
             communities_used.append(res['report_id'])
             token_estimate += len(points_text) / 4

        reduce_system_prompt = (
            "You are a master synthesizer generating a final global report. "
            "Based on the gathered data points from various intelligence communities below, "
            "write a comprehensive, integrated answer to the user's overarching question."
        )
        reduce_user_prompt = f"User Question: {question}\n\Gathered Intelligence:\n" + "\n\n".join(reduce_context)

        # Final generation
        # NOTE: standard completion, not instructor models
        try:
             final_response = await litellm.acompletion(
                 model="gemini/gemini-2.5-flash", # Use a valid Gemini model for synthesis
                 messages=[
                     {"role": "system", "content": reduce_system_prompt},
                     {"role": "user", "content": reduce_user_prompt}
                 ]
             )
             answer_text = final_response.choices[0].message.content
        except Exception as e:
             answer_text = f"Reduce phase failed: {e}"

        return {
             "question": question,
             "answer": answer_text,
             "communities_used": communities_used
        }

# ---------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------

app = FastAPI(title="Knowledge Graph Retrieval API", version="1.0")

# Global instance
retriever = None

@app.on_event("startup")
def startup_event():
    global retriever
    db_path = "data/graph.db"
    try:
        import torch
    except ImportError:
         print("Warning: torch not found. This might impact sentence-transformers performance.")
    retriever = KnowledgeRetriever(db_path=db_path)
    print("Retriever initialized.")

@app.post("/retrieve", response_model=ContextPack)
def retrieve_context(query: QueryRequest):
    if not retriever:
        raise HTTPException(status_code=503, detail="Retriever not initialized")
    
    try:
        result = retriever.get_context_pack(
            question=query.question, 
            top_k_entities=query.top_k_entities,
            top_k_claims=query.top_k_claims
        )
        
        # Log telemetry observability
        try:
            import datetime
            import json
            hit_ids = [c["claim_id"] for c in result["claims"]]
            conn = sqlite3.connect("data/graph.db")
            cursor = conn.cursor()
            timestamp = datetime.datetime.now().isoformat()
            cursor.execute('''
                INSERT INTO search_logs (timestamp, query, hits_json)
                VALUES (?, ?, ?)
            ''', (timestamp, query.question, json.dumps(hit_ids)))
            conn.commit()
            conn.close()
        except Exception as log_error:
            print(f"Failed to log search metrics: {log_error}")
            
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/global_search", response_model=GlobalSearchResponse)
async def global_search(query: QueryRequest):
    if not retriever:
        raise HTTPException(status_code=503, detail="Retriever not initialized")
    
    try:
        # Await the async global context fetcher
        result = await retriever.get_global_context(query.question)
        
        # Log telemetry observability
        try:
            import datetime
            import json
            conn = sqlite3.connect("data/graph.db")
            cursor = conn.cursor()
            timestamp = datetime.datetime.now().isoformat()
            cursor.execute('''
                INSERT INTO search_logs (timestamp, query, hits_json)
                VALUES (?, ?, ?)
            ''', (timestamp, query.question, json.dumps(result.get("communities_used", []))))
            conn.commit()
            conn.close()
        except Exception as log_error:
            print(f"Failed to log global search metrics: {log_error}")
            
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "ok", "db_loaded": retriever is not None}

# ---------------------------------------------------------
# CLI Execution
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 5: Retrieval API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host IP")
    parser.add_argument("--port", type=int, default=8000, help="Port")
    args = parser.parse_args()
    
    uvicorn.run("retriever:app", host=args.host, port=args.port, reload=True)

if __name__ == "__main__":
    main()
