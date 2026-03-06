import json
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import argparse
import uuid
import time
import asyncio
from typing import List, Dict, Any, Tuple, Literal
from thefuzz import process, fuzz
from sentence_transformers import SentenceTransformer, util
import numpy as np
import torch
import instructor
from litellm import acompletion
from pydantic import BaseModel, Field

class ResolutionResult(BaseModel):
    event: Literal["ADD", "UPDATE", "DELETE", "NONE"] = Field(
        description="The resolution action to take."
    )

# ---------------------------------------------------------
# Entity Canonicalization (String Matching)
# ---------------------------------------------------------

async def canonicalize_entities(entities: List[Dict[str, Any]], threshold: int = 85) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """
    Merges entities with similar names. 
    Returns the canonicalized list of entities and a mapping from old IDs to new canonical IDs.
    """
    canonical_entities = []
    id_mapping = {} # old_id -> canonical_id
    
    for entity in entities:
        name = entity["name"]
        ent_type = entity["type"]
        ent_id = entity["id"]
        
        # Only merge entities of the same type
        candidates = [ce for ce in canonical_entities if ce["type"] == ent_type]
        
        if not candidates:
            canonical_entities.append(entity)
            id_mapping[ent_id] = ent_id
            continue
            
        candidate_names = [ce["name"] for ce in candidates]
        
        # Use simple fuzzy matching (thefuzz) to find similarity
        best_match_tuple = process.extractOne(name, candidate_names, scorer=fuzz.token_sort_ratio)
        
        if best_match_tuple and best_match_tuple[1] >= threshold:
            best_match_name = best_match_tuple[0]
            # Find the actual candidate object
            target_entity = next(ce for ce in candidates if ce["name"] == best_match_name)
            
            # Merge into target_entity
            if name not in target_entity.get("aliases", []) and name != target_entity["name"]:
                target_entity.setdefault("aliases", []).append(name)
            
            # Optionally merge incoming aliases
            for alias in entity.get("aliases", []):
                if alias not in target_entity["aliases"] and alias != target_entity["name"]:
                    target_entity["aliases"].append(alias)
                    
            id_mapping[ent_id] = target_entity["id"]
        else:
            canonical_entities.append(entity)
            id_mapping[ent_id] = ent_id
            
    return canonical_entities, id_mapping

# ---------------------------------------------------------
# Claim Deduplication (Semantic Similarity)
# ---------------------------------------------------------

class ClaimDeduplicator:
    def __init__(self, model_name="all-MiniLM-L6-v2", threshold=0.75, llm_model="models/gemini-1.5-flash"):
        print(f"Loading SentenceTransformer: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.threshold = threshold
        self.llm_model = llm_model
        self.client = instructor.from_litellm(acompletion, mode=instructor.Mode.MD_JSON)
        self.llm_semaphore = asyncio.Semaphore(10)
        
    async def _resolve_conflict_llm(self, existing_claim: Dict[str, Any], new_claim: Dict[str, Any]) -> str:
        prompt = f"""You are a smart memory manager resolving conflicts between an existing knowledge graph claim and a newly extracted claim.
        
Existing Claim: {existing_claim.get('subject_id', '')} - {existing_claim.get('predicate', '')} - {existing_claim.get('object_id', '')}
New Claim: {new_claim.get('subject_id', '')} - {new_claim.get('predicate', '')} - {new_claim.get('object_id', '')}

Evaluate if the new claim should:
- ADD: It contains distinct or new information.
- UPDATE: It presents updated information on the exact same topic overriding the existing claim.
- DELETE: It directly contradicts the existing claim.
- NONE: It contains the exact same information or is redundant.

Respond with the appropriate event."""
        async with self.llm_semaphore:
            try:
                result = await self.client.chat.completions.create(
                    model=self.llm_model,
                    response_model=ResolutionResult,
                    messages=[{"role": "user", "content": prompt}],
                    max_retries=3
                )
                return result.event
            except Exception as e:
                print(f"Warning: LLM resolution failed, defaulting to NONE. Error: {e}")
                return "NONE"

    def _merge_evidence(self, target_claim: Dict[str, Any], new_claim: Dict[str, Any]):
        if "evidences" not in target_claim:
            target_claim["evidences"] = [target_claim["evidence"]] if "evidence" in target_claim else []
            if "evidence" in target_claim:
                del target_claim["evidence"]
            
        evidences_list = target_claim.get("evidences", [])
        new_evidence = new_claim.get("evidence")
        if new_evidence and new_evidence not in evidences_list:
             target_claim.setdefault("evidences", []).append(new_evidence)
        
    async def deduplicate(self, claims: List[Dict[str, Any]], id_mapping: Dict[str, str]) -> List[Dict[str, Any]]:
        """
        Deduplicates claims based on semantic similarity of their full textual representations.
        Requires that subject and object entities have already been canonicalized via id_mapping.
        """
        canonical_claims = []
        
        for claim in claims:
            # First, remap the entity IDs
            claim["subject_id"] = id_mapping.get(claim["subject_id"], claim["subject_id"])
            claim["object_id"] = id_mapping.get(claim["object_id"], claim["object_id"])
            
            # Build a string representation of the claim for embedding
            # e.g., "person_john_smith developed feature_auth"
            claim_text = f"{claim['subject_id']} {claim['predicate']} {claim['object_id']}"
            
            # Check for matches
            is_duplicate = False
            if canonical_claims:
                candidate_texts = [f"{c['subject_id']} {c['predicate']} {c['object_id']}" for c in canonical_claims]
                
                # Compute embeddings and cosine similarities
                # (For production with millions of claims, use FAISS or similar vector DB)
                claim_emb = self.model.encode(claim_text, convert_to_tensor=True)
                candidate_embs = self.model.encode(candidate_texts, convert_to_tensor=True)
                
                cosine_scores = util.cos_sim(claim_emb, candidate_embs)[0]
                best_idx = int(torch.argmax(cosine_scores)) if hasattr(cosine_scores, 'argmax') else np.argmax(cosine_scores.cpu().numpy())
                best_score = float(cosine_scores[best_idx])
                
                if best_score >= self.threshold:
                    target_claim = canonical_claims[best_idx]
                    
                    event = await self._resolve_conflict_llm(target_claim, claim)
                    
                    if event == "NONE":
                        import datetime
                        self._merge_evidence(target_claim, claim)
                        target_claim["last_observed_at"] = datetime.datetime.now().isoformat()
                        target_claim["confidence_score"] = 1.0
                        is_duplicate = True
                    elif event == "UPDATE":
                        import datetime
                        target_claim["predicate"] = claim["predicate"]
                        target_claim["object_id"] = claim["object_id"]
                        self._merge_evidence(target_claim, claim)
                        target_claim["last_observed_at"] = datetime.datetime.now().isoformat()
                        target_claim["confidence_score"] = 1.0
                        is_duplicate = True
                    elif event == "DELETE":
                        import datetime
                        target_claim["invalid_at"] = target_claim.get("invalid_at") or datetime.datetime.now().isoformat()
                        target_claim["confidence_score"] = 0.0
                        is_duplicate = False
                    elif event == "ADD":
                        is_duplicate = False
                         
            if not is_duplicate:
                # Format standardization for persistence
                if "evidence" in claim:
                    claim["evidences"] = [claim["evidence"]]
                    del claim["evidence"]
                
                # Assign a unique ID to the canonical claim
                claim["claim_id"] = f"claim_{uuid.uuid4().hex[:8]}"
                canonical_claims.append(claim)
                
        return canonical_claims

# ---------------------------------------------------------
# Transaction Log for Reversibility
# ---------------------------------------------------------

class AuditLog:
    def __init__(self, log_path: str):
        self.log_path = log_path
        os.makedirs(os.path.dirname(os.path.abspath(self.log_path)), exist_ok=True)
        
    def log_merge(self, merge_type: str, source_item: Any, target_item: Any, score: float):
        entry = {
            "timestamp": time.time(),
            "type": merge_type,  # 'entity' or 'claim'
            "source": source_item,
            "target": target_item,
            "confidence_score": score
        }
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

# ---------------------------------------------------------
# Main Execution
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 3: Deduplication & Canonicalization")
    parser.add_argument("--input", type=str, default="data/extracted.jsonl", help="Input JSONL from Phase 2")
    parser.add_argument("--output", type=str, default="data/canonical.json", help="Output canonicalized JSON graph")
    parser.add_argument("--model", type=str, default="groq/llama-3.3-70b-versatile", help="LiteLLM model string")
    parser.add_argument("--audit", type=str, default="data/audit_log.jsonl", help="Path to merge transaction log")
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found.")
        return
        
    all_entities = []
    all_claims = []
    
    # Load all extracted data
    print(f"Loading data from {args.input}...")
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            all_entities.extend(data.get("entities", []))
            all_claims.extend(data.get("claims", []))
            
    print(f"Loaded {len(all_entities)} entities and {len(all_claims)} claims.")
    
    async def run_dedup():
        # 1. Canonicalize Entities
        print("Canonicalizing Entities...")
        canon_entities, id_mapping = await canonicalize_entities(all_entities, threshold=85)
        print(f"Reduced to {len(canon_entities)} canonical entities.")
        
        # 2. Canonicalize Claims
        print("Canonicalizing Claims...")
        try:
            import torch
        except ImportError:
             print("Warning: torch not found. This might impact sentence-transformers performance.")
             
        deduper = ClaimDeduplicator(threshold=0.85)
        canon_claims = await deduper.deduplicate(all_claims, id_mapping)
        print(f"Reduced to {len(canon_claims)} canonical claims.")
        
        # Save results
        output_data = {
            "entities": canon_entities,
            "claims": canon_claims
        }
        
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)
            
        print(f"Successfully saved canonical graph to {args.output}")

    asyncio.run(run_dedup())

if __name__ == "__main__":
    main()
