import json
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import argparse
import asyncio
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator
import instructor
from litellm import acompletion

# ---------------------------------------------------------
# Ontology Definition
# ---------------------------------------------------------

class Entity(BaseModel):
    id: str = Field(description="Unique identifier for the entity (e.g., 'person_john_smith', 'feature_auth')")
    name: str = Field(description="Canonical name of the entity")
    type: str = Field(description="Type of entity, e.g., 'Person', 'Organization', 'Concept', 'Feature'")
    aliases: List[str] = Field(default_factory=list, description="Alternative names or abbreviations for this entity")

class Evidence(BaseModel):
    source_id: str = Field(description="ID of the source document")
    exact_excerpt: str = Field(description="Exact substring from the original text grounding this claim")
    character_start_offset: int = Field(description="Start index of exact_excerpt in the text")
    character_end_offset: int = Field(description="End index of exact_excerpt in the text")
    timestamp: str = Field(description="Timestamp of the source document")

class Claim(BaseModel):
    subject_id: str = Field(description="ID of the subject Entity reference")
    predicate: str = Field(description="Relationship or action connecting subject and object (e.g., 'developed', 'depends_on', 'is_delayed')")
    object_id: str = Field(description="ID of the object Entity reference")
    valid_at: str = Field(description="Start timeframe or condition when this claim is valid (e.g., 'Always', '2024-01-01')")
    invalid_at: str = Field(default="", description="Timeframe when this claim became invalid")
    expired_at: str = Field(default="", description="Timeframe when this claim expired")
    evidence: Evidence

class ExtractionResult(BaseModel):
    entities: List[Entity] = Field(default_factory=list)
    claims: List[Claim] = Field(default_factory=list)

# ---------------------------------------------------------
# Strict Grounding Validator
# ---------------------------------------------------------

def validate_grounding(text: str, result: ExtractionResult) -> ExtractionResult:
    """
    Verifies the LLM's output. The exact_excerpt must exist character-for-character 
    within the original source text. If it fails, attempts to correct offsets.
    If the excerpt is entirely missing from the text, it drops the claim.
    """
    valid_claims = []
    
    for claim in result.claims:
        excerpt = claim.evidence.exact_excerpt
        start = claim.evidence.character_start_offset
        end = claim.evidence.character_end_offset
        
        # Check if perfectly mapped
        if text[start:end] == excerpt:
            valid_claims.append(claim)
            continue
            
        # If offsets are wrong, let's search for the exact excerpt in the text
        actual_start = text.find(excerpt)
        if actual_start != -1:
            claim.evidence.character_start_offset = actual_start
            claim.evidence.character_end_offset = actual_start + len(excerpt)
            valid_claims.append(claim)
        else:
            # Drop the claim if the excerpt doesn't exactly exist in the text
            print(f"[Warning] Grounding validation failed for claim '{claim.predicate}'. "
                  f"Excerpt '{excerpt}' not found in source text. Dropping claim.")
            
    result.claims = valid_claims
    return result

# ---------------------------------------------------------
# Extraction Pipeline
# ---------------------------------------------------------

# Initialize litellm through instructor
client = instructor.from_litellm(acompletion, mode=instructor.Mode.MD_JSON)

async def extract_knowledge(text: str, source_id: str, timestamp: str, model: str) -> ExtractionResult:
    """
    Uses the configured LLM to generate the knowledge graph extraction from text.
    Enforces the Pydantic schema using instructor, then rigorously validates grounding.
    """
    
    # We construct a system prompt asking for taxonomic precision.
    system_prompt = (
        "You are a highly precise organizational knowledge extraction engine. "
        "Extract entities and relationship claims from the provided text using the following taxonomy: "
        "1. Architectural Decisions: Core system design, databases, patterns. "
        "2. Technical Debt: Known bugs, required refactors, outdated libraries. "
        "3. Component Relationships: Dependencies between services or modules. "
        "4. Planned Features: Future development goals or product roadmap items. "
        "CRITICAL: Every claim must be supported by an 'exact_excerpt' that is "
        "copied character-for-character from the provided text. Provide character offsets as well. "
        f"Use source_id '{source_id}' and timestamp '{timestamp}' for evidence."
    )
    
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Text to analyze:\n\n{text}"}
            ],
            response_model=ExtractionResult,
            max_retries=3,
        )
        
        # Run strict grounding validation on the parsed response
        validated_response = validate_grounding(text, response)
        return validated_response
        
    except Exception as e:
        print(f"[Error] Failed extraction for {source_id}: {str(e)}")
        return ExtractionResult(entities=[], claims=[])

# ---------------------------------------------------------
# CLI Execution
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 2: Structured Knowledge Extraction Pipeline")
    parser.add_argument("--input", type=str, default="data/corpus.jsonl", help="Input JSONL file")
    parser.add_argument("--output", type=str, default="data/extracted.jsonl", help="Output JSONL file")
    parser.add_argument("--model", type=str, default="gemini/gemini-2.5-flash", help="Model string for LiteLLM")
    parser.add_argument("--limit", type=int, default=10, help="Max items to process for testing")
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} not found. Run Phase 1 first.")
        return
        
    print(f"Starting extraction using model: {args.model}")
    print("Ensure you have set the appropriate API key environment variable (e.g., GEMINI_API_KEY).")
    
    async def process_item(item, semaphore):
        async with semaphore:
            source_id = item["source_id"]
            text = item["text"]
            timestamp = item["timestamp"]
            
            print(f"Processing {source_id} ({len(text)} chars)...")
            
            # Simple chunking/skip if text is too short or extremely long
            if len(text) < 50:
                print(f"  -> Skipping (too short)")
                return None
                
            from chunker import TextOverlapChunker
            chunker = TextOverlapChunker(max_chunk_size=4000, overlap_ratio=0.15)
            chunks = chunker.chunk_text(text)
            
            all_entities = []
            all_claims = []
            
            for i, chunk in enumerate(chunks):
                # We append a chunk index to the source ID to differentiate evidences
                chunk_source_id = f"{source_id}_chunk_{i}"
                result = await extract_knowledge(chunk["text"], chunk_source_id, timestamp, args.model)
                all_entities.extend(result.entities)
                all_claims.extend(result.claims)
            
            out_obj = {
                "source_id": source_id,
                "entities": [e.model_dump() for e in all_entities],
                "claims": [c.model_dump() for c in all_claims]
            }
            print(f"  -> Extracted {len(all_entities)} entities and {len(all_claims)} claims.")
            return out_obj

    async def run_pipeline():
        items = []
        with open(args.input, "r", encoding="utf-8") as infile:
            for line in infile:
                if len(items) >= args.limit:
                    break
                items.append(json.loads(line))
        
        semaphore = asyncio.Semaphore(5)
        tasks = [process_item(item, semaphore) for item in items]
        results = await asyncio.gather(*tasks)
        
        extracted_data = [r for r in results if r is not None]
        
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as outfile:
            for data in extracted_data:
                outfile.write(json.dumps(data) + "\n")
                
        print(f"Successfully processed {len(extracted_data)} items and saved to {args.output}")

    asyncio.run(run_pipeline())

if __name__ == "__main__":
    main()
