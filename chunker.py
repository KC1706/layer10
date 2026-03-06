import tiktoken
from typing import List, Dict

class TextOverlapChunker:
    """
    Splits text into chunks of a maximum size while preserving a specified overlap.
    This ensures that relationship contexts spanning boundaries are not lost during extraction.
    """
    def __init__(self, max_chunk_size: int = 2000, overlap_ratio: float = 0.15, separator: str = " "):
        self.max_chunk_size = max_chunk_size
        self.overlap_ratio = overlap_ratio
        self.overlap_size = int(max_chunk_size * overlap_ratio)
        self.separator = separator

    def chunk_text(self, text: str) -> List[Dict[str, str]]:
        """
        Splits text into a list of chunks with overlapping boundaries.
        Returns a list of dictionaries containing chunk text and offsets.
        """
        # A simple character-based overlap chunker. Can be upgraded to tokens if needed.
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = min(start + self.max_chunk_size, text_length)
            
            # If we aren't at the end of the text, try to find a clean break (e.g., newline or space)
            if end < text_length:
                # Look backwards for a newline to break on
                last_newline = text.rfind("\n", start, end)
                if last_newline != -1 and last_newline > start + (self.max_chunk_size // 2):
                    end = last_newline + 1
                else:
                    # Look for a space
                    last_space = text.rfind(self.separator, start, end)
                    if last_space != -1 and last_space > start + (self.max_chunk_size // 2):
                        end = last_space + len(self.separator)
            
            chunk_text = text[start:end]
            chunks.append({
                "text": chunk_text,
                "start_char": start,
                "end_char": end
            })
            
            if end >= text_length:
                break
                
            # Calculate next start with overlap
            # The next start should be `end - overlap_size`, but adjusted to a clean break
            next_start_candidate = max(start + 1, end - self.overlap_size)
            
            # Find a clean break for the overlap start
            first_newline = text.find("\n", next_start_candidate, end)
            if first_newline != -1:
                next_start = first_newline + 1
            else:
                first_space = text.find(self.separator, next_start_candidate, end)
                if first_space != -1:
                    next_start = first_space + len(self.separator)
                else:
                    next_start = next_start_candidate
                    
            start = next_start
            
        return chunks
