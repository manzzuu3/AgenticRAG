from typing import List

def recursive_chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    if not text:
        return []

    if len(text) <= chunk_size:
        return [text]

    separators = ["\n\n", "\n", " ", ""]
    separator = ""

    for separator in separators:
        if separator != "" and separator in text:
            separator = separator
            break
            
    if separator == "":
        return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size-overlap)]

    splits = text.split(separator)
    chunks = []
    current_chunk = []
    current_len = 0
    
    for split in splits:
        split_len = len(split)
        
        while current_len + split_len + (len(separator) if current_chunk else 0) > chunk_size:
            if not current_chunk:
                chunks.extend(recursive_chunk_text(split, chunk_size, overlap))
                split = "" 
                split_len = 0
                break
                
            chunk_text = separator.join(current_chunk)
            chunks.append(chunk_text)
            
            while current_len > overlap and current_chunk:
                removed = current_chunk.pop(0)
                current_len -= len(removed)
                if current_chunk:
                    current_len -= len(separator)
            
        if split_len > 0:
            current_chunk.append(split)
            current_len += split_len + (len(separator) if len(current_chunk) > 1 else 0)
            
    if current_chunk:
        chunks.append(separator.join(current_chunk))
        
    return chunks
