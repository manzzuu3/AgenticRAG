import os
import faiss
import json
import numpy as np
from typing import List, Dict
from google import genai
from google.genai.types import EmbedContentConfig
from dotenv import load_dotenv

load_dotenv()

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION")
CREDENTIALS_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

# Paths (adjust based on project structure)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

INDEX_PATH = os.path.join(DATA_DIR, "faiss.index")
METADATA_PATH = os.path.join(DATA_DIR, "metadata.json")

class RAGSearchTool:
    def __init__(self):
        self.index = None
        self.metadata = None
        self.client = None
        self._initialize()

    def _initialize(self):
        if not os.path.exists(INDEX_PATH):
            raise FileNotFoundError(f"FAISS index not found at {INDEX_PATH}")
        
        print("Loading FAISS index...")
        self.index = faiss.read_index(INDEX_PATH)
        
        print("Loading metadata...")
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
            
        print("Initializing Google GenAI Client...")
        # Authenticate
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = CREDENTIALS_PATH
        self.client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

    async def search(self, query: str, k: int = 5) -> List[Dict]:
        
        print("\n" + "="*60)
        print(f"[RAG SEARCH] Query: '{query}'")
        print(f"[RAG SEARCH] Technique: Cosine Similarity (FAISS IndexFlatIP)")
        print(f"[RAG SEARCH] Requested top-k: {k}")
        print("="*60)
        
        print(f"[RAG SEARCH] Generating query embedding (Async)...")
        
        # Internal retry logic for embeddings
        max_retries = 3
        base_delay = 1
        query_vector = None
        
        for attempt in range(max_retries):
            try:
                response = await self.client.aio.models.embed_content(
                    model="text-embedding-004",
                    contents=query,
                    config=EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
                )
                query_vector = np.array([response.embeddings[0].values], dtype=np.float32)
                break
            except Exception as e:
                # Check for rate limit (429) or other transient errors
                is_rate_limit = "429" in str(e) or "ResourceExhausted" in str(type(e).__name__)
                if is_rate_limit and attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    print(f"[RAG SEARCH] Embedding rate limited. Retrying in {delay}s...")
                    import asyncio
                    await asyncio.sleep(delay)
                else:
                    print(f"[RAG SEARCH] Embedding failed: {e}")
                    raise e
                    
        if query_vector is None:
             raise RuntimeError("Failed to generate embedding")

        print(f"[RAG SEARCH] Query vector shape: {query_vector.shape}")
        
        # Normalize for cosine similarity
        faiss.normalize_L2(query_vector)
        print(f"[RAG SEARCH] Vector normalized for cosine similarity")
        
        print(f"[RAG SEARCH] Searching FAISS index...")
        # FAISS CPU search is fast enough to run in thread pool if needed, 
        # but for small indices direct call is fine. 
        # For safety in async context, we could use run_in_executor but it's okay here.
        distances, indices = self.index.search(query_vector, k)
        print(f"[RAG SEARCH] Found {len(indices[0])} candidates")
        
        results = []
        print("\n[RAG SEARCH] Retrieved Documents:")
        print("-" * 60)
        for i, idx in enumerate(indices[0]):
            if idx != -1:
                meta = self.metadata[idx]
                score = float(distances[0][i])
                
                print(f"\n  Document {i+1}:")
                print(f"    Index ID: {idx}")
                print(f"    Similarity Score: {score:.4f}")
                print(f"    Page: {meta.get('page_number', 'N/A')}")
                print(f"    Type: {meta.get('type', 'N/A')}")
                print(f"    Element ID: {meta.get('element_id', 'N/A')}")
                
                # Format excerpt nicely
                excerpt = meta.get("content", meta.get("raw_text", ""))
                excerpt_preview = excerpt[:150] + "..." if len(excerpt) > 150 else excerpt
                print(f"    Preview: {excerpt_preview}")
                
                # Use contextual meaning if available
                if "contextual_meaning" in meta:
                    excerpt = f"{excerpt}\n\n[Context: {meta['contextual_meaning']}]"
                    print(f"    Has Contextual Meaning: Yes")
                
                results.append({
                    "score": score,
                    "element_id": meta.get("element_id"),
                    "page": meta.get("page_number"),
                    "type": meta.get("type"),
                    "excerpt": excerpt,
                    "source": "NG12 Guideline"
                })
        
        print("\n" + "="*60)        
        print(f"[RAG SEARCH] Returning {len(results)} results")
        print("="*60 + "\n")
        return results

# Singleton instance for reuse
rag_tool = RAGSearchTool()
