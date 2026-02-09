import os
import numpy as np
import json
import faiss
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"
METADATA_PATH = DATA_DIR / "metadata.json"
INDEX_PATH = DATA_DIR / "faiss.index"


def load_embeddings():
    print(f"Loading embeddings from: {EMBEDDINGS_PATH}")
    embeddings = np.load(EMBEDDINGS_PATH).astype(np.float32)
    print(f"Embeddings shape: {embeddings.shape}")
    return embeddings


def load_metadata():
    print(f"Loading metadata from: {METADATA_PATH}")
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    print(f"Loaded {len(metadata)} metadata entries")
    return metadata


def build_faiss_index():
    print("Loading embeddings and metadata...")
    embeddings = load_embeddings()
    metadata = load_metadata()

    dim = embeddings.shape[1]
    
    # HNSW parameters
    M = 32  
    ef_construction = 200  

    print(f"Creating HNSW index (cosine similarity, M={M}, ef_construction={ef_construction})...")
    index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = ef_construction
    index.hnsw.efSearch = 64  # Controls search quality vs speed

    # Normalize vectors for cosine similarity
    faiss.normalize_L2(embeddings)

    index.add(embeddings)
    print(f"Added {index.ntotal} vectors to HNSW index")

    faiss.write_index(index, str(INDEX_PATH))
    print(f"Saved FAISS index to: {INDEX_PATH}")

    return index, metadata


def search(query_embedding: np.ndarray, k: int = 5):
    print(f"Loading FAISS index from: {INDEX_PATH}")
    index = faiss.read_index(str(INDEX_PATH))
    metadata = load_metadata()

    query_embedding = query_embedding.astype(np.float32)
    if len(query_embedding.shape) == 1:
        query_embedding = query_embedding.reshape(1, -1)
    
    faiss.normalize_L2(query_embedding)

    D, I = index.search(query_embedding, k=k)

    print(f"\nTop {k} results:")
    results = []
    for rank, idx in enumerate(I[0]):
        if idx != -1:
            row = metadata[idx]
            score = float(D[0][rank])
            
            print(f"{rank + 1}. Score: {score:.4f}")
            print(f"   Element ID: {row['element_id']}")
            print(f"   Page: {row['page_number']}")
            print(f"   Type: {row['type']}")
            print("")
            
            results.append({
                "score": score,
                "metadata": row
            })
    
    return results
