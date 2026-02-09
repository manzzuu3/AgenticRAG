import os
import json
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai.types import EmbedContentConfig

load_dotenv()

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION")
CREDENTIALS_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_DIR = os.path.join(PROJECT_DIR, "data")

ENRICHED_FILE = os.path.join(DATA_DIR, "suspected-cancer-recognition-and-referral_enriched.json")
EMBEDDINGS_FILE = os.path.join(DATA_DIR, "embeddings.npy")
METADATA_FILE = os.path.join(DATA_DIR, "metadata.json")

# Global client for reuse
_client = None


def load_enriched_data(filepath: str) -> list:
    print(f"Loading enriched data from: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} elements")
    return data


def initialize_vertex_ai():
    global _client
    print("\nInitializing Google GenAI Client...")
    print(f"Project: {PROJECT_ID}, Location: {LOCATION}")
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = CREDENTIALS_PATH
    _client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)


def create_embeddings(texts: list) -> np.ndarray:
    global _client
    print(f"\nCreating embeddings for {len(texts)} texts using Google GenAI...")
    

    batch_size = 5
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        print(f"Processing batch {i // batch_size + 1}/{(len(texts) - 1) // batch_size + 1}...")
        
        for text in batch:
            response = _client.models.embed_content(
                model="text-embedding-004",
                contents=text,
                config=EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
            )
            all_embeddings.append(response.embeddings[0].values)
    
    embeddings = np.array(all_embeddings, dtype=np.float32)
    print(f"Created embeddings with shape: {embeddings.shape}")
    return embeddings


def save_embeddings(embeddings: np.ndarray, metadata: list):
    print(f"\nSaving embeddings to: {EMBEDDINGS_FILE}")
    np.save(EMBEDDINGS_FILE, embeddings)
    
    print(f"Saving metadata to: {METADATA_FILE}")
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print("Saved successfully!")



