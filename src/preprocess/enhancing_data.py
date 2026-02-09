"""Functions for enriching parsed PDF data with contextual meaning using Google GenAI."""

import json
import os
import time
from dotenv import load_dotenv
from google import genai
from src.preprocess.chunking import recursive_chunk_text


load_dotenv()

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION")
MODEL_NAME = os.getenv("MODEL_NAME")
CREDENTIALS_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP"))

# Global client for reuse
_client = None


def load_elements(filepath: str) -> list:
    print(f"Loading elements from: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Found {len(data)} elements")
    return data


def build_document_context(elements: list) -> str:
    text_parts = [item["raw_text"] for item in elements if item.get("element_type") == "text"]
    context = "\n".join(text_parts)
    print(f"Built document context: {len(context):,} characters")
    return context


def _generate_with_retry(model_name: str, prompt: str, retries: int = 6) -> str:
    global _client
    base_delay = 2
    
    for attempt in range(retries):
        try:
            response = _client.models.generate_content(model=model_name, contents=prompt)
            return response.text.strip()
        except Exception as e:
            error_str = str(e)
            is_rate_limit = "429" in error_str or "RESOURCE_EXHAUSTED" in error_str
            
            if attempt < retries - 1:
                if is_rate_limit:
                    sleep_time = base_delay * (2 ** attempt)
                    print(f" [Rate Limit: Retrying in {sleep_time}s]", end="")
                    time.sleep(sleep_time)
                else:
                    print(f" [Error: {e}. Retrying...]", end="")
                    time.sleep(2)
            else:
                print(f" [Failed: {e}]")
                return f"Error: {error_str[:100]}"
    return "Error: Max retries exhausted"


def get_contextual_meaning(model_name: str, chunk: str, full_context: str) -> str:
    prompt = f"""You are an expert document analyst.

FULL DOCUMENT CONTEXT:
{full_context}

SPECIFIC CHUNK:
{chunk}

Task: Explain what this chunk means in the context of the entire document.
Be concise (1-2 sentences). Focus on its significance and purpose."""

    return _generate_with_retry(model_name, prompt)


def get_table_summary(model_name: str, table_markdown: str, full_context: str) -> str:
    prompt = f"""You are an expert document analyst.

FULL DOCUMENT CONTEXT:
{full_context}

TABLE CONTENT:
{table_markdown}

Task: Provide:
1. A short title for this table
2. A brief summary of what the table shows

Format: Title: [title] | Summary: [summary]"""

    return _generate_with_retry(model_name, prompt)


def process_text_element(model_name: str, element: dict, full_context: str) -> list:
    results = []
    raw_text = element.get("raw_text", "")
    element_id = element.get("element_id")
    page = element.get("page_number")
    
    chunks = recursive_chunk_text(raw_text, CHUNK_SIZE, CHUNK_OVERLAP)
    print(f"Chunked into {len(chunks)} pieces")
    
    for idx, chunk in enumerate(chunks):
        print(f" Chunk {idx + 1}/{len(chunks)}: Enriching...", end=" ")
        meaning = get_contextual_meaning(model_name, chunk, full_context)
        print("Success" if not meaning.startswith("Error") else "Error")
        
        results.append({
            "element_id": f"{element_id}",
            "page_number": page,
            "type": "text",
            "raw_text": f"{chunk}\n\n{meaning}",
        })
    
    return results


def process_table_element(model_name: str, element: dict, full_context: str) -> dict:
    element_id = element.get("element_id")
    page = element.get("page_number")
    table_md = element.get("markdown", "")
    
    print(f"Getting table summary...", end=" ")
    summary = get_table_summary(model_name, table_md, full_context)
    print("Success" if not summary.startswith("Error") else "Error")
    
    return {
        "element_id": element_id,
        "page_number": page,
        "type": "table",
        "raw_text": f"{table_md}\n\n{summary}",
        "table_summary": summary
    }


def save_results(data: list, filepath: str):
    print(f"\nSaving results to: {filepath}")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(data)} enriched elements")


def initialize_vertex_ai():
    global _client
    print("\nInitializing Google GenAI Client...")
    print(f"Project: {PROJECT_ID}, Location: {LOCATION}, Model: {MODEL_NAME}")
    
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = CREDENTIALS_PATH
    _client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
    
    print("Testing connection...", end=" ")
    _client.models.generate_content(model=MODEL_NAME, contents="Hello")
    print("Connected!")
    
    return MODEL_NAME
