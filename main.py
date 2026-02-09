import os
import json
import sys
import time
import subprocess
import argparse



from src.preprocess.parsed_data import process_pdf
from src.preprocess.enhancing_data import (
    load_elements, build_document_context, initialize_vertex_ai,
    process_text_element, process_table_element, save_results
)
from src.embeddings.embeddings import (
    load_enriched_data, initialize_vertex_ai as init_embeddings,
    create_embeddings, save_embeddings
)
from src.embeddings.faiss import build_faiss_index
from tqdm import tqdm


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
PDF_FILE = os.path.join(DATA_DIR, "suspected-cancer-recognition-and-referral.pdf")
ELEMENTS_FILE = os.path.join(DATA_DIR, "suspected-cancer-recognition-and-referral_elements.json")
ENRICHED_FILE = os.path.join(DATA_DIR, "suspected-cancer-recognition-and-referral_enriched.json")
INDEX_PATH = os.path.join(DATA_DIR, "faiss.index")

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
sys.stdout.reconfigure(line_buffering=True)

def parse_pdf():
    print("=" * 50)
    print("Step 1: Parsing PDF")
    print("=" * 50)
    if not os.path.exists(PDF_FILE):
        print(f"Error: PDF not found at {PDF_FILE}")
        return False
    pages, elements = process_pdf(PDF_FILE)
    if not pages:
        return False
        
    print(f"Processed {len(pages)} pages, {len(elements)} elements")
    


    with open(ELEMENTS_FILE, "w", encoding="utf-8") as f:

        element_dicts = [e.model_dump() for e in elements]
        json.dump(element_dicts, f, indent=2)
        
    print(f"Saved elements to {ELEMENTS_FILE}")
    return True

def enhance_data():
    print("\n" + "=" * 50)
    print("Step 2: Enhancing Data")
    print("=" * 50)
    if os.path.exists(ENRICHED_FILE):
        print(f"Skipping enrichment: Output exists at {ENRICHED_FILE}")
        return True
    if not os.path.exists(ELEMENTS_FILE):
        return False
        

    
    model = initialize_vertex_ai()
    elements = load_elements(ELEMENTS_FILE)
    full_context = build_document_context(elements)
    enriched_data = []
    
    print(f"Processing {len(elements)} elements...")
    for element in tqdm(elements, desc="Enhancing Data"):
        e_type = element.get("element_type", "text") # default to text if unknown
        if e_type == "text":
            try:
                # Suppress inner prints to avoid cluttering progress bar
                import io
                import contextlib
                # Redirect stdout for inner calls, or modify enhancing_data.py to use tqdm.write
                # For now, just let it print, but maybe disable chunk progress?
                # Actually, let's keep it simple first.
                enriched_data.extend(process_text_element(model, element, full_context))
            except Exception as e:
                print(f"Error processing element {element.get('element_id')}: {e}")
        elif e_type == "table":
            try:
                enriched_data.append(process_table_element(model, element, full_context))
            except Exception as e:
                print(f"Error processing table {element.get('element_id')}: {e}")
    save_results(enriched_data, ENRICHED_FILE)
    return True

def generate_embeddings():
    print("\n" + "=" * 50)
    print("Step 3: Generating Embeddings")
    print("=" * 50)
    if not os.path.exists(ENRICHED_FILE):
        return False
    init_embeddings()
    data = load_enriched_data(ENRICHED_FILE)
    texts = [item.get("raw_text", "") for item in data]
    metadata = data 
    embeddings = create_embeddings(texts)
    save_embeddings(embeddings, metadata)
    return True

def build_index():
    print("\n" + "=" * 50)
    print("Step 4: Building FAISS Index")
    print("=" * 50)
    build_faiss_index()
    return True

def run_pipeline():
    """Execute the full data processing pipeline."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    if parse_pdf():
        if enhance_data():
            if generate_embeddings():
                build_index()
                return True
    return False



def run_command(command):
    try:
        # Force unbuffered output
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        
        if sys.platform == 'win32':
            return subprocess.Popen(command, shell=True, env=env)
        return subprocess.Popen(command.split(), env=env)
    except Exception as e:
        print(f"Error starting {command}: {e}")
        return None

def start_services():
    """Start the API and UI services."""
    print("\n" + "=" * 50)
    print(" STARTING NG12 AGENT SERVICES")
    print("=" * 50)
    
    # Detect if we are in Docker
    in_docker = os.path.exists('/.dockerenv')
    
    # Define ports (internal)
    api_port = 8000
    ui_port = 8501
    
    # If in docker, we might be mapped to different ports on host
    # We use 8001 and 8502 in our compose, so we can hint at that
    api_host_url = "http://localhost:8001" if in_docker else f"http://localhost:{api_port}"
    ui_host_url = "http://localhost:8502" if in_docker else f"http://localhost:{ui_port}"

    print(f"[1/2] Starting API (FastAPI) on port {api_port}...")
    api_proc = run_command(f"uvicorn src.api.main:app --host 0.0.0.0 --port {api_port} --reload")
    
    print(f"[2/2] Starting UI (Streamlit) on port {ui_port}...")
    time.sleep(2)
    # Use --server.address 0.0.0.0 but we will print localhost for the user
    ui_proc = run_command(f"streamlit run src/ui/streamlit_app.py --server.port {ui_port} --server.address 0.0.0.0 --server.headless true")
    
    print("\n" + "*" * 50)
    print("APPLICATION STARTED SUCCESSFULLY!")
    print("*" * 50)
    
    if in_docker:
        print("\nRUNNING IN DOCKER")
        print(f"Access Chat UI: {ui_host_url}")
        print(f"Access API Docs: {api_host_url}/docs")
        print("\nNote: These are the standard ports defined in your docker-compose.yml")
    else:
        print("\nRUNNING LOCALLY")
        print(f"Access Chat UI: {ui_host_url}")
        print(f"Access API Docs: {api_host_url}/docs")
    
    print("\n(Press Ctrl+C to stop services)")
    print("*" * 50 + "\n")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping services...")
        if api_proc: api_proc.terminate()
        if ui_proc: ui_proc.terminate()
        print("Goodbye!")


def main():
    parser = argparse.ArgumentParser(description="NG12 Agent Application")
    parser.add_argument("--pipeline-only", action="store_true", help="Run only the data pipeline")
    parser.add_argument("--app-only", action="store_true", help="Run only the application services")
    args = parser.parse_args()

    if args.pipeline_only:
        run_pipeline()
    elif args.app_only:
        start_services()
    else:

        if not os.path.exists(INDEX_PATH):
            print("Search index not found. Running data pipeline first...")
            if run_pipeline():
                start_services()
        else:
            print("Data pipeline appears complete.")
            start_services()

if __name__ == "__main__":
    main()
