# NG12 Clinical Decision Support Agent

A FastAPI-powered clinical decision support system that combines structured patient data with NICE NG12 cancer guidelines using Vertex AI and RAG (Retrieval-Augmented Generation).

## Features

- **Part 1: Patient Risk Assessor** - Evaluate cancer risk for specific patients
- **Part 2: Conversational Chat** - Ask general questions about NG12 guidelines
- **RAG Pipeline** - FAISS vector search with Vertex AI embeddings
- **Conversation Memory** - SQLite-based session management
- **Citations** - All responses include specific guideline references
- **Streamlit UI** - Interactive web interface

## Architecture

```
User Query
    ↓
FastAPI (POST /chat)
    ↓
PydanticAI Agent (Gemini 1.5)
    ├─→ get_patient_data (tool) → patients.json
    └─→ search_guidelines (tool) → FAISS + Vertex AI
            ↓
        Returns: {assessment, reasoning, citations}
    ↓
SQLite (conversation history)
    ↓
Streamlit UI
```

## Prerequisites

- Python 3.11+
- Google Cloud Project with Vertex AI enabled
- Service account credentials for Vertex AI

## Quick Start

### 1. Clone and Setup

```bash
git clone <your-repo>
cd agneticrag
```

### 2. Configure Environment

Create `.env` file:

```bash
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=us-central1
GOOGLE_APPLICATION_CREDENTIALS=path/to/credentials.json
MODEL_NAME=gemini-1.5-flash-001
```

### 3. Install Dependencies

Using `uv` (recommended):
```bash
# Install uv if not already installed
pip install uv

# Sync dependencies
uv sync
```

### 4. Run Locally

```bash
uv run main.py
```

The application will start:
- **API**: http://localhost:8000
- **UI**: http://localhost:8501
- **API Docs**: http://localhost:8000/docs

## Usage

### Part 1: Patient Risk Assessment

**API Request:**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "test123",
    "message": "Please assess patient PT-101"
  }'
```

**Response:**
```json
{
  "session_id": "test123",
  "answer": "**Assessment:** URGENT REFERRAL\n\n**Reasoning:** Patient PT-101 (John Doe, 55, Male, Current Smoker) presents with unexplained hemoptysis...",
  "assessment": "URGENT REFERRAL",
  "reasoning": "...",
  "citations": [
    {
      "source": "NG12 Guideline",
      "page": 15,
      "excerpt": "Refer people using a suspected cancer pathway referral..."
    }
  ]
}
```

### Part 2: Conversational Chat

**API Request:**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "chat456",
    "message": "What symptoms trigger an urgent referral for lung cancer?"
  }'
```

**Multi-Turn Example:**
```bash
# Turn 1
curl -X POST .../chat -d '{"session_id": "s1", "message": "What about persistent hoarseness?"}'

# Turn 2 (with context from Turn 1)
curl -X POST .../chat -d '{"session_id": "s1", "message": "What age threshold?"}'
```

### Streamlit UI

1. Open http://localhost:8501
2. Enter a patient ID (e.g., PT-101) or ask a general question
3. View assessment and citations
4. Continue multi-turn conversation

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/chat` | POST | Main chat endpoint for queries |
| `/chat/{session_id}/history` | GET | Retrieve conversation history |
| `/chat/{session_id}` | DELETE | Clear session history |
| `/health` | GET | Health check |

## Project Structure

```
agneticrag/
├── data/
│   ├── patients.json              # Structured patient data
│   ├── faiss.index                # Vector database
│   ├── metadata.json              # Document metadata
│   └── *.pdf                      # NG12 guideline PDF
├── src/
│   ├── agent/
│   │   ├── agent.py               # PydanticAI agent logic
│   │   └── PROMPTS.md             # System prompts
│   ├── api/
│   │   ├── main.py                # FastAPI application
│   │   ├── routes.py              # API endpoints
│   │   └── schemas.py             # Pydantic models
│   ├── database/
│   │   └── db_manager.py          # SQLite conversation storage
│   ├── tools/
│   │   ├── patient_data.py        # Patient lookup tool
│   │   └── rag_search.py          # FAISS search tool
│   └── ui/
│       ├── streamlit_app.py       # Streamlit interface
│       └── style.css              # UI styling
├── main.py                        # Unified entry point
├── Dockerfile                     # Docker configuration
├── docker-compose.yml             # Multi-container setup
├── pyproject.toml                 # Dependencies (uv)
└── README.md                      # This file
```

### Docker Deployment

### 1. Build and Run with Compose

```bash
docker-compose up --build
```

Access:
- API: http://localhost:8000
- UI: http://localhost:8501

### 2. Docker-Only (No Compose)

```bash
# Build image
docker build -t ng12-agent .

# Run container (pass env vars)
docker run -p 8000:8000 -p 8501:8501 --env-file .env ng12-agent
```

## Data Pipeline

The RAG pipeline preprocesses the NG12 PDF:

1. **PDF Parsing** - Extracts text and structure
2. **Chunking** - Splits into semantic chunks
3. **Embedding** - Vertex AI `text-embedding-004`
4. **Indexing** - FAISS IndexFlatIP (cosine similarity)

Pre-processed data is included in `data/` directory.

## Available Patients

| ID | Name | Age | Symptoms |
|----|------|-----|----------|
| PT-101 | John Doe | 55 | Unexplained hemoptysis |
| PT-102 | Jane Smith | 25 | Persistent cough |
| PT-103 | Robert Brown | 45 | Persistent cough, SOB |
| PT-104 | Sarah Connor | 35 | Dysphagia |
| PT-105 | Michael Chang | 65 | Iron-deficiency anaemia |
| PT-106 | Emily Blunt | 18 | Fatigue |
| PT-107 | David Bowie | 48 | Persistent hoarseness |
| PT-108 | Alice Wonderland | 32 | Unexplained breast lump |
| PT-109 | Tom Cruise | 45 | Dyspepsia |
| PT-110 | Bruce Wayne | 60 | Visible haematuria |

## Key Features

### 1. Grounding & Refusal Logic
- Low-confidence threshold (< 0.4 similarity) triggers refusal
- Agent says "I don't know" when evidence is insufficient

### 2. Multi-Turn Context
- Last 3 conversation exchanges included in context
- Enables follow-up questions like "What about age 40?"

### 3. Citation Quality
- Every response includes:
  - Source: "NG12 Guideline"
  - Page number
  - Verbatim excerpt
  - Confidence score

### 4. Structured Outputs
- Pydantic models ensure consistent JSON responses
- Type-safe, validated data

## Testing

### Test Scripts

```bash
# Test general question
uv run python test_endpoint.py

# Test with valid patient
uv run python test_with_patient.py
```

### Example Test Cases

1. **Urgent Referral**: PT-101 (hemoptysis, smoker)
2. **Routine Referral**: PT-102 (young, short duration)
3. **Multi-symptom**: PT-103 (cough + SOB)
4. **General Query**: "What triggers lung cancer referral?"
5. **Low Confidence**: "What about aspirin for prevention?"

## Configuration

### Agent Settings (agent.py)
```python
retries=3                    # Tool call retry limit
usage_limits=40              # Max LLM requests per conversation
confidence_threshold=0.4     # RAG similarity threshold
top_k=4                      # Guideline chunks retrieved
```

### RAG Settings (rag_search.py)
```python
embedding_model="text-embedding-004"    # Vertex AI model
index_type="IndexFlatIP"                # FAISS cosine similarity
vector_dimension=768                    # Embedding size
```

## Troubleshooting

### Issue: Agent hits request limit
**Cause:** Complex queries triggering excessive tool calls
**Solution:** Increase `usage_limits` or simplify query

### Issue: Low-quality citations
**Cause:** Confidence threshold too low
**Solution:** Increase threshold in `search_guidelines` (currently 0.4)

### Issue: Missing context in chat
**Cause:** Session ID not consistent across requests
**Solution:** Use same `session_id` for multi-turn conversations

### Issue: Docker build fails
**Cause:** Missing credentials file
**Solution:** Ensure `.env` points to valid credentials

## Development

### Run in Dev Mode
```bash
# Start API only
uvicorn src.api.main:app --reload

# Start Streamlit only
streamlit run src/ui/streamlit_app.py
```

### Add New Tools
1. Create tool function in `src/tools/`
2. Decorate with `@clinical_agent.tool`
3. Update system prompt in `PROMPTS.md`

### Modify System Prompt
Edit `src/agent/PROMPTS.md` and restart application

## Documentation

- **[PROMPTS.md](src/agent/PROMPTS.md)** - System prompt design
- **[API Docs](http://localhost:8000/docs)** - Interactive API documentation

## Technology Stack

- **LLM**: Google Gemini 1.5 (via Vertex AI)
- **Framework**: PydanticAI
- **API**: FastAPI
- **UI**: Streamlit
- **Vector DB**: FAISS
- **Embeddings**: Vertex AI text-embedding-004
- **Database**: SQLite
- **Package Manager**: uv
