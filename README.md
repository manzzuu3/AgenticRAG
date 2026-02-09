# NG12 Clinical Decision Support Agent

A clinical decision support system using Vertex AI and RAG to help assess cancer risk based on NICE NG12 guidelines.

## 🚀 Quick Start

1.  **Configure Environment**:
    Create a `.env` file in the root:
    ```bash
    GOOGLE_CLOUD_PROJECT=your-project-id
    GOOGLE_CLOUD_LOCATION=us-central1
    GOOGLE_APPLICATION_CREDENTIALS=credentials.json
    MODEL_NAME=gemini-2.0-flash-001
    ```
    > [!NOTE]
    > `GOOGLE_APPLICATION_CREDENTIALS` should point to your JSON key file (e.g., `credentials.json`) placed in the project root. This file is the API JSON file for google cloud access credentials. You can download it from the Google Cloud Console.

2.  **Run Application**:
    Using `uv` (recommended):
    ```bash
    uv run main.py
    ```

3.  **Access**:
    - **API**: http://localhost:8000
    - **Chat UI**: http://localhost:8501

## ✨ Features

- **Patient Assessment**: Ask about specific patients (e.g., "Assess PT-101") to evaluate cancer risk.
- **NG12 Chat**: Ask general questions about NG12 clinical guidelines.
- **Smart Citations**: Every response includes specific page references and excerpts from the guidelines.
- **Conversation History**: Chat context is preserved across multiple turns.

## 📁 Project Structure

```
agneticrag/
├── data/              # Guideline PDFs and FAISS index
├── src/
│   ├── agent/         # PydanticAI agent and prompts
│   ├── api/           # FastAPI backend
│   ├── database/      # SQLite history manager
│   ├── tools/         # RAG and lookup tools
│   └── ui/            # Streamlit frontend
├── main.py            # Main entry point (starts all services)
├── credentials.json  # GCP Credentials (you provide this)
└── pyproject.toml     # Dependency management
```

## Docker

Run the entire stack using Docker Compose or direct Docker commands. Both ensure proper port mapping to host **8001** (API) and **8502** (UI) to avoid local conflicts.

### Option 1: Docker Compose (Recommended)

1.  **Prepare Credentials**: Ensure `credentials.json` is in the project root.
2.  **Launch**:
    ```bash
    docker-compose up --build
    ```
- **API (Host)**: http://localhost:8001
- **UI (Host)**: http://localhost:8502

### Option 2: Docker Build & Run (Single Container)

1.  **Build Image**:
    ```bash
    docker build -t ng12-agent .
    ```
2.  **Run Container**:
    ```bash
    docker run -p 8001:8000 -p 8502:8501 --env-file .env ng12-agent
    ```

## Available Patients (for testing)

| ID | Name | Symptom |
|----|------|---------|
| PT-101 | John Doe | Unexplained hemoptysis |
| PT-102 | Jane Smith | Persistent cough |
| PT-103 | Robert Brown | Persistent cough & SOB |
| PT-104 | Sarah Connor | Dysphagia |
| PT-105 | Michael Chang | Iron-deficiency anaemia |

## Evaluation

Run the automated test suite to verify the agent's performance:

```bash
uv run python -m src.evaluation.evaluate
```

- **Note**: This must be run from the root directory to ensure all module imports are resolved correctly.
- **Expected Results**: The suite evaluates 4 patient cases and provides a pass/fail summary.

## Troubleshooting

- **API Errors**: Ensure your Google Cloud credentials are valid and Vertex AI is enabled.
- **Rate Limits**: The pipeline includes automatic retries for API rate limits.
- **Protobuf Errors**: If you see Protobuf errors in Docker, ensure `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` is set in your environment.
