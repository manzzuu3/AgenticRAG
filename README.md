# NG12 Clinical Decision Support Agent

A clinical decision support system using Vertex AI and RAG to help assess cancer risk based on NICE NG12 guidelines.

## 🚀 Quick Start

1.  **Configure Environment**:
    Create a `.env` file in the root:
    ```bash
    GOOGLE_CLOUD_PROJECT=your-project-id
    GOOGLE_CLOUD_LOCATION=us-central1
    GOOGLE_APPLICATION_CREDENTIALS=path/to/credentials.json
    MODEL_NAME=gemini-2.0-flash-001
    ```

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
└── pyproject.toml     # Dependency management
```

## 📋 Available Patients (for testing)

| ID | Name | Symptom |
|----|------|---------|
| PT-101 | John Doe | Unexplained hemoptysis |
| PT-102 | Jane Smith | Persistent cough |
| PT-103 | Robert Brown | Persistent cough & SOB |
| PT-104 | Sarah Connor | Dysphagia |
| PT-105 | Michael Chang | Iron-deficiency anaemia |

## 🛠 Troubleshooting

- **API Errors**: Ensure your Google Cloud credentials are valid and Vertex AI is enabled.
- **Rate Limits**: The pipeline includes automatic retries for API rate limits.

## 🧪 Evaluation

Run the automated test suite to verify the agent's performance against patient cases:

```bash
uv run python -m src.evaluation.evaluate
```

- **Note**: This must be run from the root directory to ensure all module imports are resolved correctly.
- **Expected Results**: The suite evaluates 4 patient cases and provides a pass/fail summary.

- **Docker**: For containerized deployment, run `docker-compose up --build`.
