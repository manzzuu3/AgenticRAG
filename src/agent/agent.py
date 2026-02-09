import os
import json
import asyncio
from typing import List, Optional
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext, UsageLimits
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.exceptions import ModelHTTPError
from src.tools.patient_data import patient_tool
from src.tools.rag_search import rag_tool
from dotenv import load_dotenv


load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME")
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION")


class ReferralCitation(BaseModel):
    source: str = Field(..., description="Source document")
    page: Optional[int] = Field(None, description="Page number")
    excerpt: str = Field(..., description="Relevant excerpt")


class ClinicalAssessment(BaseModel):
    summary: str = Field(..., description="Brief summary")
    assessment: str = Field(..., description="URGENT REFERRAL, URGENT INVESTIGATION, ROUTINE, or SAFETY NETTING")
    reasoning: str = Field(..., description="Why this decision was made")
    citations: List[ReferralCitation] = Field(default_factory=list, description="Supporting evidence")


# Load system prompt
PROMPT_PATH = os.path.join(os.path.dirname(__file__), "PROMPTS.md")
with open(PROMPT_PATH, "r", encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read()

# Initialize agent
model = GeminiModel(MODEL_NAME, provider='google-vertex')
clinical_agent = Agent(
    model,
    output_type=ClinicalAssessment,
    system_prompt=SYSTEM_PROMPT,
    deps_type=str,
    retries=3
)


@clinical_agent.tool
def get_patient_data(ctx: RunContext[str], patient_id: str) -> str:
    """Get patient data by ID."""
    print(f"[TOOL] get_patient_data({patient_id})")
    data = patient_tool.get_patient_data(patient_id)
    if data:
        return json.dumps(data, indent=2)
    return f"Patient {patient_id} not found."


@clinical_agent.tool
async def search_guidelines(ctx: RunContext[str], query: str) -> str:
    """Search NG12 guidelines."""
    print(f"[TOOL] search_guidelines('{query}')")
    results = await rag_tool.search(query, k=4)
    
    if not results:
        return "No relevant guidelines found."
    
    if results[0].get("score", 0) < 0.4:
        return "Insufficient evidence found in guidelines for this query."
    
    formatted = []
    for r in results:
        excerpt = r.get("excerpt", "").replace("\n", " ").strip()
        page = r.get('page', 'N/A')
        formatted.append(f"[Page {page}] {excerpt}")
    
    return "\n\n---\n\n".join(formatted)


async def run_chat(session_id: str, message: str) -> ClinicalAssessment:
    """Run a chat session with the clinical agent."""
    from src.database.db_manager import DatabaseManager
    
    print(f"\n[AGENT] Session: {session_id}")
    print(f"[AGENT] Message: {message}")
    
    # Get conversation history
    db = DatabaseManager()
    history = await db.get_history(session_id)
    
    # Save user message to database
    await db.add_message(session_id, "user", message)
    
    # Build context from last 3 exchanges
    if history:
        recent = history[-6:]
        context = "\n".join([f"{'User' if m['role']=='user' else 'Assistant'}: {m['content']}" for m in recent])
        full_message = f"Previous conversation (for context):\n{context}\n\nCurrent user question: {message}"
    else:
        full_message = message
    
    # Retry logic for rate limits
    max_retries = 3
    base_delay = 2
    
    for attempt in range(max_retries):
        try:
            # Run agent
            result = await clinical_agent.run(
                full_message,
                deps=session_id,
                usage_limits=UsageLimits(request_limit=25)
            )
            
            # Save assistant message to database
            await db.add_message(session_id, "assistant", result.output.summary)
            
            break
        except ModelHTTPError as e:
            if e.status_code == 429:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)  # Exponential backoff: 2s, 4s, 8s...
                    print(f"\n[AGENT] Rate limited (429). Retrying in {delay}s...")
                    await asyncio.sleep(delay)
                    continue
                else:
                    print(f"\n[AGENT] Max retries exceeded for rate limit.")
                    raise
            else:
                raise
    
    print(f"\n[AGENT] Result:")
    print(json.dumps(result.output.model_dump(), indent=2))
    return result.output
