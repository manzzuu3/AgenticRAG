from fastapi import APIRouter, HTTPException
from pydantic_ai.exceptions import UsageLimitExceeded
from src.api.schemas import ChatRequest, ChatResponse, Citation, HistoryResponse, Message
from src.database.db_manager import DatabaseManager
from src.agent.agent import run_chat, ClinicalAssessment

router = APIRouter()
db = DatabaseManager()

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Process a user message using the PydanticAI agent."""
    print("\n" + "*"*70)
    print(f"[API] Received POST /chat request")
    print(f"[API] Session ID: {request.session_id}")
    print(f"[API] Message: {request.message}")
    print("*"*70)
    
    try:
        print(f"[API] Saving user message to database...")
        await db.add_message(request.session_id, "user", request.message)
        
        print(f"[API] Calling agent.run_chat()...")
        result: ClinicalAssessment = await run_chat(request.session_id, request.message)
        print(f"[API] Agent returned result successfully")
        
        answer = f"**Assessment:** {result.assessment}\n\n**Reasoning:** {result.reasoning}"
        
        citations = [
            Citation(source=c.source, page=c.page, excerpt=c.excerpt)
            for c in result.citations
        ]
        print(f"[API] Formatted {len(citations)} citations")
        
        print(f"[API] Saving assistant response to database...")
        await db.add_message(request.session_id, "assistant", answer)
        
        print(f"[API] Preparing response...")
        response = ChatResponse(
            session_id=request.session_id,
            answer=answer,
            assessment=result.assessment,
            reasoning=result.reasoning,
            citations=citations
        )
        
        print("*"*70)
        print(f"[API] Returning successful response")
        print("*"*70 + "\n")
        return response
        
    except Exception as e:
        
        print("\n" + "!"*70)
        print(f"[API ERROR] Exception occurred: {type(e).__name__}")
        print(f"[API ERROR] Message: {str(e)}")
        
        if isinstance(e, UsageLimitExceeded):
            print("[API ERROR] DIAGNOSIS: Agent made too many LLM requests")
            print("[API ERROR] This indicates the model is struggling with:")
            print("[API ERROR]   - Structured output format (ClinicalAssessment)")
            print("[API ERROR]   - Or making excessive tool calls")
            print("[API ERROR] Consider simplifying the query or output format")
            error_msg = "The agent made too many requests while processing your query. Please try a simpler question or contact support."
        else:
            error_msg = str(e)
            
        print("!"*70)
        import traceback
        traceback.print_exc()
        print("!"*70 + "\n")
        raise HTTPException(status_code=500, detail=error_msg)

@router.get("/chat/{session_id}/history", response_model=HistoryResponse)
async def get_history(session_id: str):
    """Retrieve conversation history for a session."""
    history = await db.get_history(session_id)
    return HistoryResponse(
        session_id=session_id,
        history=[
            Message(role=m["role"], content=m["content"], timestamp=str(m["timestamp"]))
            for m in history
        ]
    )

@router.delete("/chat/{session_id}")
async def clear_history(session_id: str):
    """Clear conversation history for a session."""
    await db.clear_history(session_id)
    return {"message": "History cleared"}

@router.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}
