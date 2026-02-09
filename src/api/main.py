from fastapi import FastAPI
import uvicorn
from dotenv import load_dotenv
from src.api.routes import router

load_dotenv()

app = FastAPI(
    title="Clinical Decision Agent",
    version="1.0.0",
    description="Clinical Decision Support System"
)

@app.on_event("startup")
async def startup_event():
    import os
    in_docker = os.path.exists('/.dockerenv')
    url = "http://localhost:8001" if in_docker else "http://localhost:8000"
    print("\n" + "="*50)
    print(f"API is ready! Access Docs at: {url}/docs")
    print("="*50 + "\n")

app.include_router(router)

if __name__ == "__main__":
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
