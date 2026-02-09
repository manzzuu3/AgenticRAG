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

app.include_router(router)

if __name__ == "__main__":
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
