from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
from app.routers import chat

# Configure standard logging to console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("app.main")

app = FastAPI(
    title="Sreenidhi University R26 Regulations AI Assistant",
    description="A production-grade RAG Assistant for Sreenidhi University regulations.",
    version="1.0"
)

# Enable CORS for the local React development server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production deployments
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the routers
app.include_router(chat.router, tags=["chat"])

@app.get("/health")
async def health_check():
    """Service status health endpoint."""
    return {
        "status": "healthy",
        "service": "Sreenidhi R26 AI Assistant API"
    }
