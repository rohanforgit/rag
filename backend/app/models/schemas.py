from pydantic import BaseModel, Field
from typing import Optional, List

class ChatRequest(BaseModel):
    message: str = Field(..., description="The user's question about regulations.")
    session_id: Optional[str] = Field(None, description="Optional conversation session identifier.")
    top_k: Optional[int] = Field(5, description="Number of retrieved sources to use.")

class SourceInfo(BaseModel):
    page: int = Field(..., description="The page number where the information was retrieved.")
    section: str = Field(..., description="The name of the regulation section.")
    document: str = Field(..., description="The document name.")

class ChatResponse(BaseModel):
    answer: str = Field(..., description="The AI's grounding answer.")
    sources: List[SourceInfo] = Field(..., description="List of source pages and sections retrieved.")
    session_id: str = Field(..., description="The ID of the conversation session.")
