import os
from pydantic_settings import BaseSettings
from pydantic import Field

# Determine the directory where this file resides to locate the .env relative to it
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_FILE_PATH = os.path.join(BASE_DIR, ".env")

class Settings(BaseSettings):
    PINECONE_API_KEY: str = Field("", env="PINECONE_API_KEY")
    PINECONE_INDEX_NAME: str = Field("sreenidhi-r26-regs", env="PINECONE_INDEX_NAME")
    PINECONE_CLOUD: str = Field("aws", env="PINECONE_CLOUD")
    PINECONE_REGION: str = Field("us-east-1", env="PINECONE_REGION")
    GROQ_API_KEY: str = Field("", env="GROQ_API_KEY")
    GROQ_MODEL: str = Field("llama-3.3-70b-versatile", env="GROQ_MODEL")
    PORT: int = Field(8000, env="PORT")
    
    class Config:
        env_file = ENV_FILE_PATH
        env_file_encoding = "utf-8"
        extra = "ignore"

    def validate_keys(self):
        """Validate that essential keys are present and not placeholders."""
        missing = []
        if not self.PINECONE_API_KEY or "your-pinecone" in self.PINECONE_API_KEY.lower():
            missing.append("PINECONE_API_KEY")
        if not self.GROQ_API_KEY or "your-groq" in self.GROQ_API_KEY.lower():
            missing.append("GROQ_API_KEY")
        
        if missing:
            raise ValueError(
                f"Missing or invalid configuration for: {', '.join(missing)}. "
                f"Please update the backend/.env file with valid credentials."
            )

settings = Settings()
