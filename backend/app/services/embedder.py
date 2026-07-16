from pinecone import Pinecone
import logging
from app.config import settings

logger = logging.getLogger(__name__)

class EmbedderService:
    def __init__(self):
        # Validate keys before starting service operations
        settings.validate_keys()
        self.pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        self.model = "llama-text-embed-v2"

    def get_embeddings(self, texts: list[str], is_query: bool = False) -> list[list[float]]:
        """
        Generate vector embeddings for a list of texts using Pinecone Inference.
        
        Args:
            texts: List of strings to embed.
            is_query: If True, uses 'query' input type parameter, otherwise 'passage'.
        """
        if not texts:
            return []
            
        input_type = "query" if is_query else "passage"
        try:
            response = self.pc.inference.embed(
                model=self.model,
                inputs=texts,
                parameters={"input_type": input_type, "truncate": "END"}
            )
            # Pinecone returns a list-like objects containing 'values'
            return [item['values'] for item in response]
        except Exception as e:
            logger.error(f"Failed to generate Pinecone embeddings: {e}")
            raise RuntimeError(f"Embedding generation failed: {e}")
