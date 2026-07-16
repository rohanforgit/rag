from pinecone import Pinecone
import logging
from app.config import settings
from app.services.embedder import EmbedderService

logger = logging.getLogger(__name__)

class RetrieverService:
    def __init__(self, embedder: EmbedderService):
        settings.validate_keys()
        self.embedder = embedder
        self.pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        self.index = self.pc.Index(settings.PINECONE_INDEX_NAME)

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Retrieves matching sections from Pinecone using vector cosine similarity.
        """
        try:
            # 1. Embed query
            query_embeddings = self.embedder.get_embeddings([query], is_query=True)
            if not query_embeddings:
                raise ValueError("Could not embed query")
            query_vector = query_embeddings[0]

            # 2. Search index
            response = self.index.query(
                vector=query_vector,
                top_k=top_k,
                include_metadata=True
            )

            # 3. Format results
            results = []
            for match in response.get("matches", []):
                metadata = match.get("metadata", {})
                results.append({
                    "chunk_id": metadata.get("chunk_id"),
                    "page": metadata.get("page"),
                    "section": metadata.get("section"),
                    "document": metadata.get("document"),
                    "text": metadata.get("text"),
                    "score": match.get("score")
                })
            return results
        except Exception as e:
            logger.error(f"Failed retrieving from Pinecone: {e}")
            raise RuntimeError(f"Retrieval failed: {e}")
