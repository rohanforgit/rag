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
        Also retrieves neighboring chunks if list patterns are detected.
        """
        try:
            # 1. Embed query
            query_embeddings = self.embedder.get_embeddings([query], is_query=True)
            if not query_embeddings:
                raise ValueError("Could not embed query")
            query_vector = query_embeddings[0]

            # 2. Search index
            # Increase top_k if the query is a list-based query
            import re
            is_list_query = any(w in query.lower() for w in ["list", "what are", "rules", "regulations", "guidelines", "criteria", "conditions", "requirements", "all", "eligibility", "points", "steps"])
            effective_top_k = max(top_k, 8) if is_list_query else top_k

            response = self.index.query(
                vector=query_vector,
                top_k=effective_top_k,
                include_metadata=True
            )

            # 3. Format results
            initial_results = []
            for match in response.get("matches", []):
                metadata = match.get("metadata", {})
                initial_results.append({
                    "chunk_id": metadata.get("chunk_id"),
                    "page": metadata.get("page"),
                    "section": metadata.get("section"),
                    "document": metadata.get("document"),
                    "text": metadata.get("text"),
                    "score": match.get("score")
                })

            # 4. Fetch neighbors if list patterns are detected
            def contains_list_pattern(text: str) -> bool:
                patterns = [
                    r'\b\d+[\.\)]\s',
                    r'\b[a-zA-Z][\.\)]\s',
                    r'\([a-zA-Z0-9]\)\s',
                    r'\b[ivxIVX]+[\.\)]\s',
                    r'[\u2022\-\*]\s',
                    r'\b(?:PO|PEO|PSO|CO|LO|Unit)\s+\d+\b'
                ]
                combined = '|'.join(patterns)
                return bool(re.search(combined, text))

            # Find chunks that have list patterns
            list_chunks = [c for c in initial_results if contains_list_pattern(c["text"])]
            
            neighbor_ids = set()
            retrieved_ids = {c["chunk_id"] for c in initial_results if c.get("chunk_id")}
            
            for chunk in list_chunks:
                chunk_id = chunk["chunk_id"]
                if not chunk_id:
                    continue
                # Parse chunk prefix and index
                match = re.match(r'^(.*)_(\d+)$', chunk_id)
                if match:
                    prefix = match.group(1)
                    idx = int(match.group(2))
                    
                    # Generate neighbors
                    # Check idx - 1 and idx + 1
                    if idx > 0:
                        prev_id = f"{prefix}_{idx-1}"
                        if prev_id not in retrieved_ids:
                            neighbor_ids.add(prev_id)
                    
                    next_id = f"{prefix}_{idx+1}"
                    if next_id not in retrieved_ids:
                        neighbor_ids.add(next_id)

            # Fetch neighbor details in one call
            all_chunks = {c["chunk_id"]: c for c in initial_results if c.get("chunk_id")}
            
            if neighbor_ids:
                logger.info(f"Fetching {len(neighbor_ids)} neighbor chunks for list continuity: {neighbor_ids}")
                fetch_resp = self.index.fetch(ids=list(neighbor_ids))
                vectors = fetch_resp.get("vectors", {})
                
                newly_fetched = []
                for vid, vdata in vectors.items():
                    metadata = vdata.get("metadata", {})
                    chunk = {
                        "chunk_id": metadata.get("chunk_id"),
                        "page": metadata.get("page"),
                        "section": metadata.get("section"),
                        "document": metadata.get("document"),
                        "text": metadata.get("text"),
                        "score": 0.5  # default score for fetched neighbor
                    }
                    all_chunks[vid] = chunk
                    newly_fetched.append(chunk)
                    
                # Secondary lookup check: if any newly fetched succeeding neighbor has a list pattern,
                # we can fetch idx + 2 to be extra thorough.
                second_neighbor_ids = set()
                retrieved_ids.update(all_chunks.keys())
                
                for chunk in newly_fetched:
                    if contains_list_pattern(chunk["text"]):
                        chunk_id = chunk["chunk_id"]
                        match = re.match(r'^(.*)_(\d+)$', chunk_id)
                        if match:
                            prefix = match.group(1)
                            idx = int(match.group(2))
                            # Check next neighbor
                            next_id = f"{prefix}_{idx+1}"
                            if next_id not in retrieved_ids:
                                second_neighbor_ids.add(next_id)
                                
                if second_neighbor_ids:
                    logger.info(f"Fetching second-level neighbor chunks: {second_neighbor_ids}")
                    fetch_resp2 = self.index.fetch(ids=list(second_neighbor_ids))
                    vectors2 = fetch_resp2.get("vectors", {})
                    for vid, vdata in vectors2.items():
                        metadata = vdata.get("metadata", {})
                        chunk = {
                            "chunk_id": metadata.get("chunk_id"),
                            "page": metadata.get("page"),
                            "section": metadata.get("section"),
                            "document": metadata.get("document"),
                            "text": metadata.get("text"),
                            "score": 0.4
                        }
                        all_chunks[vid] = chunk

            # Return all accumulated chunks sorted by document, page, and chunk index
            def get_sort_key(c):
                doc = c.get("document", "")
                page = c.get("page", 0)
                chunk_id = c.get("chunk_id", "")
                match = re.match(r'^(.*)_(\d+)$', chunk_id)
                idx = int(match.group(2)) if match else 0
                return (doc, page, idx)

            return sorted(list(all_chunks.values()), key=get_sort_key)
            
        except Exception as e:
            logger.error(f"Failed retrieving from Pinecone: {e}")
            raise RuntimeError(f"Retrieval failed: {e}")
