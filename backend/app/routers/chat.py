from fastapi import APIRouter, HTTPException
import uuid
import time
import logging
from app.models.schemas import ChatRequest, ChatResponse, SourceInfo
from app.services.embedder import EmbedderService
from app.services.retriever import RetrieverService
from app.services.llm import LLMService
from app.services.prompt_builder import PromptBuilder
from app.services.conversation import ConversationManager

logger = logging.getLogger(__name__)

router = APIRouter()

# Instantiate services as singletons at module level
try:
    embedder_service = EmbedderService()
    retriever_service = RetrieverService(embedder_service)
    llm_service = LLMService()
except Exception as e:
    # Services initialization might fail initially due to missing credentials in env,
    # we log it but continue so that health checks or configuration updates can still run.
    logger.warning(f"Core services failed to initialize: {e}. Ensure API keys are set in backend/.env.")
    embedder_service = None
    retriever_service = None
    llm_service = None

conversation_manager = ConversationManager()

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    global embedder_service, retriever_service, llm_service
    start_time = time.time()
    
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Empty question provided.")
        
    session_id = request.session_id or str(uuid.uuid4())
    logger.info(f"Received query: '{request.message}' for session: {session_id}")
    
    # Lazy services initialization or validation check
    if not embedder_service or not retriever_service or not llm_service:
        try:
            embedder_service = EmbedderService()
            retriever_service = RetrieverService(embedder_service)
            llm_service = LLMService()
        except Exception as e:
            logger.error(f"Configuration or validation error on query: {e}")
            raise HTTPException(
                status_code=503, 
                detail=f"Service unavailable due to missing or invalid credentials. {e}"
            )
            
    try:
        # 1. Retrieve matching chunks from Pinecone
        retrieval_start = time.time()
        top_k = request.top_k or 5
        retrieved_chunks = retriever_service.retrieve(request.message, top_k=top_k)
        retrieval_time = time.time() - retrieval_start
        
        # 2. Extract unique sources
        sources = []
        seen_sources = set()
        for chunk in retrieved_chunks:
            page = chunk.get("page")
            section = chunk.get("section")
            doc = chunk.get("document", "Unknown")
            if page is not None and section:
                source_key = (page, section, doc)
                if source_key not in seen_sources:
                    seen_sources.add(source_key)
                    sources.append(SourceInfo(page=page, section=section, document=doc))
                    
        # Sort sources by document and page number for clean representation
        sources.sort(key=lambda s: (s.document, s.page))
        
        # 3. Retrieve conversation history
        history = conversation_manager.get_history(session_id)
        
        # 4. Build prompt
        system_prompt = PromptBuilder.build_system_prompt(retrieved_chunks)
        
        # Construct message list for Groq API
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history)
        messages.append({"role": "user", "content": request.message})
        
        # 5. Generate grounded answer
        llm_start = time.time()
        answer = llm_service.generate_answer(messages)
        llm_latency = time.time() - llm_start
        
        # 6. Save turn to memory
        conversation_manager.add_message(session_id, "user", request.message)
        conversation_manager.add_message(session_id, "assistant", answer)
        
        total_time = time.time() - start_time
        
        # Logging requirements from Section 21
        chunk_ids = [c.get("chunk_id") for c in retrieved_chunks]
        logger.info(
            f"Query Metrics: Session={session_id} | ChunksRetrieved={chunk_ids} | "
            f"RetrievalTime={retrieval_time:.3f}s | LLMLatency={llm_latency:.3f}s | "
            f"TotalResponseTime={total_time:.3f}s"
        )
        
        return ChatResponse(
            answer=answer,
            sources=sources,
            session_id=session_id
        )
        
    except Exception as e:
        logger.error(f"Error handling query request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@router.delete("/chat/{session_id}")
async def clear_session(session_id: str):
    conversation_manager.clear_session(session_id)
    return {"status": "success", "detail": f"Session {session_id} cleared."}
