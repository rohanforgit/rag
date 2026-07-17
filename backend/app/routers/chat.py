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

def get_linear_sequences(text: str) -> dict:
    import re
    def is_valid_list_item(start_pos: int) -> bool:
        if start_pos > 0 and text[start_pos - 1] == '.':
            return False
        preceding = text[max(0, start_pos - 45):start_pos]
        # Ignore if preceded by words indicating references to sections, semesters, dates, etc.
        if re.search(r'\b(?:semester|regulation|section|table|figure|version|page|level|class|grade|cgpa|sgpa|academic|july|credits?|point|rule|clause|appendix|annexure|marks?)\s+$', preceding.lower()):
            return False
        return True

    # Arabic
    arabic_matches = []
    for m in re.finditer(r'\b(\d+)[\.\)]\s|\((\d+)\)\s|\b(?:PO|PEO|PSO|CO|LO)\s+(\d+)\b', text):
        num_str = m.group(1) or m.group(2) or m.group(3)
        if num_str:
            if is_valid_list_item(m.start()):
                arabic_matches.append(int(num_str))
            
    # Alphabetical
    alpha_matches = []
    for m in re.finditer(r'\b([a-k])[\.\)]\s|\(([a-k])\)\s', text.lower()):
        char_str = m.group(1) or m.group(2)
        if char_str:
            if is_valid_list_item(m.start()):
                alpha_matches.append(ord(char_str) - ord('a') + 1)
            
    # Roman
    roman_map = {'i':1, 'ii':2, 'iii':3, 'iv':4, 'v':5, 'vi':6, 'vii':7, 'viii':8, 'ix':9, 'x':10}
    roman_matches = []
    for m in re.finditer(r'\b([ivx]+)[\.\)]\s|\(([ivx]+)\)\s', text.lower()):
        roman_str = m.group(1) or m.group(2)
        if roman_str in roman_map:
            if is_valid_list_item(m.start()):
                roman_matches.append(roman_map[roman_str])
            
    def extract_seqs(matches):
        if not matches:
            return []
        seqs = []
        curr = []
        for x in matches:
            if not curr:
                curr.append(x)
            else:
                if x == curr[-1] + 1:
                    curr.append(x)
                elif x <= curr[-1]:
                    if len(curr) >= 3:
                        seqs.append(curr)
                    curr = [x]
        if len(curr) >= 3:
            seqs.append(curr)
        return seqs

    return {
        "arabic": extract_seqs(arabic_matches),
        "alpha": extract_seqs(alpha_matches),
        "roman": extract_seqs(roman_matches)
    }

def is_answer_incomplete(query: str, context: str, answer: str) -> bool:
    """
    Programmatically detects if a list in the context is only partially represented in the answer.
    """
    import re
    
    # 1. Skip completeness check if user is asking for a specific list item (e.g., "what is rule 3?")
    query_lower = query.lower()
    specific_patterns = [
        r'\b(?:rule|point|item|section|clause|condition)\s+\d+',
        r'\b(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\b',
        r'\b(?:1st|2nd|3rd|4th|5th|6th|7th|8th|9th|10th)\b',
        r'\bhow\s+many\b'
    ]
    for pattern in specific_patterns:
        if re.search(pattern, query_lower):
            return False
            
    # 2. Extract linear list sequences from the text
    context_markers = get_linear_sequences(context)
    answer_markers = get_linear_sequences(answer)
    
    # Helper to get the first item's text from a list (supports standard lists and PO/PEO/PSO lists)
    def get_list_item_text(text_content: str, num: int) -> str:
        pattern = r'(?:\b(?:PO|PEO|PSO|CO|LO)\s+)?\b' + str(num) + r'(?:[\.\)]\s+|\s+)(.*?)(?=(?:\b(?:PO|PEO|PSO|CO|LO)\s+)?\b' + str(num + 1) + r'(?:[\.\)]\s+|\s+)|\Z)'
        match = re.search(pattern, text_content, re.DOTALL)
        return match.group(1).strip() if match else ""

    # Helper to get clean word set
    def get_clean_words(item_text: str) -> set:
        stop_words = {'the', 'a', 'an', 'and', 'of', 'to', 'in', 'is', 'for', 'on', 'with', 'by', 'as', 'be', 'or', 'from', 'at', 'this', 'that', 'shall', 'will', 'are', 'was', 'were'}
        words = re.findall(r'\w+', item_text.lower())
        return {w for w in words if w not in stop_words and len(w) > 2}

    # Check if any context sequence is truncated in the answer (only check arabic lists)
    c_seqs = context_markers["arabic"]
    a_seqs = answer_markers["arabic"]
    
    for c_seq in c_seqs:
        c_max = max(c_seq)
        c_min = min(c_seq)
        if c_min <= 2:
            c_item = get_list_item_text(context, c_min)
            c_words = get_clean_words(c_item)
            if not c_words:
                continue
                
            for a_seq in a_seqs:
                a_max = max(a_seq)
                a_min = min(a_seq)
                if a_min == c_min and a_max < c_max:
                    a_item = get_list_item_text(answer, a_min)
                    a_words = get_clean_words(a_item)
                    overlap = c_words.intersection(a_words)
                    union = c_words.union(a_words)
                    jaccard = len(overlap) / max(1, len(union))
                    # If similarity is >= 0.55, they are the same list and the answer list is shorter
                    if jaccard >= 0.55:
                        return True
                        
    return False

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
        
        # 4. Merge contiguous chunks and build prompt
        merged_chunks = PromptBuilder.merge_contiguous_chunks(retrieved_chunks)
        system_prompt = PromptBuilder.build_system_prompt(merged_chunks)
        
        # Construct message list for Groq API
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history)
        messages.append({"role": "user", "content": request.message})
        
        # 5. Generate grounded answer
        llm_start = time.time()
        answer = llm_service.generate_answer(messages)
        
        # 5b. Validate answer completeness and regenerate if needed
        context_text = "\n".join([c.get("text", "") for c in merged_chunks])
        if is_answer_incomplete(request.message, context_text, answer):
            logger.info("Answer detected as incomplete. Regenerating answer with strict completeness prompts...")
            regeneration_messages = list(messages)
            regeneration_messages.append({"role": "assistant", "content": answer})
            regeneration_messages.append({
                "role": "user",
                "content": (
                    "The list above appears to be incomplete. "
                    "Please review the context and output the COMPLETE list of all points/items "
                    "without omitting any details or summarizing."
                )
            })
            answer = llm_service.generate_answer(regeneration_messages)
            
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
