import os
import sys
import json
import pandas as pd
import time
import argparse
import logging
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("evaluate_orchestrator")

# Add backend directory to sys.path
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend"))
sys.path.append(backend_dir)

from evaluation.utils import ChatbotClient
from evaluation.metrics import (
    calculate_semantic_similarity,
    calculate_exact_match,
    calculate_retrieval_metrics,
    LLMJudgeEvaluator
)
from app.services.embedder import EmbedderService
from app.services.retriever import RetrieverService

def run_evaluation(
    endpoint_url: str,
    req_field: str,
    resp_field: str,
    src_field: str,
    limit: int
):
    # Load env variables
    backend_env_path = os.path.join(backend_dir, ".env")
    if os.path.exists(backend_env_path):
        load_dotenv(backend_env_path)
    else:
        load_dotenv()
        
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key or "your-groq" in groq_api_key.lower():
        logger.error("GROQ_API_KEY is missing or invalid in environment.")
        logger.error("Please ensure backend/.env contains a valid GROQ_API_KEY.")
        sys.exit(1)

    # Initialize chatbot client
    client = ChatbotClient(
        endpoint_url=endpoint_url,
        request_field=req_field,
        response_field=resp_field,
        sources_field=src_field
    )
    
    # Initialize local retriever to get text chunks
    try:
        embedder_service = EmbedderService()
        retriever_service = RetrieverService(embedder_service)
    except Exception as e:
        logger.error(f"Failed to initialize local retriever services: {e}")
        logger.error("Make sure Pinecone credentials in the env are correct.")
        sys.exit(1)
        
    # Initialize LLM judge evaluator
    judge = LLMJudgeEvaluator(api_key=groq_api_key, model_name=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"))
    
    # Load dataset
    dataset_path = os.path.join(os.path.dirname(__file__), "dataset.json")
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset file not found at: {dataset_path}")
        logger.error("Please run generate_dataset.py first to create the dataset.")
        sys.exit(1)
        
    with open(dataset_path, "r", encoding="utf-8") as f:
        test_cases = json.load(f)
        
    if limit > 0:
        logger.info(f"Limiting evaluation to the first {limit} test cases.")
        test_cases = test_cases[:limit]
        
    total_cases = len(test_cases)
    logger.info(f"Starting evaluation of {total_cases} test cases...")
    
    results = []
    
    for case in test_cases:
        q_id = case["id"]
        question = case["question"]
        ground_truth = case["ground_truth"]
        expected_doc = case["expected_document"]
        expected_sec = case.get("expected_section", "")
        topic = case.get("topic", "General")
        diff = case.get("difficulty", "medium")
        
        logger.info(f"[{q_id}/{total_cases}] Evaluating question: '{question}' (Topic: {topic})")
        
        # 1. Query Chatbot Endpoint
        # We request top_k=10 to compute Recall@10
        query_start = time.time()
        chat_res = client.query(question, top_k=10)
        query_time = chat_res["latency"]
        
        prediction = chat_res["answer"]
        retrieved_sources = chat_res["sources"] # list of dicts: document, section, page
        
        if not chat_res["success"]:
            logger.warning(f"Chatbot query failed for case ID {q_id}: {chat_res['error']}. Using fallback.")
            prediction = "Error: Chatbot endpoint failed."
            retrieved_sources = []
            
        # 2. Get local chunks for context evaluations
        try:
            chunks = retriever_service.retrieve(question, top_k=5)
            chunk_texts = [c.get("text", "") for c in chunks]
            retrieved_docs_local = [c.get("document", "") for c in chunks]
            retrieved_chunks_str = " | ".join([f"Page {c.get('page')}: {c.get('text')[:100]}..." for c in chunks])
        except Exception as e:
            logger.error(f"Failed to retrieve local chunks: {e}")
            chunk_texts = []
            retrieved_docs_local = []
            retrieved_chunks_str = "N/A"
            
        context_combined = "\n\n".join(chunk_texts)
        
        # 3. Calculate Retrieval Metrics
        ret_metrics = calculate_retrieval_metrics(
            retrieved_sources=retrieved_sources if retrieved_sources else [{"document": d, "section": ""} for d in retrieved_docs_local],
            expected_doc=expected_doc,
            expected_sec=expected_sec
        )
        
        # 4. Calculate Similarity Metrics
        semantic_score = calculate_semantic_similarity(prediction, ground_truth)
        em_score = calculate_exact_match(prediction, ground_truth)
        
        # 5. LLM-as-a-judge Generation and RAG Metrics
        # Rate limit safety sleep
        time.sleep(0.5)
        
        judge_scores = judge.evaluate_all(question, prediction, ground_truth, chunk_texts)
        
        faithfulness = judge_scores["faithfulness"]
        context_recall = judge_scores["context_recall"]
        context_precision = judge_scores["context_precision"]
        answer_relevancy = judge_scores["answer_relevancy"]
        accuracy = judge_scores["answer_accuracy"]
        
        hallucination = 1.0 - faithfulness
        citation_accuracy = ret_metrics["hit_rate"] # Hit rate represents citation accuracy
        
        # Compile result row
        res_row = {
            "id": q_id,
            "Question": question,
            "Ground Truth": ground_truth,
            "Prediction": prediction,
            "Correct": accuracy,  # LLM Answer Accuracy
            "Semantic Score": semantic_score,
            "Exact Match": em_score,
            "Retrieved Docs": ", ".join(list(set([src.get("document", "") for src in retrieved_sources])) if retrieved_sources else list(set(retrieved_docs_local))),
            "Retrieved Chunks": retrieved_chunks_str,
            "Latency": query_time,
            "Hallucination": hallucination,
            "Faithfulness": faithfulness,
            "Context Recall": context_recall,
            "Context Precision": context_precision,
            "Answer Relevancy": answer_relevancy,
            "Hit Rate": ret_metrics["hit_rate"],
            "MRR": ret_metrics["mrr"],
            "Recall@5": ret_metrics["recall_at_5"],
            "Recall@10": ret_metrics["recall_at_10"],
            "Precision@5": ret_metrics["precision_at_5"],
            "Citation Accuracy": citation_accuracy,
            "Topic": topic,
            "Difficulty": diff
        }
        
        results.append(res_row)
        
    # Write to CSV
    df = pd.DataFrame(results)
    output_csv = os.path.join(os.path.dirname(__file__), "results.csv")
    
    # Save specific columns requested for results.csv
    csv_columns = [
        "Question", "Ground Truth", "Prediction", "Correct", "Semantic Score",
        "Retrieved Docs", "Retrieved Chunks", "Latency", "Hallucination", "Faithfulness"
    ]
    # We save all columns to a complete CSV to make report generation easy, 
    # but write the requested columns clearly to results.csv
    df[csv_columns].to_csv(output_csv, index=False)
    
    # Also save an intermediate raw results for the report generator to use
    raw_csv = os.path.join(os.path.dirname(__file__), "results_raw.csv")
    df.to_csv(raw_csv, index=False)
    
    logger.info(f"Evaluation complete. Results written to '{output_csv}' and raw metrics to '{raw_csv}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Evaluation Framework Orchestrator")
    parser.add_argument("--url", type=str, default="http://localhost:8000/chat", help="Chatbot API endpoint URL")
    # By default, we use 'message' for the request field to match Sreenidhi's FastAPI endpoint, 
    # but it can be overridden to 'question' as required by user prompt.
    parser.add_argument("--req-field", type=str, default="message", help="Request field name (message/question)")
    parser.add_argument("--resp-field", type=str, default="answer", help="Response field name")
    parser.add_argument("--src-field", type=str, default="sources", help="Sources field name")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of evaluation questions (0 = evaluate all)")
    
    args = parser.parse_args()
    
    # Start timer
    start_eval = time.time()
    run_evaluation(
        endpoint_url=args.url,
        req_field=args.req_field,
        resp_field=args.resp_field,
        src_field=args.src_field,
        limit=args.limit
    )
    logger.info(f"Total evaluation run took {time.time() - start_eval:.2f} seconds.")
