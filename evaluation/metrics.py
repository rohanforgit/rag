import os
import sys
import json
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("eval_metrics")

# Initialize SentenceTransformer model lazily
_sentence_model = None

def get_sentence_transformer():
    global _sentence_model
    if _sentence_model is None:
        logger.info("Loading sentence-transformers model 'all-MiniLM-L6-v2'...")
        _sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _sentence_model

def calculate_semantic_similarity(str1: str, str2: str) -> float:
    """
    Computes cosine similarity between two strings using sentence-transformers.
    """
    if not str1.strip() or not str2.strip():
        return 0.0
    try:
        model = get_sentence_transformer()
        embeddings = model.encode([str1, str2])
        sim = cosine_similarity([embeddings[0]], [embeddings[1]])
        return float(sim[0][0])
    except Exception as e:
        logger.error(f"Error computing semantic similarity: {e}")
        return 0.0

def calculate_exact_match(str1: str, str2: str) -> float:
    """
    Computes binary exact match (1.0 if identical after basic normalization, else 0.0).
    """
    norm1 = " ".join(str1.lower().strip().split())
    norm2 = " ".join(str2.lower().strip().split())
    return 1.0 if norm1 == norm2 else 0.0

def calculate_retrieval_metrics(retrieved_sources: list, expected_doc: str, expected_sec: str = None) -> dict:
    """
    Computes retrieval metrics comparing retrieved sources against expected document/section.
    retrieved_sources: list of dicts, e.g., [{"document": "...", "section": "..."}]
    """
    if not retrieved_sources:
        return {
            "hit_rate": 0.0,
            "mrr": 0.0,
            "recall_at_5": 0.0,
            "recall_at_10": 0.0,
            "precision_at_5": 0.0
        }
        
    # Check match for each retrieved source
    matches = []
    for src in retrieved_sources:
        doc_name = src.get("document", "") or src.get("doc", "")
        sec_name = src.get("section", "") or ""
        
        # Substring or exact match for robustness
        doc_match = expected_doc.lower() in doc_name.lower() or doc_name.lower() in expected_doc.lower()
        sec_match = True
        if expected_sec:
            sec_match = expected_sec.lower() in sec_name.lower() or sec_name.lower() in expected_sec.lower()
            
        matches.append(1.0 if (doc_match and sec_match) else 0.0)
        
    hit_rate = 1.0 if sum(matches) > 0 else 0.0
    
    # MRR (Mean Reciprocal Rank)
    mrr = 0.0
    for idx, match in enumerate(matches):
        if match == 1.0:
            mrr = 1.0 / (idx + 1)
            break
            
    # Recall@5 and Recall@10
    # Since we have exactly one expected document/section target, Recall is 1.0 if matched, 0.0 otherwise.
    recall_at_5 = 1.0 if sum(matches[:5]) > 0 else 0.0
    recall_at_10 = 1.0 if sum(matches[:10]) > 0 else 0.0
    
    # Precision@5
    precision_at_5 = sum(matches[:5]) / 5.0
    
    return {
        "hit_rate": hit_rate,
        "mrr": mrr,
        "recall_at_5": recall_at_5,
        "recall_at_10": recall_at_10,
        "precision_at_5": precision_at_5
    }

class LLMJudgeEvaluator:
    """
    Provides robust, rate-limit safe RAG evaluation metrics using Groq API as an LLM judge.
    Acts as a reliable backup/replacement for the Ragas library.
    """
    def __init__(self, api_key: str, model_name: str = "llama-3.3-70b-versatile"):
        self.client = Groq(api_key=api_key)
        self.model = model_name
        
    def _call_judge(self, system_prompt: str, user_prompt: str) -> float:
        try:
            completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model=self.model,
                temperature=0.0
            )
            resp = completion.choices[0].message.content.strip()
            # Find any float/number in response
            import re
            match = re.search(r"(\d+(?:\.\d+)?)", resp)
            if match:
                val = float(match.group(1))
                return min(max(val, 0.0), 1.0) # Clamp between 0.0 and 1.0
            return 0.5
        except Exception as e:
            logger.error(f"LLM Judge call failed: {e}")
            return 0.5

    def evaluate_faithfulness(self, answer: str, context: str) -> float:
        """
        Faithfulness: Is the generated answer fully grounded in and supported by the retrieved context?
        """
        if not answer.strip():
            return 0.0
        if not context.strip():
            return 0.0
            
        system_prompt = (
            "You are an expert AI evaluator assessing RAG system Faithfulness.\n"
            "Analyze the generated answer and the provided context. Determine if all facts stated in "
            "the generated answer are directly supported by the context.\n"
            "Respond ONLY with a single numeric score between 0.0 (completely hallucinated/not supported) "
            "and 1.0 (100% supported by the context). Do not write any explanations or text."
        )
        user_prompt = f"Context:\n{context}\n\nGenerated Answer:\n{answer}"
        return self._call_judge(system_prompt, user_prompt)

    def evaluate_context_recall(self, ground_truth: str, context: str) -> float:
        """
        Context Recall: Does the retrieved context contain all the information from the ground truth answer?
        """
        if not ground_truth.strip():
            return 1.0
        if not context.strip():
            return 0.0
            
        system_prompt = (
            "You are an expert AI evaluator assessing RAG system Context Recall.\n"
            "Analyze the ground truth answer and the retrieved context. Determine what percentage of the "
            "information in the ground truth answer is present in the retrieved context.\n"
            "Respond ONLY with a single numeric score between 0.0 (none of the ground truth is in the context) "
            "and 1.0 (all ground truth details are covered in the context). Do not write any explanations or text."
        )
        user_prompt = f"Ground Truth:\n{ground_truth}\n\nRetrieved Context:\n{context}"
        return self._call_judge(system_prompt, user_prompt)

    def evaluate_context_precision(self, question: str, retrieved_chunks: list) -> float:
        """
        Context Precision: Are the retrieved context chunks relevant to the user question?
        """
        if not question.strip() or not retrieved_chunks:
            return 0.0
            
        combined_chunks = "\n\n".join([f"Chunk {i+1}: {chunk}" for i, chunk in enumerate(retrieved_chunks)])
        system_prompt = (
            "You are an expert AI evaluator assessing RAG system Context Precision.\n"
            "Analyze the user question and the retrieved chunks. Determine the proportion of retrieved chunks "
            "that are relevant to answering the question.\n"
            "Respond ONLY with a single numeric score between 0.0 (no chunks are relevant) and 1.0 (all chunks are relevant). "
            "Do not write any explanations or text."
        )
        user_prompt = f"Question: {question}\n\nRetrieved Chunks:\n{combined_chunks}"
        return self._call_judge(system_prompt, user_prompt)

    def evaluate_answer_relevancy(self, question: str, answer: str) -> float:
        """
        Answer Relevancy: Does the generated answer directly address the user question without redundant/fluff info?
        """
        if not question.strip() or not answer.strip():
            return 0.0
            
        system_prompt = (
            "You are an expert AI evaluator assessing RAG system Answer Relevancy.\n"
            "Analyze the user question and the generated answer. Rate how relevant, direct, and helpful "
            "the answer is to the question. Penalize answers that are vague, incomplete, or contain fluff.\n"
            "Respond ONLY with a single numeric score between 0.0 (completely irrelevant) and 1.0 (extremely relevant and precise). "
            "Do not write any explanations or text."
        )
        user_prompt = f"Question: {question}\n\nGenerated Answer:\n{answer}"
        return self._call_judge(system_prompt, user_prompt)

    def evaluate_answer_accuracy(self, ground_truth: str, answer: str) -> float:
        """
        Answer Accuracy (LLM-as-a-judge): How factually accurate is the generated answer compared to the ground truth?
        """
        if not ground_truth.strip() or not answer.strip():
            return 0.0
            
        system_prompt = (
            "You are an expert AI evaluator assessing Answer Accuracy.\n"
            "Compare the generated answer against the ground truth answer. Determine if the generated answer "
            "contains the same facts and correctly answers the question as defined by the ground truth.\n"
            "Respond ONLY with a single numeric score between 0.0 (completely incorrect/inaccurate) and 1.0 (completely correct). "
            "Do not write any explanations or text."
        )
        user_prompt = f"Ground Truth:\n{ground_truth}\n\nGenerated Answer:\n{answer}"
        return self._call_judge(system_prompt, user_prompt)

    def evaluate_all(self, question: str, answer: str, ground_truth: str, context_chunks: list) -> dict:
        """
        Evaluates all 5 RAG metrics (faithfulness, context_recall, context_precision, answer_relevancy, answer_accuracy)
        in a single LLM API call to maximize speed and minimize rate limit risks.
        """
        if not answer.strip():
            return {
                "faithfulness": 0.0,
                "context_recall": 0.0,
                "context_precision": 0.0,
                "answer_relevancy": 0.0,
                "answer_accuracy": 0.0
            }
            
        combined_context = "\n\n".join(context_chunks) if isinstance(context_chunks, list) else context_chunks
        combined_chunks_labeled = "\n\n".join([f"Chunk {i+1}: {chunk}" for i, chunk in enumerate(context_chunks)]) if isinstance(context_chunks, list) else context_chunks

        system_prompt = (
            "You are an expert AI evaluator assessing RAG system performance.\n"
            "Evaluate the system output on these 5 metrics on a scale from 0.0 (very poor) to 1.0 (perfect):\n"
            "1. faithfulness: Are the facts in the answer fully supported by the retrieved context?\n"
            "2. context_recall: Does the retrieved context cover all details in the ground truth answer?\n"
            "3. context_precision: Are the retrieved chunks relevant to the question?\n"
            "4. answer_relevancy: Does the answer directly address the question without fluff?\n"
            "5. answer_accuracy: How factually accurate is the answer compared to the ground truth?\n\n"
            "You MUST respond ONLY with a raw JSON object containing these 5 keys: "
            "'faithfulness', 'context_recall', 'context_precision', 'answer_relevancy', 'answer_accuracy'.\n"
            "Do not include any markup, markdown code blocks, explanations, or text other than the JSON object."
        )
        
        user_prompt = (
            f"Question: {question}\n\n"
            f"Retrieved Context / Chunks:\n{combined_chunks_labeled}\n\n"
            f"Generated Answer:\n{answer}\n\n"
            f"Ground Truth:\n{ground_truth}"
        )
        
        try:
            completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model=self.model,
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            resp = completion.choices[0].message.content.strip()
            data = json.loads(resp)
            
            # Extract and clamp scores
            scores = {}
            for k in ['faithfulness', 'context_recall', 'context_precision', 'answer_relevancy', 'answer_accuracy']:
                val = float(data.get(k, 0.5))
                scores[k] = min(max(val, 0.0), 1.0)
            return scores
        except Exception as e:
            logger.error(f"Batch LLM judge call failed: {e}. Falling back to individual evaluations.")
            # Fallback to individual evaluations
            return {
                "faithfulness": self.evaluate_faithfulness(answer, combined_context),
                "context_recall": self.evaluate_context_recall(ground_truth, combined_context),
                "context_precision": self.evaluate_context_precision(question, context_chunks if isinstance(context_chunks, list) else [context_chunks]),
                "answer_relevancy": self.evaluate_answer_relevancy(question, answer),
                "answer_accuracy": self.evaluate_answer_accuracy(ground_truth, answer)
            }
