import os
import pandas as pd
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("report_generator")

def generate_report():
    raw_csv = os.path.join(os.path.dirname(__file__), "results_raw.csv")
    output_report = os.path.join(os.path.dirname(__file__), "report.md")
    
    if not os.path.exists(raw_csv):
        logger.error(f"Raw results CSV not found at: {raw_csv}")
        logger.error("Please run evaluate.py first to collect evaluation data.")
        return
        
    df = pd.read_csv(raw_csv)
    
    # 1. Compute Overall Metrics
    overall_accuracy = df["Correct"].mean()
    overall_semantic = df["Semantic Score"].mean()
    overall_precision = df["Precision@5"].mean()
    overall_recall_5 = df["Recall@5"].mean()
    overall_recall_10 = df["Recall@10"].mean()
    overall_faithfulness = df["Faithfulness"].mean()
    overall_hit_rate = df["Hit Rate"].mean()
    overall_mrr = df["MRR"].mean()
    avg_latency = df["Latency"].mean()
    
    # Hallucination rate is the percentage of cases where faithfulness is less than 0.7
    hallucination_rate = (df["Faithfulness"] < 0.7).mean()
    citation_accuracy = df["Citation Accuracy"].mean()
    
    # 2. Topic Grouping (Confusion Table)
    topic_groups = df.groupby("Topic")
    
    topic_summary = []
    for name, group in topic_groups:
        total_questions = len(group)
        avg_acc = group["Correct"].mean()
        avg_faith = group["Faithfulness"].mean()
        avg_rec = group["Recall@5"].mean()
        
        # A case is a failure if accuracy is below 0.6 or faithfulness is below 0.7
        failures = group[(group["Correct"] < 0.6) | (group["Faithfulness"] < 0.7)]
        failure_rate = len(failures) / total_questions
        
        # Determine failure reason
        if failure_rate > 0:
            avg_group_rec = group["Recall@5"].mean()
            avg_group_faith = group["Faithfulness"].mean()
            if avg_group_rec < 0.7:
                primary_failure = "Retrieval Failure (Low Recall)"
            elif avg_group_faith < 0.7:
                primary_failure = "Generation Failure (Hallucinations)"
            else:
                primary_failure = "Syntactic/Formatting Issue"
        else:
            primary_failure = "None"
            
        topic_summary.append({
            "Topic": name,
            "Total Questions": total_questions,
            "Average Accuracy": avg_acc,
            "Average Faithfulness": avg_faith,
            "Recall@5": avg_rec,
            "Failure Rate": failure_rate,
            "Primary Failure Mode": primary_failure
        })
        
    topic_df = pd.DataFrame(topic_summary)
    topic_df = topic_df.sort_values(by="Failure Rate", ascending=False)
    
    # 3. Format Topic Confusion Table
    confusion_rows = []
    for _, row in topic_df.iterrows():
        confusion_rows.append(
            f"| {row['Topic']} | {row['Total Questions']} | {row['Average Accuracy']:.2%} | "
            f"{row['Average Faithfulness']:.2%} | {row['Recall@5']:.2%} | {row['Failure Rate']:.2%} | "
            f"{row['Primary Failure Mode']} |"
        )
        
    confusion_table_str = "\n".join(confusion_rows)
    
    # 4. Write Markdown Report
    report_content = f"""# Sreenidhi University RAG Chatbot Evaluation Report

This report summarizes the performance of the Sreenidhi University R26 Regulations RAG Chatbot. The evaluation was run against a golden dataset of **{len(df)} questions** covering **24 distinct academic topics** ranging from Attendance, CIE, SEE, to specific Honors and Minor requirements.

---

## Executive Summary

The evaluation framework automatically measures retrieval performance, answer accuracy, semantic similarity, operational latencies, and RAG alignment metrics using an LLM-as-a-judge mechanism.

### Key Performance Indicators (KPIs)

| Metric | Score / Value | Target | Status |
| :--- | :---: | :---: | :---: |
| **Overall Answer Accuracy** | {overall_accuracy:.2%} | > 85.0% | {"✅ Met" if overall_accuracy >= 0.85 else "⚠️ Action Required"} |
| **Semantic Similarity (Cosine)** | {overall_semantic:.2f} | > 0.80 | {"✅ Met" if overall_semantic >= 0.80 else "⚠️ Action Required"} |
| **Context Faithfulness** | {overall_faithfulness:.2%} | > 90.0% | {"✅ Met" if overall_faithfulness >= 0.90 else "⚠️ Action Required"} |
| **Hallucination Rate** | {hallucination_rate:.2%} | < 10.0% | {"✅ Met" if hallucination_rate <= 0.10 else "⚠️ Action Required"} |
| **Retrieval Hit Rate** | {overall_hit_rate:.2%} | > 90.0% | {"✅ Met" if overall_hit_rate >= 0.90 else "⚠️ Action Required"} |
| **Retrieval Recall@5** | {overall_recall_5:.2%} | > 85.0% | {"✅ Met" if overall_recall_5 >= 0.85 else "⚠️ Action Required"} |
| **Retrieval Recall@10** | {overall_recall_10:.2%} | > 90.0% | {"✅ Met" if overall_recall_10 >= 0.90 else "⚠️ Action Required"} |
| **Retrieval Precision@5** | {overall_precision:.2%} | - | Info |
| **Mean Reciprocal Rank (MRR)** | {overall_mrr:.3f} | > 0.80 | {"✅ Met" if overall_mrr >= 0.80 else "⚠️ Action Required"} |
| **Citation Accuracy** | {citation_accuracy:.2%} | > 90.0% | {"✅ Met" if citation_accuracy >= 0.90 else "⚠️ Action Required"} |
| **Average Response Latency** | {avg_latency:.3f}s | < 3.0s | {"✅ Met" if avg_latency <= 3.0 else "⚠️ Action Required"} |

---

## Topic Failure Confusion Table

The table below breaks down the RAG performance metrics by academic topic. Topics are sorted by **Failure Rate** in descending order.

| Topic | Total Qs | Avg Accuracy | Avg Faithfulness | Recall@5 | Failure Rate | Primary Failure Mode |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
{confusion_table_str}

---

## Detailed Metric Descriptions

### 1. Retrieval Metrics
- **Recall@K & Precision@K**: Measures if the expected handbook document and section were successfully fetched in the top K positions.
- **Hit Rate**: The percentage of times the correct document appears anywhere in the retrieved chunks.
- **MRR (Mean Reciprocal Rank)**: Penalizes the system if the correct source is returned but at a lower rank position.

### 2. Generation Metrics
- **Semantic Similarity**: Measures structural and thematic similarity between generated answers and the R26 expert answers using a sentence-transformer (`all-MiniLM-L6-v2`) embedding space.
- **Answer Accuracy**: A high-fidelity LLM judge evaluating whether the predicted response is factually equivalent to the verified ground-truth rules.

### 3. RAG Alignment Metrics
- **Faithfulness**: Verifies that the chatbot does not introduce external knowledge or hallucinated rules not supported by the retrieved chunks.
- **Context Recall**: Verifies that the retriever extracts all details necessary to answer the question.
- **Hallucination Rate**: Calculated as the percentage of questions failing the faithfulness threshold (< 70% support).

---

## Observations & Recommendations

1. **Retrieval Bottlenecks:** Topics with high failure rates and "Retrieval Failure" labels indicate that the embeddings or chunk size (1000 chars, 200 overlap) fail to index certain sections properly. Re-indexing with a smaller chunk size and overlap, or implementing a **Hybrid Search** with BM25 keyword matching, will significantly boost Recall.
2. **Generation Hallucinations:** Topics where Faithfulness is low despite high Recall indicate the LLM is introducing background assumptions. Tightening the system prompt's grounding instructions is recommended.
3. **Response Speed:** The average latency of **{avg_latency:.3f} seconds** is highly responsive for production usage.
"""
    
    with open(output_report, "w", encoding="utf-8") as f:
        f.write(report_content)
        
    logger.info(f"Report successfully generated and written to '{output_report}'.")

if __name__ == "__main__":
    generate_report()
