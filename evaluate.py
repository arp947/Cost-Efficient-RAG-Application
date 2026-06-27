import numpy as np
import pandas as pd
from embeddings import embedding_service
from database import vector_db
from llm_service import llm_service

EVAL_DATASET = [
    {"query": "What is the primary revenue model?",       "gold_doc_id": "doc_1", "expected_keywords": ["subscription", "recurring"]},
    {"query": "How are system errors handled?",           "gold_doc_id": "doc_2", "expected_keywords": ["idempotency", "retry"]},
    {"query": "What is the maximum token limitation?",    "gold_doc_id": "doc_1", "expected_keywords": ["8192", "tokens"]},
    {"query": "Where is user asset log saved?",           "gold_doc_id": "doc_3", "expected_keywords": ["s3 bucket", "storage"]},
    {"query": "Who is responsible for firewall rule changes?", "gold_doc_id": "doc_4", "expected_keywords": ["secops", "security team"]},
    {"query": "What database holds the state vectors?",   "gold_doc_id": "doc_2", "expected_keywords": ["lancedb", "embedded"]},
    {"query": "How long is cold storage retention?",      "gold_doc_id": "doc_5", "expected_keywords": ["7 years", "retention"]},
    {"query": "What deployment method is enforced?",      "gold_doc_id": "doc_3", "expected_keywords": ["helm chart", "kubernetes"]},
    {"query": "How frequent are automated snapshots?",    "gold_doc_id": "doc_4", "expected_keywords": ["24 hours", "daily"]},
    {"query": "Which compression algorithm is used?",     "gold_doc_id": "doc_1", "expected_keywords": ["zstd", "parquet"]},
    {"query": "What triggers an auto-scaling group?",     "gold_doc_id": "doc_5", "expected_keywords": ["cpu target", "75%"]},
    {"query": "Who signs off external API audits?",       "gold_doc_id": "doc_2", "expected_keywords": ["cto", "compliance"]},
    {"query": "What encryption standard is applied at rest?", "gold_doc_id": "doc_3", "expected_keywords": ["aes-256"]},
    {"query": "How are broken vector states synchronized?","gold_doc_id": "doc_4", "expected_keywords": ["re-indexing", "cron"]},
    {"query": "What protocol handles external routing?",  "gold_doc_id": "doc_5", "expected_keywords": ["bgp", "gateway"]},
]

MOCK_CORPUS = [
    {"text": "The primary revenue model uses recurring subscription systems. Token limit is capped at 8192 tokens. Data is stored using zstd compression inside Parquet files.", "doc_id": "doc_1", "file_type": "md"},
    {"text": "System errors are handled through strict idempotency routines with automatic retry logic. State vectors are stored in LanceDB embedded store. External API audits are signed off by the CTO compliance team.", "doc_id": "doc_2", "file_type": "pdf"},
    {"text": "User asset logs are permanently saved inside an encrypted S3 bucket storage system. Deployment is enforced via Helm Chart on Kubernetes. All data at rest uses AES-256 encryption.", "doc_id": "doc_3", "file_type": "html"},
    {"text": "The SecOps security team controls all firewall rule changes. Automated snapshots run every 24 hours on a daily schedule. Broken vector states are resynchronized via re-indexing cron jobs.", "doc_id": "doc_4", "file_type": "pdf"},
    {"text": "Cold storage enforces a 7 years retention policy. Auto-scaling groups are triggered at a CPU target of 75%. External routing is handled by BGP at the gateway layer.", "doc_id": "doc_5", "file_type": "md"},
]


def _judge_faithfulness(answer: str, context_text: str) -> float:
    """LLM-as-judge: returns 1.0 if answer is grounded, 0.0 if not."""
    prompt = (
        f"Context: {context_text}\n\n"
        f"Answer: {answer}\n\n"
        "Is the answer fully supported by the context above? Reply only 'yes' or 'no'."
    )
    try:
        resp, _ = llm_service.generate_answer("", [{"doc_id": "judge", "text": prompt}])
        return 1.0 if "yes" in resp.strip().lower() else 0.0
    except Exception:
        return 0.0


def run_evaluation():
    hits, rr_scores, ndcg_scores, faithfulness_scores = 0, [], [], []

    print("Running retrieval + faithfulness evaluation...")

    for item in EVAL_DATASET:
        q_vec = embedding_service.embed_texts([item["query"]])[0]
        results = vector_db.search(q_vec, k=3)
        retrieved_docs = [r["doc_id"] for r in results]
        context_text = " ".join([r["text"] for r in results])

        # Recall@3 / Hit Rate
        if item["gold_doc_id"] in retrieved_docs:
            hits += 1

        # MRR + nDCG@3
        try:
            rank = retrieved_docs.index(item["gold_doc_id"]) + 1
            rr_scores.append(1.0 / rank)
            ndcg_scores.append(1.0 / np.log2(rank + 1))
        except ValueError:
            rr_scores.append(0.0)
            ndcg_scores.append(0.0)

        # LLM-as-judge faithfulness
        answer, _ = llm_service.generate_answer(item["query"], results)
        faithfulness_scores.append(_judge_faithfulness(answer, context_text))

    total = len(EVAL_DATASET)
    metrics = {
        "Recall@3_Hit_Rate": hits / total,
        "MRR": np.mean(rr_scores),
        "nDCG@3": np.mean(ndcg_scores),
        "Faithfulness_LLM_Judge": np.mean(faithfulness_scores),
    }

    df = pd.DataFrame([metrics])
    df.to_csv("evaluation_results.csv", index=False)
    print("Done. Results saved to evaluation_results.csv")
    print(df.to_string())


if __name__ == "__main__":
    vector_db.ingest_chunks(MOCK_CORPUS)
    run_evaluation()
