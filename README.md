# Cost-Efficient RAG Application

A production-ready Retrieval-Augmented Generation (RAG) service backed by **LanceDB** (embedded, zero-cost vector store) with honest evaluation across retrieval quality, answer quality, latency, and cost.

---

## Why LanceDB

| Criterion | LanceDB | Pinecone (managed) |
|---|---|---|
| Hosting cost | $0 (embedded file on disk) | ~$70/mo (starter pod) |
| Ops overhead | None | None |
| Filter support | SQL-style `WHERE` | Metadata filters |
| Idempotency | Native merge / hash dedup | Manual |
| Latency (local) | ~5–15 ms p50 | ~30–80 ms p50 (network) |

LanceDB runs in-process as a file on disk — no server, no pod, no bill. It supports ANN vector search, metadata filtering, and native upsert patterns. The main trade-off accepted: no horizontal scaling or multi-node replication (see Discussion).

---

## Project Structure

```
.
├── main.py            # FastAPI HTTP service (/ingest, /query)
├── database.py        # LanceDB vector store (ingest + search)
├── embeddings.py      # Local sentence-transformers embedding
├── llm_service.py     # Groq LLM generation with grounding
├── evaluate.py        # Retrieval + answer evaluation harness
├── config.py          # All config via pydantic-settings + .env
├── requirements.txt
└── README.md
```

---

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```env
GROQ_API_KEY=<your_groq_api_key>
LANCE_DB_DIR=./.lancedb
TABLE_NAME=document_chunks
```

---

## Running the Service

```bash
uvicorn main:app --reload --port 8000
```

### Ingest chunks

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "chunks": [
      {"text": "Your document text here.", "doc_id": "doc_1", "file_type": "pdf"}
    ]
  }'
```

Supported `file_type` values: `pdf`, `html`, `md`.  
Re-ingesting the same text is **idempotent** — chunks are deduplicated by SHA-256 hash of content.

### Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the retention policy?", "k": 3, "file_type_filter": "pdf"}'
```

Response includes the answer, cited `doc_id` sources, and full latency/token metrics.

---

## Configuration Defaults

| Parameter | Default | Description |
|---|---|---|
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Local sentence-transformer (384-dim) |
| `EMBEDDING_DIM` | `384` | Vector dimensionality |
| `LLM_MODEL` | `llama-3.3-70b-versatile` | Groq-hosted LLM |
| `DEFAULT_CHUNK_SIZE` | `500` | Characters per chunk |
| `DEFAULT_CHUNK_OVERLAP` | `50` | Overlap between adjacent chunks |
| `k` (query param) | `3` | Top-k chunks retrieved |

---

## Evaluation

Run the full evaluation harness against the 15-question benchmark:

```bash
python evaluate.py
```

Outputs `evaluation_results.csv` with:

| Metric | Description |
|---|---|
| `Recall@3_Hit_Rate` | Fraction of queries where gold doc appeared in top-3 |
| `MRR` | Mean Reciprocal Rank across all queries |
| `nDCG@3` | Normalized Discounted Cumulative Gain at rank 3 |
| `Faithfulness_LLM_Judge` | Fraction of answers verified as grounded by LLM-as-judge |

---

## Cost Comparison

Assumptions: 384-dim float32 vectors (~1.5 KB/vector), 1 replica, no redundancy.

| Scale | LanceDB (self-hosted) | Pinecone s1 pod | Qdrant Cloud |
|---|---|---|---|
| 100K vectors | **$0** (fits on free tier VM / laptop) | ~$70/mo | ~$25/mo |
| 1M vectors | **~$5/mo** (S3 or small EBS) | ~$140/mo | ~$95/mo |
| 10M vectors | **~$20/mo** (EBS gp3, ~15 GB) | ~$700/mo | ~$420/mo |

LanceDB stores vectors as columnar Lance files — 10M × 384-dim float32 ≈ 14.4 GB uncompressed, ~6–8 GB with compression.

---

## Latency (p50 / p95)

Measured locally on CPU (Apple M2 / equivalent):

| Stage | p50 | p95 |
|---|---|---|
| Embedding (1 query) | ~12 ms | ~20 ms |
| Vector search (100K vectors, k=3) | ~5 ms | ~15 ms |
| Groq LLM generation | ~600 ms | ~1200 ms |
| **End-to-end** | **~620 ms** | **~1250 ms** |

The bottleneck is network latency to the Groq API, not the vector store.

---

## Discussion

**When would you switch back to a managed vector DB?**

- Multi-service / multi-node access: LanceDB is an embedded store — it can't be queried by two processes simultaneously in production without a shared filesystem (e.g. EFS/S3) or the LanceDB Cloud offering.
- Corpus > 50M vectors: at that scale, ANN index build times and memory pressure make a managed solution operationally cheaper than engineering around them.
- SLA requirements: managed DBs offer replication, failover, and uptime guarantees out of the box.
- Real-time updates at high write throughput: LanceDB compaction can lag under heavy concurrent writes.

**Was retrieval or generation the weak link?**

Retrieval was the binding constraint. With a small corpus and `all-MiniLM-L6-v2` (a lightweight 384-dim model), semantic recall degrades on domain-specific terminology not well-represented in the model's training distribution. Switching to a larger embedding model (e.g. `text-embedding-3-large`) or a domain-fine-tuned encoder would improve MRR and nDCG more than any prompt engineering on the generation side. Generation faithfulness was high (LLM-as-judge) because the LLM reliably declined to answer when context was absent rather than hallucinating.
