import time
import uuid
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional
from database import vector_db
from embeddings import embedding_service
from llm_service import llm_service

app = FastAPI(title="Cost-Efficient RAG System API")

class ChunkInput(BaseModel):
    text: str
    doc_id: str
    file_type: str

class IngestionPayload(BaseModel):
    chunks: List[ChunkInput]

class QueryRequest(BaseModel):
    query: str
    k: int = 3
    file_type_filter: Optional[str] = None

@app.post("/ingest")
def ingest_documents(payload: IngestionPayload):
    try:
        chunks_dict = [c.model_dump() for c in payload.chunks]
        vector_db.ingest_chunks(chunks_dict)
        return {"status": "success", "processed_chunks": len(payload.chunks)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
def query_rag(request: QueryRequest):
    overall_start = time.time()
    
    # 1. Embed query
    vec_start = time.time()
    query_vector = embedding_service.embed_texts([request.query])[0]
    vec_latency = (time.time() - vec_start) * 1000
    
    # 2. Retrieve contexts
    retrieval_start = time.time()
    contexts = vector_db.search(query_vector, k=request.k, file_type_filter=request.file_type_filter)
    retrieval_latency = (time.time() - retrieval_start) * 1000
    
    # 3. Grounded generation
    answer, llm_metrics = llm_service.generate_answer(request.query, contexts)
    
    total_latency = (time.time() - overall_start) * 1000
    
    # System execution logging per query
    print(f"[METRICS] Total Latency: {total_latency:.2f}ms | Vector Store Latency: {retrieval_latency:.2f}ms | Chunks Returned: {len(contexts)} | Prompt Tokens: {llm_metrics['prompt_tokens']}")
    
    return {
        "answer": answer,
        "citations": [c["doc_id"] for c in contexts],
        "metrics": {
            "total_latency_ms": total_latency,
            "vector_search_latency_ms": retrieval_latency,
            "embedding_latency_ms": vec_latency,
            "llm_generation_latency_ms": llm_metrics["latency_ms"],
            "chunk_count": len(contexts),
            "token_usage": llm_metrics
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
