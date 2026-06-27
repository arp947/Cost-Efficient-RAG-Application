import lancedb
import pyarrow as pa
import hashlib
from config import settings
from embeddings import embedding_service
from typing import List, Dict, Any

class VectorDB:
    def __init__(self):
        self.db = lancedb.connect(settings.LANCE_DB_DIR)
        self.schema = pa.schema([
            pa.field("id", pa.string()),          # Deterministic hash of text to guarantee idempotency
            pa.field("vector", pa.list_(pa.float32(), settings.EMBEDDING_DIM)),
            pa.field("text", pa.string()),
            pa.field("doc_id", pa.string()),
            pa.field("file_type", pa.string()),   # Metadata for filtering
        ])
        
    def get_or_create_table(self):
        if settings.TABLE_NAME in self.db.table_names():
            return self.db.open_table(settings.TABLE_NAME)
        return self.db.create_table(settings.TABLE_NAME, schema=self.schema)

    def compute_hash(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def ingest_chunks(self, chunks: List[Dict[str, Any]]):
        table = self.get_or_create_table()
        
        # Prepare batch payloads while enforcing idempotency manually or via primary key overwrite
        data_to_insert = []
        texts = [c["text"] for c in chunks]
        vectors = embedding_service.embed_texts(texts)

        for chunk, vector in zip(chunks, vectors):
            chunk_id = self.compute_hash(chunk["text"])
            
            # Simple read-before-write check to maintain clean dataset idempotency
            # LanceDB also supports merge/upsert patterns natively
            existing = table.search().where(f"id = '{chunk_id}'", prefilter=True).to_list()
            if not existing:
                data_to_insert.append({
                    "id": chunk_id,
                    "vector": vector,
                    "text": chunk["text"],
                    "doc_id": chunk["doc_id"],
                    "file_type": chunk["file_type"]
                })

        if data_to_insert:
            table.add(data_to_insert)
            
    def search(self, query_vector: List[float], k: int, file_type_filter: str = None) -> List[Dict[str, Any]]:
        table = self.get_or_create_table()
        query = table.search(query_vector).limit(k)

        if file_type_filter:
            # Whitelist allowed values to prevent injection via string interpolation
            allowed = {"pdf", "html", "md"}
            if file_type_filter not in allowed:
                raise ValueError(f"Invalid file_type_filter: {file_type_filter}")
            query = query.where(f"file_type = '{file_type_filter}'", prefilter=True)

        return query.to_list()

vector_db = VectorDB()
