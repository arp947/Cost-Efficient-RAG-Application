from sentence_transformers import SentenceTransformer
from config import settings
from typing import List

class EmbeddingService:
    def __init__(self):
        # Local, free, open-source embedding model running locally on CPU/GPU
        self.model = SentenceTransformer(settings.EMBEDDING_MODEL)
        self.dimension = settings.EMBEDDING_DIM

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()

embedding_service = EmbeddingService()
