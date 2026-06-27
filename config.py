import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    LANCE_DB_DIR: str = "./.lancedb"
    TABLE_NAME: str = "document_chunks"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIM: int = 384
    GROQ_API_KEY: str = "mock-key-for-local-eval"
    LLM_MODEL: str = "llama-3.3-70b-versatile"
    DEFAULT_CHUNK_SIZE: int = 500
    DEFAULT_CHUNK_OVERLAP: int = 50

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

settings = Settings()
