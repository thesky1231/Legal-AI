import os
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = CURRENT_DIR

DATA_DIR = os.getenv("DATA_DIR", os.path.join(PROJECT_ROOT, "data"))
DB_PATH = os.getenv("DB_PATH", os.path.join(PROJECT_ROOT, "db"))
PDF_PATH = os.getenv("PDF_PATH", os.path.join(DATA_DIR, "law.pdf"))
GOLDEN_DATASET_PATH = os.getenv("GOLDEN_DATASET_PATH", os.path.join(DATA_DIR, "golden_dataset.json"))

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "law_data")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
NORMALIZE_EMBEDDINGS = os.getenv("NORMALIZE_EMBEDDINGS", "true").lower() == "true"

RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base")
RERANK_DEVICE = os.getenv("RERANK_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))
LLM_NUM_CTX = int(os.getenv("LLM_NUM_CTX", "4096"))

VECTOR_RECALL_K = int(os.getenv("VECTOR_RECALL_K", "30"))
BM25_RECALL_K = int(os.getenv("BM25_RECALL_K", "10"))
FINAL_TOP_K = int(os.getenv("FINAL_TOP_K", "3"))

SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("SERVER_PORT", "8000"))

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")