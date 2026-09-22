import os


EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)

LLM_MODEL_NAME = os.getenv(
    "LLM_MODEL_NAME",
    "Qwen/Qwen2.5-0.5B-Instruct",
)

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "700"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))

RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "5"))
MMR_FETCH_K = int(os.getenv("MMR_FETCH_K", "12"))
MMR_LAMBDA = float(os.getenv("MMR_LAMBDA", "0.65"))

MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "250"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))
REPETITION_PENALTY = float(os.getenv("REPETITION_PENALTY", "1.12"))
DO_SAMPLE = os.getenv("DO_SAMPLE", "false").lower() == "true"

DEFAULT_LANGUAGE = os.getenv("DEFAULT_LANGUAGE", "ru")