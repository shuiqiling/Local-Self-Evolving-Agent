from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path(os.environ.get("LEA_DATA_DIR", "./data"))
    model_url: str = os.environ.get("LEA_MODEL_URL", "http://127.0.0.1:18801")
    precision: str = os.environ.get("LEA_PRECISION", "BF16")
    model_name: str = os.environ.get("LEA_MODEL_NAME", "Qwen3-4B-Instruct-2507-" + os.environ.get("LEA_PRECISION", "BF16"))
    working_tokens: int = int(os.environ.get("LEA_WORKING_TOKENS", "3072"))
    backend_context: int = 8192
    handoff_reserve: int = 1024
    max_steps: int = 10
    max_readers: int = 4
    max_memory_queries: int = int(os.environ.get("LEA_MEMORY_QUERY_BUDGET", "1"))
    call_timeout: int = 480
    query_timeout: int = 1200
    model_status_file: str = os.environ.get("LEA_MODEL_STATUS_FILE", "")
    primary_adapter_id: int = int(os.environ.get('LEA_PRIMARY_ADAPTER_ID', '-1'))
    primary_adapter_version: str = os.environ.get('LEA_PRIMARY_ADAPTER_VERSION', '')
    memory_backend: str = os.environ.get('LEA_MEMORY_BACKEND', 'hybrid')
    embedding_url: str = os.environ.get('LEA_EMBEDDING_URL', 'http://127.0.0.1:18803')
    embedding_model: str = 'Qwen3-Embedding-0.6B'
    embedding_fingerprint: str = os.environ.get('LEA_EMBEDDING_FINGERPRINT', 'Qwen3-Embedding-0.6B-f16')
    retrieval_top_k: int = 6
    retrieval_max_chunks: int = 20000
    retrieval_min_cosine: float = float(os.environ.get('LEA_RETRIEVAL_MIN_COSINE', '.65'))
    answer_max_chars: int = int(os.environ.get('LEA_ANSWER_MAX_CHARS', '200'))
    primary_output_tokens: int = int(os.environ.get('LEA_PRIMARY_OUTPUT_TOKENS', '384'))
    memory_read_chars: int = 400
    window_prompt_layout: str = os.environ.get('LEA_WINDOW_PROMPT_LAYOUT', 'legacy')
    primary_cache_slot: int = int(os.environ.get('LEA_PRIMARY_CACHE_SLOT', '-1'))
    auxiliary_cache_slot: int = int(os.environ.get('LEA_AUXILIARY_CACHE_SLOT', '-1'))
    prompt_cache: bool = os.environ.get('LEA_PROMPT_CACHE', '1') == '1'

