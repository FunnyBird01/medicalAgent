"""Application configuration for the fully local medical assistant."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


@dataclass(slots=True)
class AppConfig:
    """Runtime options. All paths intentionally point to the local workspace."""

    data_dir: Path = DATA_DIR
    db_path: Path = DATA_DIR / "medical_assistant.sqlite3"
    index_path: Path = DATA_DIR / "knowledge_index.json"
    model_name: str = "deepseek-r1:8b"
    ollama_url: str = "http://127.0.0.1:11434"
    use_ollama: bool = True
    temperature: float = 0.1
    top_k: int = 5
    rrf_weight: float = 0.65
    chunk_size: int = 420
    chunk_overlap: int = 60

    @classmethod
    def from_env(cls) -> "AppConfig":
        config = cls()
        config.data_dir = Path(os.getenv("MEDICAL_DATA_DIR", str(config.data_dir)))
        config.db_path = Path(os.getenv("MEDICAL_DB_PATH", str(config.data_dir / "medical_assistant.sqlite3")))
        config.index_path = Path(os.getenv("MEDICAL_INDEX_PATH", str(config.data_dir / "knowledge_index.json")))
        config.model_name = os.getenv("OLLAMA_MODEL", config.model_name)
        config.ollama_url = os.getenv("OLLAMA_URL", config.ollama_url).rstrip("/")
        config.use_ollama = os.getenv("MEDICAL_USE_OLLAMA", "1").lower() in {"1", "true", "yes"}
        config.temperature = float(os.getenv("MEDICAL_TEMPERATURE", config.temperature))
        config.top_k = max(1, int(os.getenv("MEDICAL_TOP_K", config.top_k)))
        config.rrf_weight = min(1.0, max(0.0, float(os.getenv("MEDICAL_RRF_WEIGHT", config.rrf_weight))))
        return config

    def ensure_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
