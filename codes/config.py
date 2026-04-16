"""
路径常量 + 环境变量加载
"""
from dotenv import load_dotenv
load_dotenv()
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).parent.parent
WEIGHTS_DIR = BASE_DIR / "weights" / "bge-m3"
DATA_DIR = BASE_DIR / "data"
INDEX_DIR = DATA_DIR / "index"
DATASET_DIR = BASE_DIR / "dataset"

FAISS_INDEX_PATH = INDEX_DIR / "faiss.index"
SPARSE_PATH = INDEX_DIR / "sparse_vectors.pkl"
METADATA_PATH = INDEX_DIR / "metadata.pkl"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    kimi_api_key: str = os.getenv("KIMI_API_KEY")
    kimi_base_url: str = os.getenv("KIMI_BASE_URL")
    kimi_model: str = os.getenv("KIMI_MODEL", "kimi-k2.5")

    top_k: int = 3
    sparse_weight: float = 0.3
    dense_weight: float = 0.7
    high_confidence_threshold: float = 0.92

    host: str = "0.0.0.0"
    port: int = 8000
    session_ttl_hours: int = 24

    enable_timing: bool = True

    baidu_translate_app_id: str = os.getenv("BAIDU_TRANSLATE_APP_ID", "")
    baidu_translate_api_key: str = os.getenv("BAIDU_TRANSLATE_API_KEY", "")


settings = Settings()
