import os
from pathlib import Path
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    # Milvus
    milvus_host: str = "localhost"
    milvus_port: int = 19530

    # LLM Provider
    llm_provider: str = "openai"  
    llm_api_key: str = ""
    llm_model: str = "qwen3-max"
    llm_base_url: str = ""

    # Embedding Provider
    embedding_provider: str = "openai"  
    embedding_model: str = "text-embedding-v3"
    embedding_dim: int = 1024

    # Vision（阿里云多模态，可选）
    vision_provider: str = "openai"
    vision_model: str = "qwen3-vl-flash"

    # arXiv
    arxiv_max_results: int = 10

    # GitHub
    github_token: str = ""
    github_max_results: int = 10
    github_blacklist: str = ""  # 逗号分隔的仓库黑名单，如: owner/repo1,owner/repo2
    github_rate_limit: int = 5  # 并发速率限制

    # Serper (Web Search)
    serper_api_key: str = ""
    serper_timeout: float = 20.0
    serper_concurrency: int = 5
    serper_max_retries: int = 3

    # Web Visit
    web_visit_timeout: float = 20.0
    web_visit_max_chars: int = 20000

    # File Parse
    file_parse_max_chars: int = 60000
    file_parse_max_rows: int = 200
    file_parse_max_download_size: int = 50 * 1024 * 1024  # 50MB

    # Content Extract
    content_top_paragraphs: int = 12
    content_max_quotes: int = 8

    # Server
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000

    # Data directories
    data_dir: Path = Path("./data")
    upload_dir: Path = Path("./data/uploads")
    reports_dir: Path = Path("./data/reports")

    # DeepResearch defaults
    default_max_rounds: int = 5
    default_top_k: int = 10
    default_max_papers: int = 10
    default_max_repos: int = 5 



    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

    def ensure_directories(self):
        """Ensure all data directories exist"""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    settings = Settings()
    settings.ensure_directories()
    return settings