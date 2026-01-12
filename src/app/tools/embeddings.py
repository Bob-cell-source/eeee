"""
Embedding Tools - 文本嵌入
可选供应商，可自行添加: openai, ollama
"""
import logging
import hashlib
from typing import List, Optional
from abc import ABC, abstractmethod
import numpy as np
import httpx

from ..config import get_settings

logger = logging.getLogger(__name__)


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers"""
    
    @abstractmethod
    async def embed(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        pass
    
    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        pass


class DummyEmbeddingProvider(EmbeddingProvider):
    """
    测试使用
    """
    
    def __init__(self, dim: int = 1536):
        self.dim = dim
    
    async def embed(self, text: str) -> List[float]:
        """Generate a deterministic embedding based on text hash"""
        # Create deterministic embedding from text hash
        hash_bytes = hashlib.sha256(text.encode()).digest()
        # Use hash to seed numpy random for reproducibility
        seed = int.from_bytes(hash_bytes[:4], 'big')
        rng = np.random.RandomState(seed)
        embedding = rng.randn(self.dim).astype(np.float32)
        # Normalize to unit vector
        embedding = embedding / np.linalg.norm(embedding)
        return embedding.tolist()
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        return [await self.embed(text) for text in texts]
    
class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embedding provider"""
    
    def __init__(self, api_key: str, model: str = "text-embedding-3-small", base_url: Optional[str] = None, dimensions: Optional[int] = None):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url or "https://api.openai.com/v1"
        self.dimensions = dimensions
    
    async def embed(self, text: str) -> List[float]:
        """Generate embedding using OpenAI API"""
        embeddings = await self.embed_batch([text])
        return embeddings[0] if embeddings else []
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI API"""
        try:
            async with httpx.AsyncClient() as client:
                # Build request payload
                payload = {
                    "model": self.model,
                    "input": texts,
                }
                # Add dimensions parameter if specified (required for some models like阿里云 text-embedding-v3)
                if self.dimensions:
                    payload["dimensions"] = self.dimensions
                
                response = await client.post(
                    f"{self.base_url}/embeddings",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=60.0,
                )
                response.raise_for_status()
                data = response.json()
                
                # Sort by index to ensure correct order
                embeddings_data = sorted(data["data"], key=lambda x: x["index"])
                return [item["embedding"] for item in embeddings_data]
        except httpx.HTTPStatusError as e:
            logger.error(f"OpenAI embedding HTTP error: {e}, Response: {e.response.text}")
            # Fallback to dummy embeddings with correct dimensions
            dummy = DummyEmbeddingProvider(dim=self.dimensions if self.dimensions else 1536)
            return await dummy.embed_batch(texts)
        except Exception as e:
            logger.error(f"OpenAI embedding error: {e}")
            # Fallback to dummy embeddings with correct dimensions
            dummy = DummyEmbeddingProvider(dim=self.dimensions if self.dimensions else 1536)
            return await dummy.embed_batch(texts)


class OllamaEmbeddingProvider(EmbeddingProvider):
    """Ollama embedding provider"""
    
    def __init__(self, model: str = "nomic-embed-text", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
    
    async def embed(self, text: str) -> List[float]:
        """Generate embedding using Ollama API"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/api/embeddings",
                    json={
                        "model": self.model,
                        "prompt": text,
                    },
                    timeout=60.0,
                )
                response.raise_for_status()
                data = response.json()
                return data["embedding"]
        except Exception as e:
            logger.error(f"Ollama embedding error: {e}")
            # Get correct dimension from settings
            settings = get_settings()
            dummy = DummyEmbeddingProvider(dim=settings.embedding_dim)
            return await dummy.embed(text)
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        return [await self.embed(text) for text in texts]


def get_embedding_provider() -> EmbeddingProvider:
    """Get the configured embedding provider"""
    settings = get_settings()
    provider = settings.embedding_provider.lower()
    
    if provider == "openai":
        return OpenAIEmbeddingProvider(
            api_key=settings.llm_api_key,
            model=settings.embedding_model,
            base_url=settings.llm_base_url if settings.llm_base_url else None,
            dimensions=settings.embedding_dim,  # Pass dimensions for models that support it
        )
    elif provider == "ollama":
        return OllamaEmbeddingProvider(
            model=settings.embedding_model,
            base_url=settings.llm_base_url if settings.llm_base_url else "http://localhost:11434",
        )
    else:
        # Default to dummy provider
        return DummyEmbeddingProvider(dim=settings.embedding_dim)
