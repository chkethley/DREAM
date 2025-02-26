"""Models module for the DREAM system."""

import torch
import numpy as np
from typing import Dict, List
from functools import lru_cache
from sentence_transformers import SentenceTransformer
from transformers import PreTrainedModel
from tenacity import retry, stop_after_attempt, wait_exponential
import logging

from .config import EnhancedDreamConfig

logger = logging.getLogger(__name__)

class ModelCache:
    """Manages model caching and resource cleanup."""
    _instance = None
    _models: Dict[str, PreTrainedModel] = {}

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @lru_cache(maxsize=5)
    def get_model(self, model_name: str) -> PreTrainedModel:
        if model_name not in self._models:
            self._models[model_name] = self._load_model(model_name)
        return self._models[model_name]

    def _load_model(self, model_name: str) -> PreTrainedModel:
        logger.info(f"Loading model: {model_name}")
        return PreTrainedModel.from_pretrained(model_name)

    def cleanup(self):
        """Release model resources."""
        for model in self._models.values():
            del model
        self._models.clear()
        torch.cuda.empty_cache()

class TextEmbedder:
    """Handles text embedding operations."""
    
    def __init__(self, config: EnhancedDreamConfig):
        logger.info("Initializing Embedding Model")
        self.model = SentenceTransformer(config.embedding_model)
        self.config = config
        self.batch_size = config.batch_size

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Batch process embeddings for better performance."""
        try:
            embeddings = self.model.encode(texts, batch_size=self.batch_size)
            return embeddings
        except Exception as e:
            logger.error(f"Embedding error: {str(e)}")
            raise

    async def embed(self, text: str) -> np.ndarray:
        """Embed a single text string."""
        return (await self.embed_batch([text]))[0] 