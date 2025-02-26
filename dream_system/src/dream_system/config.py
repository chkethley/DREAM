"""Configuration module for the DREAM system."""

import os
from dataclasses import dataclass
from pathlib import Path

@dataclass
class EnhancedDreamConfig:
    """Configuration class for the DREAM system."""
    
    embedding_model: str = "all-MiniLM-L6-v2"
    generation_model: str = "gpt2"
    embedding_dim: int = 384
    max_new_tokens: int = 100
    temperature: float = 0.7
    top_k: int = 50
    memory_size: int = 1000
    debate_timeout: int = 30
    log_file: str = "dream_journal.json"
    similarity_threshold: float = 0.65
    cache_dir: str = "cache"
    persistence_path: str = "memory.faiss"
    batch_size: int = 32
    max_retries: int = 3
    
    def __post_init__(self):
        """Initialize and validate configuration."""
        self.validate()
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def validate(self):
        """Validate configuration parameters."""
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if self.temperature <= 0 or self.temperature > 1:
            raise ValueError("temperature must be between 0 and 1")
        if self.similarity_threshold <= 0 or self.similarity_threshold > 1:
            raise ValueError("similarity_threshold must be between 0 and 1") 