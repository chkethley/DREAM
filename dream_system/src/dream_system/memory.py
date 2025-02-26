"""Memory module for the DREAM system."""

import faiss
import numpy as np
import hashlib
import pickle
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import logging

from .config import EnhancedDreamConfig

logger = logging.getLogger(__name__)

class PersistentMemorySystem:
    """Manages persistent memory storage and retrieval."""
    
    def __init__(self, config: EnhancedDreamConfig):
        logger.info("Initializing Persistent Memory System")
        self.config = config
        self.index_path = Path(config.persistence_path)
        self.index = self._load_or_create_index()
        self.memory = []
        self.memory_map = {}

    def _load_or_create_index(self) -> faiss.IndexFlatIP:
        """Load existing index or create new one."""
        if self.index_path.exists():
            logger.info("Loading existing FAISS index")
            return faiss.read_index(str(self.index_path))
        logger.info("Creating new FAISS index")
        return faiss.IndexFlatIP(self.config.embedding_dim)

    def persist(self):
        """Save the current state to disk."""
        faiss.write_index(self.index, str(self.index_path))
        with open("memory_map.pkl", "wb") as f:
            pickle.dump(self.memory_map, f)

    async def add_memory(self, text: str, embedding: np.ndarray, response: str) -> str:
        """Add a new memory entry."""
        mem_id = hashlib.sha256(f"{text}{response}".encode()).hexdigest()
        entry = {
            "id": mem_id,
            "text": text,
            "embedding": embedding,
            "response": response,
            "timestamp": datetime.now().isoformat()
        }
        self.memory.append(entry)
        self.memory_map[mem_id] = entry
        self.index.add(np.expand_dims(embedding, 0))
        
        if len(self.memory) > self.config.memory_size:
            await self._prune_memory()
        
        self.persist()
        return mem_id

    async def _prune_memory(self):
        """Smart memory pruning based on relevance and age."""
        timestamps = [datetime.fromisoformat(m["timestamp"]) for m in self.memory]
        age_scores = np.array([(datetime.now() - ts).total_seconds() for ts in timestamps])
        age_scores = age_scores / age_scores.max()  # Normalize

        # Combine age and relevance scores
        final_scores = age_scores
        threshold_idx = len(self.memory) - self.config.memory_size//2
        threshold = np.partition(final_scores, threshold_idx)[threshold_idx]
        
        keep_indices = final_scores <= threshold
        self.memory = [m for m, keep in zip(self.memory, keep_indices) if keep]
        
        self.index.reset()
        embeddings = np.array([m["embedding"] for m in self.memory])
        self.index.add(embeddings)

    async def retrieve(self, query_embedding: np.ndarray, top_k=5) -> List[dict]:
        """Retrieve similar memories."""
        distances, indices = self.index.search(
            np.expand_dims(query_embedding, 0), top_k)
        return [
            {**self.memory[i], "similarity": float(distances[0][j])}
            for j, i in enumerate(indices[0])
            if i < len(self.memory) and distances[0][j] >= self.config.similarity_threshold
        ] 