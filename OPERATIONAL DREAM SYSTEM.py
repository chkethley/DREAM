from tomlkit import key
import torch
import numpy as np
import asyncio
import json
import hashlib
import logging
import faiss
import os
import aiohttp
import pickle
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union
from sentence_transformers import SentenceTransformer
from transformers import pipeline, PreTrainedModel
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from tenacity import retry, stop_after_attempt, wait_exponential

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dream_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class EnhancedDreamConfig:
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
        return (await self.embed_batch([text]))[0]

class PersistentMemorySystem:
    def __init__(self, config: EnhancedDreamConfig):
        logger.info("Initializing Persistent Memory System")
        self.config = config
        self.index_path = Path(config.persistence_path)
        self.index = self._load_or_create_index()
        self.memory = []
        self.memory_map = {}

    def _load_or_create_index(self) -> faiss.IndexFlatIP:
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
        distances, indices = self.index.search(
            np.expand_dims(query_embedding, 0), top_k)
        return [
            {**self.memory[i], "similarity": float(distances[0][j])}
            for j, i in enumerate(indices[0])
            if i < len(self.memory) and distances[0][j] >= self.config.similarity_threshold
        ]

class AsyncDebateAgent:
    def __init__(self, config: EnhancedDreamConfig):
        self.config = config
        self.generator = pipeline(
            "text-generation",
            model=config.generation_model,
            device=0 if torch.cuda.is_available() else -1,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_k=config.top_k
        )
        self.session = None

    async def setup(self):
        """Initialize aiohttp session for async operations."""
        if self.session is None:
            self.session = aiohttp.ClientSession()

    async def cleanup(self):
        """Cleanup resources."""
        if self.session:
            await self.session.close()
            self.session = None

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def generate_response(self, prompt: str) -> str:
        loop = asyncio.get_event_loop()
        try:
            response = await loop.run_in_executor(
                None, 
                lambda: self.generator(prompt, pad_token_id=50256)[0]['generated_text']
            )
            return response.strip()
        except Exception as e:
            logger.error(f"Generation error: {str(e)}")
            raise

class EnhancedDebateSystem:
    def __init__(self, config: EnhancedDreamConfig):
        logger.info("Initializing Enhanced Debate System")
        self.config = config
        self.agents = {
            "Analyst": AsyncDebateAgent(config),
            "Innovator": AsyncDebateAgent(config),
            "Critic": AsyncDebateAgent(config)
        }
        self.executor = ThreadPoolExecutor(max_workers=3)

    async def setup(self):
        """Initialize all debate agents."""
        await asyncio.gather(*(agent.setup() for agent in self.agents.values()))

    async def cleanup(self):
        """Cleanup all resources."""
        await asyncio.gather(*(agent.cleanup() for agent in self.agents.values()))
        self.executor.shutdown()

    async def conduct_debate(self, prompt: str) -> Dict[str, str]:
        futures = {}
        for role, agent in self.agents.items():
            futures[role] = asyncio.ensure_future(
                agent.generate_response(prompt)
            )
        
        responses = {}
        try:
            done, pending = await asyncio.wait(
                futures.values(),
                timeout=self.config.debate_timeout
            )
            
            for role, future in futures.items():
                if future in done:
                    try:
                        responses[role] = await future
                    except Exception as e:
                        logger.error(f"Error in {role}'s response: {str(e)}")
                        responses[role] = f"Error: {str(e)}"
                else:
                    responses[role] = "Response timed out"
                    future.cancel()
        
        except Exception as e:
            logger.error(f"Debate error: {str(e)}")
            return {role: "System error occurred" for role in self.agents.keys()}
        
        return responses

class EnhancedResponseEvolution:
    def __init__(self, config: EnhancedDreamConfig):
        logger.info("Initializing Enhanced Evolution Engine")
        self.config = config
        self.refiner = pipeline(
            "text2text-generation",
            model="t5-small",
            max_new_tokens=config.max_new_tokens
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def refine_responses(
        self, 
        prompt: str, 
        responses: Dict[str, str],
        context: Optional[List[dict]] = None
    ) -> str:
        debate_context = "\n".join([
            f"{role}:\n{resp}" 
            for role, resp in responses.items()
        ])
        
        context_str = ""
        if context:
            context_str = "\nRelevant context:\n" + "\n".join(
                f"- {m['text']}" for m in context
            )
        
        refinement_prompt = (
            f"Original prompt: {prompt}\n"
            f"Debate perspectives:\n{debate_context}\n"
            f"{context_str}\n"
            "Synthesize a comprehensive response:"
        )
        
        loop = asyncio.get_event_loop()
        try:
            refined = await loop.run_in_executor(
                None,
                lambda: self.refiner(refinement_prompt)[0]['generated_text']
            )
            return refined.strip()
        except Exception as e:
            logger.error(f"Refinement error: {str(e)}")
            raise

class EnhancedCognitiveJournal:
    def __init__(self, config: EnhancedDreamConfig):
        self.config = config
        self.entries = self._load_entries()
        self.analytics = {}

    def _load_entries(self) -> List[dict]:
        """Load existing journal entries."""
        try:
            with open(self.config.log_file, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    async def log_interaction(
        self, 
        prompt: str, 
        response: str, 
        memories: List[dict],
        metadata: Optional[Dict[str, Any]] = None
    ) -> dict:
        entry = {
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt,
            "response": response,
            "context": [m["text"][:50] for m in memories],
            "insights": await self._generate_insights(memories),
            "metadata": metadata or {}
        }
        self.entries.append(entry)
        await self._update_analytics()
        await self._save_to_disk()
        return entry

    async def _generate_insights(self, memories: List[dict]) -> List[str]:
        """Generate insights based on interaction patterns."""
        if not memories:
            return []
        
        insights = []
        texts = [m["text"] for m in memories]
        
        # Pattern detection
        common_words = set.intersection(*[set(t.split()) for t in texts])
        if common_words:
            insights.append(f"Common themes: {', '.join(common_words)}")
        
        # Temporal analysis
        timestamps = [datetime.fromisoformat(m["timestamp"]) for m in memories]
        if len(timestamps) > 1:
            time_diffs = [(t2 - t1).total_seconds() for t1, t2 in zip(timestamps[:-1], timestamps[1:])]
            avg_interval = sum(time_diffs) / len(time_diffs)
            insights.append(f"Average interaction interval: {avg_interval:.2f} seconds")
        
        return insights

    async def _update_analytics(self):
        """Update analytics based on journal entries."""
        self.analytics = {
            "total_interactions": len(self.entries),
            "unique_prompts": len(set(e["prompt"] for e in self.entries)),
            "avg_response_length": sum(len(e["response"]) for e in self.entries) / len(self.entries) if self.entries else 0,
            "last_updated": datetime.now().isoformat()
        }

    async def _save_to_disk(self):
        """Save journal entries and analytics to disk."""
        with open(self.config.log_file, "w") as f:
            json.dump(self.entries, f, indent=2)
        
        with open("journal_analytics.json", "w") as f:
            json.dump(self.analytics, f, indent=2)

class EnhancedDreamSystem:
    def __init__(self, config: EnhancedDreamConfig):
        logger.info("Initializing Enhanced Dream System")
        self.config = config
        self.embedder = TextEmbedder(config)
        self.memory = PersistentMemorySystem(config)
        self.debate = EnhancedDebateSystem(config)
        self.evolution = EnhancedResponseEvolution(config)
        self.journal = EnhancedCognitiveJournal(config)

    async def setup(self):
        """Initialize all subsystems."""
        await self.debate.setup()

    async def cleanup(self):
        """Cleanup all resources."""
        await self.debate.cleanup()
        self.memory.persist()
        ModelCache.get_instance().cleanup()

    async def process_query(self, prompt: str) -> str:
        logger.info(f"Processing query: {prompt}")
        
        try:
            # Embed and retrieve context
            embedding = await self.embedder.embed(prompt)
            context = await self.memory.retrieve(embedding)
            
            # Conduct multi-agent debate
            responses = await self.debate.conduct_debate(prompt)
            
            # Evolve final response with context
            final_response = await self.evolution.refine_responses(
                prompt, responses, context
            )
            
            # Store in memory
            mem_id = await self.memory.add_memory(prompt, embedding, final_response)
            
            # Log interaction with metadata
            metadata = {
                "embedding_norm": float(np.linalg.norm(embedding)),
                "context_size": len(context),
                "response_length": len(final_response)
            }
            await self.journal.log_interaction(prompt, final_response, context, metadata)
            
            return f"{final_response}\n[Memory ID: {mem_id}]"
        
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return f"An error occurred: {str(e)}"

async def main():
    config = EnhancedDreamConfig(
        generation_model="gpt2-medium",
        memory_size=500,
        similarity_threshold=0.7
    )
    
    dream_system = EnhancedDreamSystem(config)
    await dream_system.setup()
    
    try:
        queries = [
            "Explain how neural networks can exhibit creative behavior",
            "What are the philosophical implications of artificial creativity?",
            "How do transformers models differ from biological neural systems?"
        ]
        
        for query in queries:
            print(f"\nUser Query: {query}")
            response = await dream_system.process_query(query)
            print(f"System Response:\n{response}\n")
    
    finally:
        await dream_system.cleanup()

if __name__ == '__main__':
    asyncio.run(main())
