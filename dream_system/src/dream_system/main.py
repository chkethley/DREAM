"""Main module for the DREAM system."""

import logging
import numpy as np
from typing import Dict, Any

from .config import EnhancedDreamConfig
from .models import TextEmbedder, ModelCache
from .memory import PersistentMemorySystem
from .debate import EnhancedDebateSystem
from .evolution import EnhancedResponseEvolution
from .journal import EnhancedCognitiveJournal

logger = logging.getLogger(__name__)

class EnhancedDreamSystem:
    """Enhanced DREAM system for dynamic response generation."""
    
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
        """Process a user query and generate a response."""
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