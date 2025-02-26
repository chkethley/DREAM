"""Evolution module for the DREAM system."""

import asyncio
from typing import Dict, List, Optional
from transformers import pipeline
from tenacity import retry, stop_after_attempt, wait_exponential
import logging

from .config import EnhancedDreamConfig

logger = logging.getLogger(__name__)

class EnhancedResponseEvolution:
    """Enhanced response evolution system for refining responses."""
    
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
        """Refine multiple responses into a coherent answer."""
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