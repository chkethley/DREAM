"""Debate module for the DREAM system."""

import asyncio
import aiohttp
import torch
from typing import Dict, Any
from transformers import pipeline
from tenacity import retry, stop_after_attempt, wait_exponential
from concurrent.futures import ThreadPoolExecutor
import logging

from .config import EnhancedDreamConfig

logger = logging.getLogger(__name__)

class AsyncDebateAgent:
    """Asynchronous debate agent for generating responses."""
    
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
        """Generate a response to the given prompt."""
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
    """Enhanced debate system managing multiple debate agents."""
    
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
        """Conduct a debate among agents on the given prompt."""
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