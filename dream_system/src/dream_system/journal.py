"""Journal module for the DREAM system."""

import json
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging

from .config import EnhancedDreamConfig

logger = logging.getLogger(__name__)

class EnhancedCognitiveJournal:
    """Enhanced cognitive journal for logging interactions and generating insights."""
    
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
        """Log an interaction with the system."""
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