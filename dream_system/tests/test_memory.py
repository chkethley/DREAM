"""Tests for the memory system module."""

import pytest
import numpy as np
from pathlib import Path
import asyncio

from dream_system.config import EnhancedDreamConfig
from dream_system.memory import PersistentMemorySystem

@pytest.fixture
def config():
    """Create a test configuration."""
    return EnhancedDreamConfig(
        embedding_dim=64,
        memory_size=10,
        similarity_threshold=0.5
    )

@pytest.fixture
def memory_system(config, tmp_path):
    """Create a test memory system."""
    config.persistence_path = str(tmp_path / "test_memory.faiss")
    return PersistentMemorySystem(config)

@pytest.mark.asyncio
async def test_memory_add_and_retrieve(memory_system):
    """Test adding and retrieving memories."""
    text = "test query"
    embedding = np.random.rand(64).astype(np.float32)
    response = "test response"
    
    # Add memory
    mem_id = await memory_system.add_memory(text, embedding, response)
    assert mem_id is not None
    
    # Retrieve memory
    results = await memory_system.retrieve(embedding)
    assert len(results) > 0
    assert results[0]["text"] == text
    assert results[0]["response"] == response

@pytest.mark.asyncio
async def test_memory_pruning(memory_system):
    """Test memory pruning mechanism."""
    # Add more memories than the size limit
    for i in range(15):
        text = f"test query {i}"
        embedding = np.random.rand(64).astype(np.float32)
        response = f"test response {i}"
        await memory_system.add_memory(text, embedding, response)
    
    # Check if memory size is maintained
    assert len(memory_system.memory) <= memory_system.config.memory_size

@pytest.mark.asyncio
async def test_memory_persistence(config, tmp_path):
    """Test memory persistence across instances."""
    config.persistence_path = str(tmp_path / "test_memory.faiss")
    
    # Create first instance and add memory
    memory1 = PersistentMemorySystem(config)
    text = "test query"
    embedding = np.random.rand(64).astype(np.float32)
    response = "test response"
    await memory1.add_memory(text, embedding, response)
    memory1.persist()
    
    # Create second instance and verify memory
    memory2 = PersistentMemorySystem(config)
    results = await memory2.retrieve(embedding)
    assert len(results) > 0
    assert results[0]["text"] == text
    assert results[0]["response"] == response 