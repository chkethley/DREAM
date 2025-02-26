"""Tests for the configuration module."""

import pytest
from dream_system.config import EnhancedDreamConfig

def test_default_config():
    """Test default configuration values."""
    config = EnhancedDreamConfig()
    assert config.embedding_model == "all-MiniLM-L6-v2"
    assert config.generation_model == "gpt2"
    assert config.embedding_dim == 384
    assert 0 < config.temperature < 1
    assert config.memory_size > 0

def test_config_validation():
    """Test configuration validation."""
    with pytest.raises(ValueError):
        EnhancedDreamConfig(embedding_dim=-1)
    
    with pytest.raises(ValueError):
        EnhancedDreamConfig(temperature=1.5)
    
    with pytest.raises(ValueError):
        EnhancedDreamConfig(similarity_threshold=2.0)

def test_custom_config():
    """Test custom configuration values."""
    config = EnhancedDreamConfig(
        embedding_model="custom-model",
        generation_model="gpt2-medium",
        embedding_dim=512,
        temperature=0.8,
        memory_size=2000
    )
    
    assert config.embedding_model == "custom-model"
    assert config.generation_model == "gpt2-medium"
    assert config.embedding_dim == 512
    assert config.temperature == 0.8
    assert config.memory_size == 2000 