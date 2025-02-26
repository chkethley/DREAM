# DREAM System

A Dynamic Response and Engagement Artificial Mind System that leverages advanced NLP techniques for intelligent conversation and knowledge synthesis.

## Features

- **Multi-Agent Debate System**: Multiple AI agents engage in structured debates to generate comprehensive responses
- **Persistent Memory**: Long-term storage and retrieval of conversation context using FAISS
- **Response Evolution**: Sophisticated response refinement through multi-stage processing
- **Cognitive Journal**: Detailed logging and analytics of system interactions
- **Asynchronous Processing**: Efficient handling of concurrent operations
- **Error Resilience**: Robust error handling and automatic retries
- **Resource Management**: Smart caching and cleanup of model resources

## Installation

### From PyPI

```bash
pip install dream-system
```

### From Source

```bash
git clone https://github.com/craig/dream-system.git
cd dream-system
pip install -e ".[dev]"
```

## Quick Start

```python
import asyncio
from dream_system import EnhancedDreamConfig, EnhancedDreamSystem

async def main():
    # Initialize configuration
    config = EnhancedDreamConfig(
        generation_model="gpt2-medium",
        memory_size=500,
        similarity_threshold=0.7
    )
    
    # Create and setup system
    dream_system = EnhancedDreamSystem(config)
    await dream_system.setup()
    
    try:
        # Process a query
        response = await dream_system.process_query(
            "Explain how neural networks can exhibit creative behavior"
        )
        print(f"Response: {response}")
    
    finally:
        # Cleanup resources
        await dream_system.cleanup()

if __name__ == '__main__':
    asyncio.run(main())
```

## System Architecture

The DREAM system consists of several key components:

1. **Text Embedder**: Converts text into semantic vectors using sentence transformers
2. **Memory System**: Stores and retrieves relevant context using FAISS
3. **Debate System**: Manages multiple AI agents for response generation
4. **Evolution Engine**: Refines and synthesizes responses
5. **Cognitive Journal**: Tracks system interactions and generates insights

## Configuration

The system can be configured through the `EnhancedDreamConfig` class:

```python
config = EnhancedDreamConfig(
    embedding_model="all-MiniLM-L6-v2",  # Model for text embedding
    generation_model="gpt2",             # Model for text generation
    embedding_dim=384,                   # Embedding dimension
    max_new_tokens=100,                  # Max tokens in responses
    temperature=0.7,                     # Generation temperature
    memory_size=1000,                    # Max memories to store
    similarity_threshold=0.65,           # Memory retrieval threshold
)
```

## Development

### Setup Development Environment

```bash
pip install -e ".[dev]"
```

### Running Tests

```bash
pytest tests/
```

### Code Style

The project uses:
- Black for code formatting
- isort for import sorting
- mypy for type checking
- flake8 for linting

Run all checks:

```bash
black .
isort .
mypy src/
flake8
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 