# Hierarchical Memory System Documentation

## Overview

The Hierarchical Memory System is a key component of the Bangkong LLM Training System that provides enhanced memory capabilities for language models. This system implements a three-tier differentiable memory architecture that allows models to maintain different types of information with appropriate temporal scopes.

## Architecture

The hierarchical memory system consists of three distinct memory tiers:

### 1. Scratchpad Memory (64 slots)
- **Purpose**: Immediate computation context for transient information
- **Use Case**: Short-term working memory for immediate computations
- **Characteristics**: Fast access, high turnover, limited persistence

### 2. Context Memory (128 slots)
- **Purpose**: Session/topic information for mid-term context
- **Use Case**: Maintaining conversation or document context
- **Characteristics**: Medium-term persistence, session-scoped information

### 3. Semantic Memory (256 slots)
- **Purpose**: Persistent knowledge storage for long-term information
- **Use Case**: Long-term knowledge retention and retrieval
- **Characteristics**: High persistence, semantic knowledge storage

## Technical Implementation

The hierarchical memory system is implemented as a PyTorch module that integrates with the transformer architecture. It provides:

1. **Differentiable Memory Modules**: Each memory tier is implemented as a differentiable memory module with key-value storage.

2. **Attention-Based Access**: Memory access is performed through attention mechanisms, allowing for soft attention over memory slots.

3. **Memory Controllers**: Specialized controllers for each memory tier that manage read/write operations.

4. **Memory Integration**: A memory integrator that combines information from all memory tiers with the base hidden states.

## Usage

The hierarchical memory system is automatically integrated with specialized model architectures in Bangkong:

```python
from bangkong.pre_intelligent.memory.hierarchical_memory import HierarchicalMemory

# Initialize memory system
memory = HierarchicalMemory(
    hidden_size=768,
    scratchpad_size=64,
    context_size=128,
    semantic_size=256
)

# Use memory in read mode
enhanced_states = memory(hidden_states, memory_mode="read")

# Use memory in write mode
memory(hidden_states, memory_mode="write")
```

## Memory State Management

The system provides comprehensive memory state management:

```python
# Save memory state
memory_state = memory.get_memory_state()

# Load memory state
memory.load_memory_state(memory_state)
```

## Integration with Training

The hierarchical memory system is designed to work seamlessly with the Bangkong training pipeline:

1. **Automatic Checkpoint Integration**: Memory states are automatically saved and loaded with model checkpoints.

2. **Configuration Support**: Memory system parameters can be configured through the models configuration.

3. **Performance Optimized**: The system is optimized for both training and inference performance.

## Examples

See `examples/hierarchical_memory_demo.py` for a complete demonstration of the hierarchical memory system in action.