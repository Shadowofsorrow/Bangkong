#!/usr/bin/env python3
"""
Test script for hierarchical memory system in Bangkong LLM Training System
"""

import sys
import os
from pathlib import Path
import torch

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_hierarchical_memory():
    """Test the hierarchical memory system."""
    print("Testing Hierarchical Memory System...")
    
    try:
        # Import the hierarchical memory system
        from bangkong.pre_intelligent.memory.hierarchical_memory import HierarchicalMemory
        import torch
        
        # Initialize hierarchical memory
        memory = HierarchicalMemory()
        
        # Create sample input
        batch_size, seq_len, hidden_size = 2, 10, 768
        input_tensor = torch.randn(batch_size, seq_len, hidden_size)
        
        # Test reading from memory
        memory_output = memory(input_tensor, memory_mode="read")
        print(f"Input shape: {input_tensor.shape}")
        print(f"Output shape: {memory_output.shape}")
        
        # Test writing to memory
        memory(input_tensor, memory_mode="write")
        print("Memory write operation completed")
        
        # Test memory state
        memory_state = memory.get_memory_state()
        print(f"Memory state keys: {list(memory_state.keys())}")
        
        print("Hierarchical memory test completed successfully!")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_hierarchical_memory()