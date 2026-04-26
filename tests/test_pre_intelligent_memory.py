#!/usr/bin/env python3
"""
Test script for pre_intelligent memory system in Bangkong LLM Training System
"""

import sys
import os
from pathlib import Path
import torch

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_pre_intelligent_memory():
    """Test the pre_intelligent memory system."""
    print("Testing Pre-Intelligent Memory System...")
    
    # Import the hierarchical memory from pre_intelligent
    try:
        from bangkong.pre_intelligent.memory.hierarchical_memory import HierarchicalMemory
        
        # Initialize hierarchical memory
        memory = HierarchicalMemory()
        
        # Create sample input
        batch_size, seq_len, hidden_size = 2, 10, 768
        input_tensor = torch.randn(batch_size, seq_len, hidden_size)
        
        # Test reading from memory
        memory_output = memory(input_tensor, "read")
        print(f"Input shape: {input_tensor.shape}")
        print(f"Output shape: {memory_output.shape}")
        
        # Test writing to memory
        memory(input_tensor, "write")
        print("Memory write operation completed")
        
        # Test memory state
        memory_state = memory.get_memory_state()
        print(f"Memory state keys: {list(memory_state.keys())}")
        
        print("Pre-intelligent memory test completed successfully!")
    except ImportError as e:
        print(f"Import error: {e}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_pre_intelligent_memory()