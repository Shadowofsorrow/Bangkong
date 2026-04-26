#!/usr/bin/env python3
"""
Demonstration of Bangkong's Hierarchical Memory System
"""

import sys
import os
from pathlib import Path
import torch

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

def demonstrate_hierarchical_memory():
    """Demonstrate the hierarchical memory system capabilities."""
    print("=== Bangkong Hierarchical Memory System Demonstration ===\n")
    
    try:
        # Import the hierarchical memory system
        from bangkong.pre_intelligent.memory.hierarchical_memory import HierarchicalMemory
        
        # Create hierarchical memory
        memory = HierarchicalMemory(hidden_size=768)
        print("✓ Hierarchical Memory System initialized successfully")
        
        # Create sample input
        batch_size, seq_len, hidden_size = 1, 5, 768
        input_tensor = torch.randn(batch_size, seq_len, hidden_size)
        print(f"✓ Created sample input tensor with shape: {input_tensor.shape}")
        
        # Test reading from memory
        print("\n--- Testing Memory Read Operation ---")
        output_read = memory(input_tensor, memory_mode="read")
        print(f"✓ Read operation completed successfully")
        print(f"  Input shape: {input_tensor.shape}")
        print(f"  Output shape: {output_read.shape}")
        
        # Test writing to memory
        print("\n--- Testing Memory Write Operation ---")
        output_write = memory(input_tensor, memory_mode="write")
        print(f"✓ Write operation completed successfully")
        print(f"  Output shape: {output_write.shape}")
        
        # Test memory state
        print("\n--- Testing Memory State Management ---")
        memory_state = memory.get_memory_state()
        print(f"✓ Memory state retrieved successfully")
        print(f"  Memory state contains {len(memory_state)} components:")
        for key in memory_state.keys():
            print(f"    - {key}")
        
        print("\n=== Hierarchical Memory System Demonstration Completed ===")
        print("System is ready for training with enhanced memory capabilities!")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")

if __name__ == "__main__":
    demonstrate_hierarchical_memory()