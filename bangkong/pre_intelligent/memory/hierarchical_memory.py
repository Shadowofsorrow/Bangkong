#!/usr/bin/env python3
"""
Hierarchical differentiable memory system for pre-intelligent LLMs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Optional
import math

class DifferentiableMemory(nn.Module):
    """Base class for differentiable memory modules."""
    
    def __init__(self, memory_size: int, key_dim: int, value_dim: int):
        """
        Initialize differentiable memory.
        
        Args:
            memory_size: Number of memory slots
            key_dim: Dimension of memory keys
            value_dim: Dimension of memory values
        """
        super(DifferentiableMemory, self).__init__()
        self.memory_size = memory_size
        self.key_dim = key_dim
        self.value_dim = value_dim
        
        # Memory banks
        self.keys = nn.Parameter(torch.randn(memory_size, key_dim))
        self.values = nn.Parameter(torch.randn(memory_size, value_dim))
        
        # Initialize memory
        nn.init.xavier_uniform_(self.keys)
        nn.init.xavier_uniform_(self.values)
    
    def read(self, query: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Read from memory using attention mechanism.
        
        Args:
            query: Query vector (batch_size, key_dim)
            
        Returns:
            Tuple of (read_values, attention_weights)
        """
        # Compute attention scores
        scores = torch.matmul(query, self.keys.t())  # (batch_size, memory_size)
        attention = F.softmax(scores, dim=-1)  # (batch_size, memory_size)
        
        # Read values
        read_values = torch.matmul(attention, self.values)  # (batch_size, value_dim)
        
        return read_values, attention
    
    def write(self, query: torch.Tensor, value: torch.Tensor, erase: bool = False):
        """
        Write to memory using attention mechanism.
        
        Args:
            query: Query vector (batch_size, key_dim)
            value: Value to write (batch_size, value_dim)
            erase: Whether to erase instead of write
        """
        # Compute attention scores
        scores = torch.matmul(query, self.keys.t())  # (batch_size, memory_size)
        attention = F.softmax(scores, dim=-1)  # (batch_size, memory_size)
        
        # Update values
        if erase:
            # Erase values
            erase_vector = torch.ones_like(value) - value
            update = torch.matmul(attention.t(), erase_vector)
            self.values.data = self.values.data * (1 - update.unsqueeze(0))
        else:
            # Write values
            update = torch.matmul(attention.t(), value)
            self.values.data = self.values.data + update.unsqueeze(0)

class ScratchpadMemory(DifferentiableMemory):
    """Short-term scratchpad memory for immediate computations."""
    
    def __init__(self, memory_size: int = 64, key_dim: int = 128, value_dim: int = 128):
        """
        Initialize scratchpad memory.
        
        Args:
            memory_size: Number of memory slots
            key_dim: Dimension of memory keys
            value_dim: Dimension of memory values
        """
        super(ScratchpadMemory, self).__init__(memory_size, key_dim, value_dim)

class ContextMemory(DifferentiableMemory):
    """Mid-term context memory for session/topic information."""
    
    def __init__(self, memory_size: int = 128, key_dim: int = 256, value_dim: int = 256):
        """
        Initialize context memory.
        
        Args:
            memory_size: Number of memory slots
            key_dim: Dimension of memory keys
            value_dim: Dimension of memory values
        """
        super(ContextMemory, self).__init__(memory_size, key_dim, value_dim)

class SemanticMemory(DifferentiableMemory):
    """Long-term semantic memory for persistent knowledge."""
    
    def __init__(self, memory_size: int = 256, key_dim: int = 512, value_dim: int = 512):
        """
        Initialize semantic memory.
        
        Args:
            memory_size: Number of memory slots
            key_dim: Dimension of memory keys
            value_dim: Dimension of memory values
        """
        super(SemanticMemory, self).__init__(memory_size, key_dim, value_dim)

class HierarchicalMemory(nn.Module):
    """Hierarchical memory system combining different memory types."""
    
    def __init__(self, 
                 hidden_size: int = 768,
                 scratchpad_size: int = 64,
                 context_size: int = 128,
                 semantic_size: int = 256):
        """
        Initialize hierarchical memory system.
        
        Args:
            hidden_size: Size of hidden representations
            scratchpad_size: Size of scratchpad memory
            context_size: Size of context memory
            semantic_size: Size of semantic memory
        """
        super(HierarchicalMemory, self).__init__()
        
        self.hidden_size = hidden_size
        
        # Memory modules
        self.scratchpad = ScratchpadMemory(
            memory_size=scratchpad_size,
            key_dim=hidden_size,
            value_dim=hidden_size
        )
        
        self.context = ContextMemory(
            memory_size=context_size,
            key_dim=hidden_size,
            value_dim=hidden_size
        )
        
        self.semantic = SemanticMemory(
            memory_size=semantic_size,
            key_dim=hidden_size,
            value_dim=hidden_size
        )
        
        # Memory controllers
        self.scratchpad_controller = nn.Linear(hidden_size, hidden_size)
        self.context_controller = nn.Linear(hidden_size, hidden_size)
        self.semantic_controller = nn.Linear(hidden_size, hidden_size)
        
        # Memory integrator
        self.memory_integrator = nn.Linear(hidden_size * 4, hidden_size)
        
        # Initialize controllers
        self._init_controllers()
    
    def _init_controllers(self):
        """Initialize memory controllers with appropriate weights."""
        nn.init.xavier_uniform_(self.scratchpad_controller.weight)
        nn.init.xavier_uniform_(self.context_controller.weight)
        nn.init.xavier_uniform_(self.semantic_controller.weight)
        nn.init.xavier_uniform_(self.memory_integrator.weight)
        
        nn.init.zeros_(self.scratchpad_controller.bias)
        nn.init.zeros_(self.context_controller.bias)
        nn.init.zeros_(self.semantic_controller.bias)
        nn.init.zeros_(self.memory_integrator.bias)
    
    def forward(self, 
                hidden_states: torch.Tensor,
                memory_mode: str = "read") -> torch.Tensor:
        """
        Process through hierarchical memory system.
        
        Args:
            hidden_states: Input hidden states (batch_size, seq_len, hidden_size)
            memory_mode: Whether to read from or write to memory
            
        Returns:
            Enhanced hidden states with memory information
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Reshape for processing
        hidden_flat = hidden_states.view(-1, hidden_size)  # (batch_size * seq_len, hidden_size)
        
        if memory_mode == "read":
            # Read from all memory levels
            scratchpad_read, _ = self.scratchpad.read(
                self.scratchpad_controller(hidden_flat)
            )
            
            context_read, _ = self.context.read(
                self.context_controller(hidden_flat)
            )
            
            semantic_read, _ = self.semantic.read(
                self.semantic_controller(hidden_flat)
            )
            
            # Integrate memory information
            memory_enhanced = self.memory_integrator(
                torch.cat([hidden_flat, scratchpad_read, context_read, semantic_read], dim=-1)
            )
            
        elif memory_mode == "write":
            # Write to all memory levels (simplified for demonstration)
            # In practice, you'd have more sophisticated write mechanisms
            self.scratchpad.write(
                self.scratchpad_controller(hidden_flat),
                hidden_flat
            )
            
            self.context.write(
                self.context_controller(hidden_flat),
                hidden_flat
            )
            
            self.semantic.write(
                self.semantic_controller(hidden_flat),
                hidden_flat
            )
            
            # Return original hidden states for write operations
            memory_enhanced = hidden_flat
        
        else:
            raise ValueError(f"Unknown memory mode: {memory_mode}")
        
        # Reshape back to original dimensions
        memory_enhanced = memory_enhanced.view(batch_size, seq_len, hidden_size)
        
        return memory_enhanced
    
    def clear_scratchpad(self):
        """Clear scratchpad memory."""
        with torch.no_grad():
            self.scratchpad.keys.zero_()
            self.scratchpad.values.zero_()
            nn.init.xavier_uniform_(self.scratchpad.keys)
            nn.init.xavier_uniform_(self.scratchpad.values)
    
    def get_memory_state(self) -> Dict[str, torch.Tensor]:
        """
        Get current state of all memory modules.
        
        Returns:
            Dictionary of memory states
        """
        return {
            'scratchpad_keys': self.scratchpad.keys.detach().cpu(),
            'scratchpad_values': self.scratchpad.values.detach().cpu(),
            'context_keys': self.context.keys.detach().cpu(),
            'context_values': self.context.values.detach().cpu(),
            'semantic_keys': self.semantic.keys.detach().cpu(),
            'semantic_values': self.semantic.values.detach().cpu()
        }
    
    def load_memory_state(self, state_dict: Dict[str, torch.Tensor]):
        """
        Load memory state from dictionary.
        
        Args:
            state_dict: Dictionary of memory states
        """
        with torch.no_grad():
            self.scratchpad.keys.copy_(state_dict['scratchpad_keys'])
            self.scratchpad.values.copy_(state_dict['scratchpad_values'])
            self.context.keys.copy_(state_dict['context_keys'])
            self.context.values.copy_(state_dict['context_values'])
            self.semantic.keys.copy_(state_dict['semantic_keys'])
            self.semantic.values.copy_(state_dict['semantic_values'])

class MemoryEnhancedTransformerLayer(nn.Module):
    """Transformer layer enhanced with hierarchical memory."""
    
    def __init__(self, hidden_size: int = 768, num_heads: int = 12):
        """
        Initialize memory-enhanced transformer layer.
        
        Args:
            hidden_size: Size of hidden representations
            num_heads: Number of attention heads
        """
        super(MemoryEnhancedTransformerLayer, self).__init__()
        
        # Standard transformer components
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True
        )
        
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        
        # Hierarchical memory system
        self.memory = HierarchicalMemory(hidden_size=hidden_size)
    
    def forward(self, 
                x: torch.Tensor,
                memory_mode: str = "read") -> torch.Tensor:
        """
        Forward pass through memory-enhanced transformer layer.
        
        Args:
            x: Input tensor (batch_size, seq_len, hidden_size)
            memory_mode: Whether to read from or write to memory
            
        Returns:
            Output tensor (batch_size, seq_len, hidden_size)
        """
        # Self-attention
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        
        # Memory enhancement
        x = self.memory(x, memory_mode)
        
        # Feed-forward
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x

def demonstrate_hierarchical_memory():
    """Demonstrate hierarchical memory system."""
    print("Demonstrating hierarchical memory system...")
    
    # Create hierarchical memory
    memory = HierarchicalMemory(hidden_size=768)
    
    # Create sample input
    batch_size, seq_len, hidden_size = 2, 10, 768
    input_tensor = torch.randn(batch_size, seq_len, hidden_size)
    
    # Read from memory
    output_read = memory(input_tensor, memory_mode="read")
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output (read) shape: {output_read.shape}")
    
    # Write to memory
    output_write = memory(input_tensor, memory_mode="write")
    print(f"Output (write) shape: {output_write.shape}")
    
    # Get memory state
    memory_state = memory.get_memory_state()
    print(f"Memory state keys: {list(memory_state.keys())}")
    
    print("Hierarchical memory demonstration completed!")

if __name__ == "__main__":
    demonstrate_hierarchical_memory()