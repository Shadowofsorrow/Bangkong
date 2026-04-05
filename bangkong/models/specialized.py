"""
Specialized model architectures for Bangkong LLM Training System
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from transformers import GPT2Model, GPT2Config
from ..config.schemas import BangkongConfig


class CodeGPT2Model(nn.Module):
    """Specialized GPT-2 model for code generation and understanding."""
    
    def __init__(self, config: BangkongConfig):
        """
        Initialize code-specialized GPT-2 model.
        
        Args:
            config: Bangkong configuration
        """
        super().__init__()
        self.config = config
        
        # Create GPT-2 configuration with code-specific settings
        gpt2_config = GPT2Config(
            vocab_size=config.model.vocab_size,
            n_positions=config.model.sequence_length,
            n_embd=config.model.hidden_size,
            n_layer=config.model.num_layers,
            n_head=config.model.num_heads,
            # Code-specific settings
            resid_pdrop=0.1,  # Slightly higher dropout for code
            embd_pdrop=0.1,
            attn_pdrop=0.1,
            layer_norm_epsilon=1e-5,  # Tighter layer norm for code
        )
        
        # Initialize transformer
        self.transformer = GPT2Model(gpt2_config)
        
        # Code-specific output layer with bias
        self.lm_head = nn.Linear(config.model.hidden_size, config.model.vocab_size, bias=True)
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        """Initialize model weights with code-specific initialization."""
        # Initialize with smaller standard deviation for more stable training
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through code-specialized GPT-2 model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Labels for language modeling
            
        Returns:
            Dictionary with model outputs
        """
        # Pass through transformer
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Get hidden states
        hidden_states = transformer_outputs.last_hidden_state
        
        # Apply language modeling head
        lm_logits = self.lm_head(hidden_states)
        
        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            # Shift logits and labels for causal language modeling
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Calculate loss with label smoothing for better code generation
            loss_fct = nn.CrossEntropyLoss(label_smoothing=0.1)
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        
        return {
            "loss": loss,
            "logits": lm_logits,
            "hidden_states": hidden_states,
            "attentions": transformer_outputs.attentions
        }


class MathGPT2Model(nn.Module):
    """Specialized GPT-2 model for mathematical reasoning."""
    
    def __init__(self, config: BangkongConfig):
        """
        Initialize math-specialized GPT-2 model.
        
        Args:
            config: Bangkong configuration
        """
        super().__init__()
        self.config = config
        
        # Create GPT-2 configuration with math-specific settings
        gpt2_config = GPT2Config(
            vocab_size=config.model.vocab_size,
            n_positions=config.model.sequence_length,
            n_embd=config.model.hidden_size,
            n_layer=config.model.num_layers,
            n_head=config.model.num_heads,
            # Math-specific settings
            resid_pdrop=0.05,  # Lower dropout for precise math
            embd_pdrop=0.05,
            attn_pdrop=0.05,
            layer_norm_epsilon=1e-6,  # Very tight layer norm for math
        )
        
        # Initialize transformer
        self.transformer = GPT2Model(gpt2_config)
        
        # Math-specific output layer
        self.lm_head = nn.Linear(config.model.hidden_size, config.model.vocab_size, bias=False)
        
        # Add specialized math embedding for mathematical symbols
        self.math_embedding = nn.Embedding(1000, config.model.hidden_size)  # For special math tokens
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        """Initialize model weights with math-specific initialization."""
        # Initialize with very careful weights for numerical stability
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=0.01)  # Smaller std for math
                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through math-specialized GPT-2 model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Labels for language modeling
            
        Returns:
            Dictionary with model outputs
        """
        # Get base embeddings
        inputs_embeds = self.transformer.wte(input_ids)
        
        # Pass through transformer
        transformer_outputs = self.transformer(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask
        )
        
        # Get hidden states
        hidden_states = transformer_outputs.last_hidden_state
        
        # Apply language modeling head
        lm_logits = self.lm_head(hidden_states)
        
        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            # Shift logits and labels for causal language modeling
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Calculate loss with high precision for math
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        
        return {
            "loss": loss,
            "logits": lm_logits,
            "hidden_states": hidden_states,
            "attentions": transformer_outputs.attentions
        }


class ScientificGPT2Model(nn.Module):
    """Specialized GPT-2 model for scientific text processing."""
    
    def __init__(self, config: BangkongConfig):
        """
        Initialize scientific-specialized GPT-2 model.
        
        Args:
            config: Bangkong configuration
        """
        super().__init__()
        self.config = config
        
        # Create GPT-2 configuration with scientific-specific settings
        gpt2_config = GPT2Config(
            vocab_size=config.model.vocab_size,
            n_positions=config.model.sequence_length,
            n_embd=config.model.hidden_size,
            n_layer=config.model.num_layers,
            n_head=config.model.num_heads,
            # Scientific-specific settings
            resid_pdrop=0.1,
            embd_pdrop=0.1,
            attn_pdrop=0.1,
            layer_norm_epsilon=1e-5,
        )
        
        # Initialize transformer
        self.transformer = GPT2Model(gpt2_config)
        
        # Scientific-specific output layer
        self.lm_head = nn.Linear(config.model.hidden_size, config.model.vocab_size, bias=False)
        
        # Add specialized embeddings for scientific entities
        self.entity_embedding = nn.Embedding(5000, config.model.hidden_size)  # For scientific entities
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        """Initialize model weights with scientific-specific initialization."""
        # Initialize with domain-appropriate weights
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through scientific-specialized GPT-2 model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Labels for language modeling
            
        Returns:
            Dictionary with model outputs
        """
        # Get base embeddings
        inputs_embeds = self.transformer.wte(input_ids)
        
        # Pass through transformer
        transformer_outputs = self.transformer(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask
        )
        
        # Get hidden states
        hidden_states = transformer_outputs.last_hidden_state
        
        # Apply language modeling head
        lm_logits = self.lm_head(hidden_states)
        
        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            # Shift logits and labels for causal language modeling
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Calculate loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        
        return {
            "loss": loss,
            "logits": lm_logits,
            "hidden_states": hidden_states,
            "attentions": transformer_outputs.attentions
        }


def create_specialized_model(config: BangkongConfig) -> nn.Module:
    """
    Create a specialized model based on domain configuration.
    
    Args:
        config: Bangkong configuration
        
    Returns:
        Specialized model
    """
    domain = getattr(config.model, 'domain', 'general')
    
    if domain == 'code':
        return CodeGPT2Model(config)
    elif domain == 'math':
        return MathGPT2Model(config)
    elif domain == 'scientific':
        return ScientificGPT2Model(config)
    else:
        # Default to standard GPT-2 model
        gpt2_config = GPT2Config(
            vocab_size=config.model.vocab_size,
            n_positions=config.model.sequence_length,
            n_embd=config.model.hidden_size,
            n_layer=config.model.num_layers,
            n_head=config.model.num_heads,
        )
        return GPT2Model(gpt2_config)