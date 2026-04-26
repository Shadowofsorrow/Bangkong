"""
Bangkong CA-MoE Model Implementation.
Integrates Cluster-Aware Sparse MoE into a standard GPT-2 transformer backbone.
"""
import math
import torch
import torch.nn as nn
from transformers import GPT2Config
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

# We import the core algorithm we built
try:
    from bangkong_moe import CAMoELayer
except ImportError:
    # Fallback if standalone library isn't installed, use local relative imports
    from bangkong.models.moelayer import CAMoELayer  # Placeholder for future structure

class CAMoEConfig(GPT2Config):
    """Configuration for CA-MoE models."""
    model_type = "ca_moe"
    
    def __init__(
        self,
        num_experts=8,
        top_k=2,
        expert_dim=None, # If None, defaults to intermediate_size
        num_clusters=4,
        num_domains=4,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_experts = num_experts
        self.top_k = top_k
        self.expert_dim = expert_dim if expert_dim is not None else self.intermediate_size
        self.num_clusters = num_clusters
        self.num_domains = num_domains

class CAMoEBlock(nn.Module):
    """Transformer Block with Sparse MoE instead of Dense FFN."""
    def __init__(self, config):
        super().__init__()
        hidden_size = config.hidden_size
        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attn_pdrop,
            batch_first=True
        )
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        
        # THE SWAP: Standard FFN -> CA-MoE Layer
        self.moe = CAMoELayer(
            hidden_size=hidden_size,
            num_experts=config.num_experts,
            top_k=config.top_k,
            expert_dim=config.expert_dim,
            num_clusters=config.num_clusters,
            num_domains=config.num_domains,
            dropout=config.resid_pdrop
        )

    def forward(self, hidden_states, attention_mask=None, past_key_value=None, output_attentions=False, use_cache=False):
        # Self Attention
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_output, attn_weights = self.attn(hidden_states, hidden_states, hidden_states, attn_mask=attention_mask)
        hidden_states = residual + attn_output

        # Sparse MoE
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        hidden_states = self.moe(hidden_states) # Sparse execution happens here
        hidden_states = residual + hidden_states

        return hidden_states

class BangkongCAMoEForCausalLM(nn.Module):
    """
    Full GPT-2 style Causal LM with Sparse MoE backbone.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.hidden_size)
        self.wpe = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([CAMoEBlock(config) for _ in range(config.num_hidden_layers)])
        self.ln_f = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Tie weights
        self.lm_head.weight = self.wte.weight

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0)
        
        hidden_states = self.wte(input_ids) + self.wpe(position_ids)
        hidden_states = self.drop(hidden_states)
        
        for block in self.h:
            hidden_states = block(hidden_states, attention_mask=attention_mask)
            
        hidden_states = self.ln_f(hidden_states)
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
        return CausalLMOutputWithCrossAttentions(loss=loss, logits=logits)
