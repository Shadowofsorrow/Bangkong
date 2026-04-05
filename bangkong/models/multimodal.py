"""
Multimodal models for Bangkong LLM Training System
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from transformers import GPT2Model, GPT2Config
from ..config.schemas import BangkongConfig


class MultimodalEmbedding(nn.Module):
    """Multimodal embedding layer that combines text, image, audio, and video embeddings."""
    
    def __init__(self, config: BangkongConfig):
        """
        Initialize multimodal embedding layer.
        
        Args:
            config: Bangkong configuration
        """
        super().__init__()
        self.config = config
        self.hidden_size = config.model.hidden_size
        
        # Text embedding (using GPT-2 embedding)
        self.text_embedding = nn.Embedding(config.model.vocab_size, self.hidden_size)
        
        # Image embedding (if image size is specified)
        if getattr(config.model, 'image_size', None):
            self.image_embedding = nn.Conv2d(
                in_channels=3,  # RGB channels
                out_channels=self.hidden_size,
                kernel_size=16,
                stride=16
            )
        
        # Audio embedding (if audio sample rate is specified)
        if getattr(config.model, 'audio_sample_rate', None):
            self.audio_embedding = nn.Conv1d(
                in_channels=1,  # Mono audio
                out_channels=self.hidden_size,
                kernel_size=16,
                stride=16
            )
        
        # Video embedding (if video fps is specified)
        if getattr(config.model, 'video_fps', None):
            self.video_embedding = nn.Conv3d(
                in_channels=3,  # RGB channels
                out_channels=self.hidden_size,
                kernel_size=(3, 16, 16),
                stride=(1, 16, 16)
            )
        
        # Modality type embedding
        self.modality_embedding = nn.Embedding(4, self.hidden_size)  # text, image, audio, video
        
        # Positional embedding
        self.position_embedding = nn.Embedding(
            config.model.sequence_length, 
            self.hidden_size
        )
        
        # Layer normalization and dropout
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        self.dropout = nn.Dropout(0.1)
    
    def forward(
        self, 
        input_ids: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        audio_values: Optional[torch.Tensor] = None,
        video_values: Optional[torch.Tensor] = None,
        modality_types: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through multimodal embedding layer.
        
        Args:
            input_ids: Text input IDs
            pixel_values: Image pixel values
            audio_values: Audio values
            video_values: Video values
            modality_types: Modality type indicators
            position_ids: Position IDs
            
        Returns:
            Embedded representations
        """
        embeddings = []
        batch_size = 0
        
        # Process text inputs
        if input_ids is not None:
            batch_size = input_ids.shape[0]
            text_embeds = self.text_embedding(input_ids)
            embeddings.append(text_embeds)
        
        # Process image inputs
        if pixel_values is not None:
            if batch_size == 0:
                batch_size = pixel_values.shape[0]
            # Reshape and embed images
            image_embeds = self.image_embedding(pixel_values)
            # Reshape to (batch_size, sequence_length, hidden_size)
            image_embeds = image_embeds.flatten(2).transpose(1, 2)
            embeddings.append(image_embeds)
        
        # Process audio inputs
        if audio_values is not None:
            if batch_size == 0:
                batch_size = audio_values.shape[0]
            # Embed audio
            audio_embeds = self.audio_embedding(audio_values)
            # Reshape to (batch_size, sequence_length, hidden_size)
            audio_embeds = audio_embeds.transpose(1, 2)
            embeddings.append(audio_embeds)
        
        # Process video inputs
        if video_values is not None:
            if batch_size == 0:
                batch_size = video_values.shape[0]
            # Embed video
            video_embeds = self.video_embedding(video_values)
            # Reshape to (batch_size, sequence_length, hidden_size)
            video_embeds = video_embeds.flatten(2).transpose(1, 2)
            embeddings.append(video_embeds)
        
        if not embeddings:
            raise ValueError("At least one modality must be provided")
        
        # Concatenate all embeddings
        combined_embeds = torch.cat(embeddings, dim=1)
        
        # Add modality type embeddings
        if modality_types is not None:
            modality_embeds = self.modality_embedding(modality_types)
            combined_embeds = combined_embeds + modality_embeds
        
        # Add positional embeddings
        if position_ids is None:
            seq_length = combined_embeds.shape[1]
            position_ids = torch.arange(seq_length, dtype=torch.long, device=combined_embeds.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        position_embeds = self.position_embedding(position_ids)
        combined_embeds = combined_embeds + position_embeds
        
        # Apply layer normalization and dropout
        combined_embeds = self.layer_norm(combined_embeds)
        combined_embeds = self.dropout(combined_embeds)
        
        return combined_embeds


class MultimodalGPT2Model(nn.Module):
    """Multimodal GPT-2 model that can process text, images, audio, and video."""
    
    def __init__(self, config: BangkongConfig):
        """
        Initialize multimodal GPT-2 model.
        
        Args:
            config: Bangkong configuration
        """
        super().__init__()
        self.config = config
        
        # Multimodal embedding layer
        self.embeddings = MultimodalEmbedding(config)
        
        # GPT-2 transformer
        gpt2_config = GPT2Config(
            vocab_size=config.model.vocab_size,
            n_positions=config.model.sequence_length,
            n_embd=config.model.hidden_size,
            n_layer=config.model.num_layers,
            n_head=config.model.num_heads
        )
        self.transformer = GPT2Model(gpt2_config)
        
        # Output layer for language modeling
        self.lm_head = nn.Linear(config.model.hidden_size, config.model.vocab_size, bias=False)
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        """Initialize model weights."""
        # Initialize text embedding weights from GPT-2
        if hasattr(self.transformer, 'wte'):
            self.embeddings.text_embedding.weight = self.transformer.wte.weight
        
        # Initialize output layer weights
        self.lm_head.weight = self.embeddings.text_embedding.weight
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        audio_values: Optional[torch.Tensor] = None,
        video_values: Optional[torch.Tensor] = None,
        modality_types: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through multimodal GPT-2 model.
        
        Args:
            input_ids: Text input IDs
            pixel_values: Image pixel values
            audio_values: Audio values
            video_values: Video values
            modality_types: Modality type indicators
            attention_mask: Attention mask
            labels: Labels for language modeling
            
        Returns:
            Dictionary with model outputs
        """
        # Get embeddings
        embeddings = self.embeddings(
            input_ids=input_ids,
            pixel_values=pixel_values,
            audio_values=audio_values,
            video_values=video_values,
            modality_types=modality_types
        )
        
        # Pass through transformer
        transformer_outputs = self.transformer(
            inputs_embeds=embeddings,
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


def create_multimodal_model(config: BangkongConfig) -> MultimodalGPT2Model:
    """
    Create a multimodal model based on configuration.
    
    Args:
        config: Bangkong configuration
        
    Returns:
        Multimodal model
    """
    return MultimodalGPT2Model(config)