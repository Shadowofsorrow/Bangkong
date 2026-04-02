#!/usr/bin/env python3
"""
Reasoning organs (auxiliary heads) for pre-intelligent LLMs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, List, Optional
import math

class GraphConstructionHead(nn.Module):
    """Graph construction head for building entity-relation graphs."""
    
    def __init__(self, hidden_size: int = 768, graph_dim: int = 128):
        """
        Initialize graph construction head.
        
        Args:
            hidden_size: Size of hidden representations
            graph_dim: Dimension of graph embeddings
        """
        super(GraphConstructionHead, self).__init__()
        
        self.hidden_size = hidden_size
        self.graph_dim = graph_dim
        
        # Entity extraction
        self.entity_extractor = nn.Linear(hidden_size, graph_dim)
        
        # Relation classifier
        self.relation_classifier = nn.Linear(hidden_size * 2, graph_dim)
        
        # Graph embedding generator
        self.graph_embedder = nn.Linear(graph_dim * 2, graph_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with appropriate values."""
        nn.init.xavier_uniform_(self.entity_extractor.weight)
        nn.init.xavier_uniform_(self.relation_classifier.weight)
        nn.init.xavier_uniform_(self.graph_embedder.weight)
        
        nn.init.zeros_(self.entity_extractor.bias)
        nn.init.zeros_(self.relation_classifier.bias)
        nn.init.zeros_(self.graph_embedder.bias)
    
    def extract_entities(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Extract entity representations from hidden states.
        
        Args:
            hidden_states: Hidden states (batch_size, seq_len, hidden_size)
            
        Returns:
            Entity embeddings (batch_size, seq_len, graph_dim)
        """
        entities = self.entity_extractor(hidden_states)
        return F.relu(entities)
    
    def classify_relations(self, 
                          entity_i: torch.Tensor, 
                          entity_j: torch.Tensor) -> torch.Tensor:
        """
        Classify relations between entity pairs.
        
        Args:
            entity_i: First entity embedding (batch_size, graph_dim)
            entity_j: Second entity embedding (batch_size, graph_dim)
            
        Returns:
            Relation embedding (batch_size, graph_dim)
        """
        # Concatenate entity pairs
        entity_pair = torch.cat([entity_i, entity_j], dim=-1)
        relations = self.relation_classifier(entity_pair)
        return F.relu(relations)
    
    def construct_graph(self, hidden_states: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Construct graph from hidden states.
        
        Args:
            hidden_states: Hidden states (batch_size, seq_len, hidden_size)
            
        Returns:
            Dictionary containing graph components
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Extract entities
        entities = self.extract_entities(hidden_states)  # (batch_size, seq_len, graph_dim)
        
        # Simple relation extraction (between consecutive entities)
        relations = []
        for i in range(seq_len - 1):
            rel = self.classify_relations(entities[:, i, :], entities[:, i+1, :])
            relations.append(rel)
        
        # Pad relations to match sequence length
        if relations:
            relations = torch.stack(relations, dim=1)  # (batch_size, seq_len-1, graph_dim)
            # Pad to match entities sequence length
            pad = torch.zeros(batch_size, 1, self.graph_dim, device=relations.device)
            relations = torch.cat([relations, pad], dim=1)  # (batch_size, seq_len, graph_dim)
        else:
            relations = torch.zeros_like(entities)
        
        # Combine entities and relations for graph representation
        graph_rep = self.graph_embedder(
            torch.cat([entities, relations], dim=-1)
        )  # (batch_size, seq_len, graph_dim)
        
        return {
            'entities': entities,
            'relations': relations,
            'graph_embedding': graph_rep
        }

class ReasoningValidator(nn.Module):
    """Self-critic head for validating reasoning chains."""
    
    def __init__(self, hidden_size: int = 768, validator_dim: int = 256):
        """
        Initialize reasoning validator.
        
        Args:
            hidden_size: Size of hidden representations
            validator_dim: Dimension of validator representations
        """
        super(ReasoningValidator, self).__init__()
        
        self.hidden_size = hidden_size
        self.validator_dim = validator_dim
        
        # Consistency checker
        self.consistency_checker = nn.Linear(hidden_size, validator_dim)
        
        # Logical validity classifier
        self.validity_classifier = nn.Linear(validator_dim, 2)  # Valid/Invalid
        
        # Confidence scorer
        self.confidence_scorer = nn.Linear(validator_dim, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with appropriate values."""
        nn.init.xavier_uniform_(self.consistency_checker.weight)
        nn.init.xavier_uniform_(self.validity_classifier.weight)
        nn.init.xavier_uniform_(self.confidence_scorer.weight)
        
        nn.init.zeros_(self.consistency_checker.bias)
        nn.init.zeros_(self.validity_classifier.bias)
        nn.init.zeros_(self.confidence_scorer.bias)
    
    def check_consistency(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Check consistency of reasoning steps.
        
        Args:
            hidden_states: Hidden states representing reasoning steps
                          (batch_size, seq_len, hidden_size)
            
        Returns:
            Consistency scores (batch_size, seq_len, validator_dim)
        """
        consistency = self.consistency_checker(hidden_states)
        return F.relu(consistency)
    
    def classify_validity(self, consistency_scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Classify validity of reasoning steps.
        
        Args:
            consistency_scores: Consistency scores
                               (batch_size, seq_len, validator_dim)
            
        Returns:
            Tuple of (validity_logits, confidence_scores)
        """
        # Global pooling for sequence-level classification
        pooled = torch.mean(consistency_scores, dim=1)  # (batch_size, validator_dim)
        
        # Classify validity
        validity_logits = self.validity_classifier(pooled)  # (batch_size, 2)
        
        # Compute confidence
        confidence_scores = torch.sigmoid(self.confidence_scorer(pooled))  # (batch_size, 1)
        
        return validity_logits, confidence_scores
    
    def validate_reasoning(self, hidden_states: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Validate reasoning chain.
        
        Args:
            hidden_states: Hidden states representing reasoning steps
                          (batch_size, seq_len, hidden_size)
            
        Returns:
            Dictionary containing validation results
        """
        # Check consistency
        consistency_scores = self.check_consistency(hidden_states)
        
        # Classify validity
        validity_logits, confidence_scores = self.classify_validity(consistency_scores)
        
        return {
            'consistency_scores': consistency_scores,
            'validity_logits': validity_logits,
            'confidence_scores': confidence_scores,
            'is_valid': torch.argmax(validity_logits, dim=-1),  # 0=Invalid, 1=Valid
            'confidence': confidence_scores.squeeze(-1)
        }

class TemporalReasoningHead(nn.Module):
    """Temporal reasoning head for sequence planning."""
    
    def __init__(self, hidden_size: int = 768, temporal_dim: int = 128):
        """
        Initialize temporal reasoning head.
        
        Args:
            hidden_size: Size of hidden representations
            temporal_dim: Dimension of temporal representations
        """
        super(TemporalReasoningHead, self).__init__()
        
        self.hidden_size = hidden_size
        self.temporal_dim = temporal_dim
        
        # Temporal encoder
        self.temporal_encoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=temporal_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        # Planning decoder
        self.planning_decoder = nn.Linear(temporal_dim * 2, hidden_size)
        
        # Sequence length predictor
        self.length_predictor = nn.Linear(temporal_dim * 2, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with appropriate values."""
        nn.init.xavier_uniform_(self.planning_decoder.weight)
        nn.init.xavier_uniform_(self.length_predictor.weight)
        
        nn.init.zeros_(self.planning_decoder.bias)
        nn.init.zeros_(self.length_predictor.bias)
    
    def encode_temporal(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode temporal patterns in hidden states.
        
        Args:
            hidden_states: Hidden states (batch_size, seq_len, hidden_size)
            
        Returns:
            Tuple of (encoded_temporal, final_hidden)
        """
        encoded_temporal, (hidden, cell) = self.temporal_encoder(hidden_states)
        return encoded_temporal, hidden
    
    def decode_planning(self, temporal_encoding: torch.Tensor) -> torch.Tensor:
        """
        Decode temporal encoding into planning sequence.
        
        Args:
            temporal_encoding: Temporal encoding (batch_size, seq_len, temporal_dim*2)
            
        Returns:
            Planning sequence (batch_size, seq_len, hidden_size)
        """
        planning = self.planning_decoder(temporal_encoding)
        return F.relu(planning)
    
    def predict_sequence_length(self, temporal_encoding: torch.Tensor) -> torch.Tensor:
        """
        Predict optimal sequence length.
        
        Args:
            temporal_encoding: Temporal encoding (batch_size, seq_len, temporal_dim*2)
            
        Returns:
            Predicted sequence lengths (batch_size, 1)
        """
        # Use final timestep for prediction
        final_encoding = temporal_encoding[:, -1, :]  # (batch_size, temporal_dim*2)
        length_logits = self.length_predictor(final_encoding)  # (batch_size, 1)
        return torch.relu(length_logits)
    
    def temporal_reasoning(self, hidden_states: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Perform temporal reasoning.
        
        Args:
            hidden_states: Hidden states (batch_size, seq_len, hidden_size)
            
        Returns:
            Dictionary containing temporal reasoning results
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Encode temporal patterns
        temporal_encoding, final_hidden = self.encode_temporal(hidden_states)
        
        # Decode into planning sequence
        planning_sequence = self.decode_planning(temporal_encoding)
        
        # Predict sequence length
        predicted_length = self.predict_sequence_length(temporal_encoding)
        
        return {
            'temporal_encoding': temporal_encoding,
            'planning_sequence': planning_sequence,
            'predicted_length': predicted_length,
            'final_hidden': final_hidden
        }

class ReasoningOrgans(nn.Module):
    """Combined reasoning organs for pre-intelligent LLMs."""
    
    def __init__(self, hidden_size: int = 768):
        """
        Initialize reasoning organs.
        
        Args:
            hidden_size: Size of hidden representations
        """
        super(ReasoningOrgans, self).__init__()
        
        self.hidden_size = hidden_size
        
        # Individual reasoning organs
        self.graph_head = GraphConstructionHead(hidden_size, graph_dim=128)
        self.validator_head = ReasoningValidator(hidden_size, validator_dim=256)
        self.temporal_head = TemporalReasoningHead(hidden_size, temporal_dim=128)
        
        # Organ coordinator
        self.organ_coordinator = nn.Linear(hidden_size * 3, hidden_size)
        
        # Initialize coordinator
        self._init_coordinator()
    
    def _init_coordinator(self):
        """Initialize organ coordinator weights."""
        nn.init.xavier_uniform_(self.organ_coordinator.weight)
        nn.init.zeros_(self.organ_coordinator.bias)
    
    def forward(self, hidden_states: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Process through all reasoning organs.
        
        Args:
            hidden_states: Hidden states (batch_size, seq_len, hidden_size)
            
        Returns:
            Dictionary containing outputs from all reasoning organs
        """
        # Process through individual organs
        graph_output = self.graph_head.construct_graph(hidden_states)
        validator_output = self.validator_head.validate_reasoning(hidden_states)
        temporal_output = self.temporal_head.temporal_reasoning(hidden_states)
        
        # Combine organ outputs for enhanced representation
        organ_features = torch.cat([
            graph_output['graph_embedding'],
            validator_output['consistency_scores'],
            temporal_output['temporal_encoding'].mean(dim=1, keepdim=True).expand(-1, hidden_states.size(1), -1)
        ], dim=-1)
        
        # Coordinate organ outputs
        coordinated_output = self.organ_coordinator(organ_features)
        
        return {
            'graph': graph_output,
            'validator': validator_output,
            'temporal': temporal_output,
            'coordinated_output': coordinated_output,
            'enhanced_hidden': hidden_states + coordinated_output  # Residual connection
        }

def demonstrate_reasoning_organs():
    """Demonstrate reasoning organs functionality."""
    print("Demonstrating reasoning organs...")
    
    # Create reasoning organs
    organs = ReasoningOrgans(hidden_size=768)
    
    # Create sample input
    batch_size, seq_len, hidden_size = 2, 10, 768
    input_tensor = torch.randn(batch_size, seq_len, hidden_size)
    
    # Process through reasoning organs
    outputs = organs(input_tensor)
    
    print(f"Input shape: {input_tensor.shape}")
    print(f"Enhanced hidden shape: {outputs['enhanced_hidden'].shape}")
    print(f"Graph entities shape: {outputs['graph']['entities'].shape}")
    print(f"Validator validity: {outputs['validator']['is_valid']}")
    print(f"Temporal planning shape: {outputs['temporal']['planning_sequence'].shape}")
    
    print("Reasoning organs demonstration completed!")

if __name__ == "__main__":
    demonstrate_reasoning_organs()