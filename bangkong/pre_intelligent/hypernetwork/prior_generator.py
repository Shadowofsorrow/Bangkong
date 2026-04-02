#!/usr/bin/env python3
"""
Hypernetwork for generating learned priors for pre-intelligent initialization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any
import json

class HyperNetwork(nn.Module):
    """Hypernetwork that generates weights for target networks."""
    
    def __init__(self, 
                 input_dim: int = 128,
                 hidden_dim: int = 256,
                 output_dims: List[int] = None,
                 num_layers: int = 3):
        """
        Initialize hypernetwork.
        
        Args:
            input_dim: Dimension of input latent vector
            hidden_dim: Dimension of hidden layers
            output_dims: List of output dimensions for each weight tensor
            num_layers: Number of layers in hypernetwork
        """
        super(HyperNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dims = output_dims or [768*8, 8*768]  # Default for LoRA adapters (768, 8)
        self.num_layers = num_layers
        
        # Input embedding
        self.input_embedding = nn.Linear(input_dim, hidden_dim)
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 1)
        ])
        
        # Output heads for each weight tensor
        self.output_heads = nn.ModuleList([
            nn.Linear(hidden_dim, output_dim) for output_dim in self.output_dims
        ])
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with small values for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, z: torch.Tensor) -> List[torch.Tensor]:
        """
        Generate weight tensors from latent vector.
        
        Args:
            z: Latent vector (batch_size, input_dim)
            
        Returns:
            List of generated weight tensors
        """
        # Embed input
        h = F.relu(self.input_embedding(z))
        
        # Process through hidden layers
        for layer in self.hidden_layers:
            h = F.relu(layer(h))
        
        # Generate output weight tensors
        weights = []
        for head in self.output_heads:
            w = head(h)
            weights.append(w)
        
        return weights

class PriorGenerator:
    """Generator for learned priors using hypernetworks."""
    
    def __init__(self, 
                 latent_dim: int = 128,
                 knowledge_dim: int = 64,
                 adapter_rank: int = 8):
        """
        Initialize prior generator.
        
        Args:
            latent_dim: Dimension of latent space
            knowledge_dim: Dimension of knowledge embeddings
            adapter_rank: Rank for LoRA adapters
        """
        self.latent_dim = latent_dim
        self.knowledge_dim = knowledge_dim
        self.adapter_rank = adapter_rank
        
        # Hypernetwork for generating adapter weights
        self.hypernetwork = HyperNetwork(
            input_dim=latent_dim * 2,  # Both knowledge and task latents
            hidden_dim=256,
            output_dims=[768 * adapter_rank, adapter_rank * 768],  # LoRA A and B matrices
            num_layers=3
        )
        
        # Knowledge encoder (simulates knowledge base embeddings)
        self.knowledge_encoder = nn.Sequential(
            nn.Linear(knowledge_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        
        # Task context encoder
        self.task_encoder = nn.Sequential(
            nn.Linear(32, 64),  # Task features
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
    
    def encode_knowledge(self, knowledge_vector: torch.Tensor) -> torch.Tensor:
        """
        Encode knowledge vector into latent space.
        
        Args:
            knowledge_vector: Knowledge base embedding
            
        Returns:
            Encoded latent vector
        """
        return self.knowledge_encoder(knowledge_vector)
    
    def encode_task(self, task_features: torch.Tensor) -> torch.Tensor:
        """
        Encode task features into latent space.
        
        Args:
            task_features: Task-specific features
            
        Returns:
            Encoded latent vector
        """
        return self.task_encoder(task_features)
    
    def generate_priors(self, 
                       knowledge_vector: torch.Tensor, 
                       task_features: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Generate priors for model initialization.
        
        Args:
            knowledge_vector: Knowledge base embedding
            task_features: Optional task-specific features
            
        Returns:
            Dictionary of generated priors
        """
        # Encode knowledge
        z_knowledge = self.encode_knowledge(knowledge_vector)
        
        # Ensure z_knowledge has the right shape (batch dimension)
        if z_knowledge.dim() == 1:
            z_knowledge = z_knowledge.unsqueeze(0)  # Add batch dimension
        
        # Encode task if provided
        if task_features is not None:
            z_task = self.encode_task(task_features)
            if z_task.dim() == 1:
                z_task = z_task.unsqueeze(0)  # Add batch dimension
            # Combine knowledge and task embeddings
            z = torch.cat([z_knowledge, z_task], dim=-1)
        else:
            # If no task features, pad with zeros or use only knowledge
            # Create zero tensor with task dimension
            batch_size = z_knowledge.shape[0]
            z_task = torch.zeros(batch_size, self.latent_dim, device=z_knowledge.device)
            z = torch.cat([z_knowledge, z_task], dim=-1)
        
        print(f"Combined latent vector z shape: {z.shape}")
        
        # Generate weights using hypernetwork
        weights = self.hypernetwork(z)
        
        # Reshape weights into LoRA adapter matrices
        adapter_A = weights[0].view(-1, 768, self.adapter_rank)
        adapter_B = weights[1].view(-1, self.adapter_rank, 768)
        
        return {
            'adapter_A': adapter_A,
            'adapter_B': adapter_B,
            'latent_vector': z
        }
    
    def save_priors(self, priors: Dict[str, torch.Tensor], filepath: str):
        """
        Save generated priors to file.
        
        Args:
            priors: Dictionary of priors
            filepath: Path to save priors
        """
        # Convert tensors to CPU and numpy for saving
        priors_cpu = {k: v.detach().cpu() for k, v in priors.items()}
        
        # Save as torch tensors
        torch.save(priors_cpu, filepath)
        print(f"Priors saved to {filepath}")
    
    def load_priors(self, filepath: str) -> Dict[str, torch.Tensor]:
        """
        Load priors from file.
        
        Args:
            filepath: Path to load priors from
            
        Returns:
            Dictionary of loaded priors
        """
        priors = torch.load(filepath)
        print(f"Priors loaded from {filepath}")
        return priors

class KnowledgeBase:
    """Simulated knowledge base for generating embeddings."""
    
    def __init__(self):
        """Initialize knowledge base."""
        self.concepts = {
            'math': torch.randn(64),
            'logic': torch.randn(64),
            'reasoning': torch.randn(64),
            'code': torch.randn(64),
            'language': torch.randn(64)
        }
    
    def get_concept_embedding(self, concept: str) -> torch.Tensor:
        """
        Get embedding for a concept.
        
        Args:
            concept: Concept name
            
        Returns:
            Concept embedding
        """
        if concept in self.concepts:
            return self.concepts[concept]
        else:
            # Return random embedding for unknown concepts
            return torch.randn(64)
    
    def get_composite_embedding(self, concepts: List[str]) -> torch.Tensor:
        """
        Get composite embedding for multiple concepts.
        
        Args:
            concepts: List of concept names
            
        Returns:
            Composite embedding
        """
        embeddings = [self.get_concept_embedding(concept) for concept in concepts]
        return torch.stack(embeddings).mean(dim=0)

def train_hypernetwork(num_epochs: int = 1000):
    """
    Train hypernetwork to generate priors.
    
    Args:
        num_epochs: Number of training epochs
        
    Returns:
        Trained prior generator
    """
    # Initialize components
    prior_generator = PriorGenerator()
    knowledge_base = KnowledgeBase()
    
    # Optimizer
    optimizer = torch.optim.Adam(prior_generator.parameters(), lr=0.001)
    
    print("Training hypernetwork for prior generation...")
    
    for epoch in range(num_epochs):
        # Sample knowledge concepts
        concepts = ['math', 'logic', 'reasoning']
        knowledge_vector = knowledge_base.get_composite_embedding(concepts)
        
        # Sample task features (simulated)
        task_features = torch.randn(32)
        
        # Generate priors
        priors = prior_generator.generate_priors(knowledge_vector, task_features)
        
        # Simple loss function (just for demonstration)
        # In practice, this would be based on how well the generated priors work
        loss = 0
        for weight in priors.values():
            if isinstance(weight, torch.Tensor):
                loss += weight.pow(2).mean()
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 200 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.6f}")
    
    print("Hypernetwork training completed!")
    return prior_generator

if __name__ == "__main__":
    # Train hypernetwork
    prior_generator = train_hypernetwork(num_epochs=1000)
    
    # Generate sample priors
    knowledge_base = KnowledgeBase()
    knowledge_vector = knowledge_base.get_composite_embedding(['math', 'reasoning'])
    task_features = torch.randn(32)
    
    priors = prior_generator.generate_priors(knowledge_vector, task_features)
    
    print(f"Generated priors:")
    for name, tensor in priors.items():
        if isinstance(tensor, torch.Tensor):
            print(f"  {name}: {tensor.shape}")
    
    # Save priors
    prior_generator.save_priors(priors, "sample_priors.pt")