"""Shared pytest fixtures for Bangkong tests."""
import pytest
import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


@pytest.fixture
def device():
    """Return CPU device for consistent testing."""
    return torch.device("cpu")


@pytest.fixture
def tiny_model_config():
    """Return a minimal model config for fast testing."""
    return {
        "vocab_size": 1000,
        "hidden_size": 64,
        "num_layers": 2,
        "num_heads": 4,
        "max_position_embeddings": 128,
    }


@pytest.fixture
def sample_input_ids():
    """Return sample input token IDs."""
    return torch.randint(0, 1000, (2, 32))


@pytest.fixture
def sample_attention_mask():
    """Return sample attention mask."""
    return torch.ones(2, 32, dtype=torch.long)
