"""Tests for bangkong.config module."""
import pytest
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from bangkong.config.loader import ConfigLoader
from bangkong.config.schemas import ModelConfig, TrainingConfig


class TestConfigLoader:
    """Test configuration loading."""

    def test_load_yaml_config(self):
        """Test loading a YAML config file."""
        config_path = project_root / "configs" / "default.yaml"
        if config_path.exists():
            loader = ConfigLoader()
            config = loader.load(str(config_path))
            assert config is not None

    def test_load_nonexistent_config(self):
        """Test loading a non-existent config raises error."""
        loader = ConfigLoader()
        with pytest.raises(FileNotFoundError):
            loader.load("/nonexistent/path/config.yaml")

    def test_config_merge(self):
        """Test merging two config dicts."""
        base = {"model": {"hidden_size": 768}, "training": {"batch_size": 4}}
        override = {"training": {"batch_size": 8}}
        loader = ConfigLoader()
        merged = loader._merge(base, override)
        assert merged["model"]["hidden_size"] == 768
        assert merged["training"]["batch_size"] == 8


class TestModelConfig:
    """Test model config schema."""

    def test_model_config_defaults(self):
        """Test model config with default values."""
        config = ModelConfig()
        assert config.hidden_size == 768
        assert config.num_layers == 12
        assert config.num_heads == 12

    def test_model_config_custom(self):
        """Test model config with custom values."""
        config = ModelConfig(hidden_size=256, num_layers=4, num_heads=4)
        assert config.hidden_size == 256
        assert config.num_layers == 4
        assert config.num_heads == 4


class TestTrainingConfig:
    """Test training config schema."""

    def test_training_config_defaults(self):
        """Test training config with default values."""
        config = TrainingConfig()
        assert config.batch_size == 4
        assert config.max_epochs == 10

    def test_training_config_custom(self):
        """Test training config with custom values."""
        config = TrainingConfig(batch_size=8, max_epochs=5)
        assert config.batch_size == 8
        assert config.max_epochs == 5
