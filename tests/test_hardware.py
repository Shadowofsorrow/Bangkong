"""Tests for hardware detection module."""
import pytest
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from bangkong.hardware.detector import HardwareDetector


class TestHardwareDetector:
    """Test hardware detection."""

    def test_detect_cpu(self):
        """Test CPU detection always works."""
        detector = HardwareDetector()
        info = detector.detect()
        assert "cpu" in info
        assert info["cpu"]["available"] is True

    def test_detect_memory(self):
        """Test memory detection."""
        detector = HardwareDetector()
        info = detector.detect()
        assert "memory" in info
        assert info["memory"]["total_gb"] > 0

    def test_recommend_batch_size_cpu(self):
        """Test batch size recommendation for CPU."""
        detector = HardwareDetector()
        batch_size = detector.recommend_batch_size("cpu")
        assert batch_size > 0
        assert isinstance(batch_size, int)

    def test_recommend_num_workers(self):
        """Test num_workers recommendation."""
        detector = HardwareDetector()
        num_workers = detector.recommend_num_workers()
        assert num_workers >= 0
        assert isinstance(num_workers, int)
