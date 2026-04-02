"""
Validation module for Bangkong LLM Training System

This module provides tools for validating and measuring the effectiveness of various system components.
"""

from .scaling_law_validator import ScalingLawValidator

__all__ = [
    "ScalingLawValidator"
]