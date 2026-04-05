"""
Curriculum learning for Bangkong LLM Training System
"""

import random
import torch
from typing import List, Dict, Any, Callable
from ..config.schemas import BangkongConfig


class CurriculumLearning:
    """Curriculum learning controller that adapts training difficulty over time."""
    
    def __init__(self, config: BangkongConfig):
        """
        Initialize curriculum learning controller.
        
        Args:
            config: Bangkong configuration
        """
        self.config = config
        self.curriculum_type = getattr(config.training, 'curriculum_type', 'none')
        self.curriculum_stages = getattr(config.training, 'curriculum_stages', 1)
        self.current_stage = 0
        self.samples_seen = 0
        self.stage_samples_threshold = getattr(config.training, 'stage_samples_threshold', 10000)
    
    def should_advance_stage(self) -> bool:
        """
        Determine if we should advance to the next curriculum stage.
        
        Returns:
            True if we should advance, False otherwise
        """
        if self.curriculum_type == 'none':
            return False
        
        # Advance based on number of samples seen
        if self.samples_seen >= self.stage_samples_threshold:
            return True
        
        return False
    
    def advance_stage(self):
        """Advance to the next curriculum stage."""
        if self.current_stage < self.curriculum_stages - 1:
            self.current_stage += 1
            self.samples_seen = 0
            print(f"Advanced to curriculum stage {self.current_stage + 1}")
    
    def get_current_difficulty(self) -> float:
        """
        Get the current difficulty level (0.0 to 1.0).
        
        Returns:
            Current difficulty level
        """
        if self.curriculum_stages <= 1:
            return 1.0
        
        return (self.current_stage + 1) / self.curriculum_stages
    
    def apply_curriculum_filter(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply curriculum-based filtering to samples.
        
        Args:
            samples: List of data samples
            
        Returns:
            Filtered list of samples based on current curriculum stage
        """
        if self.curriculum_type == 'none':
            return samples
        
        if self.curriculum_type == 'sequence_length':
            return self._filter_by_sequence_length(samples)
        elif self.curriculum_type == 'complexity':
            return self._filter_by_complexity(samples)
        elif self.curriculum_type == 'topic':
            return self._filter_by_topic(samples)
        else:
            return samples
    
    def _filter_by_sequence_length(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter samples by sequence length based on curriculum stage.
        
        Args:
            samples: List of data samples
            
        Returns:
            Filtered list of samples
        """
        if not samples:
            return samples
        
        # Calculate target max length based on current stage
        max_length = self.config.model.sequence_length
        current_max_length = int(max_length * self.get_current_difficulty())
        
        filtered_samples = []
        for sample in samples:
            if 'text' in sample:
                # Filter by text length
                if len(sample['text']) <= current_max_length:
                    filtered_samples.append(sample)
            else:
                # Keep non-text samples
                filtered_samples.append(sample)
        
        return filtered_samples
    
    def _filter_by_complexity(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter samples by complexity based on curriculum stage.
        
        Args:
            samples: List of data samples
            
        Returns:
            Filtered list of samples
        """
        if not samples:
            return samples
        
        # Simple complexity heuristic: number of unique words
        filtered_samples = []
        for sample in samples:
            if 'text' in sample:
                # Calculate complexity score
                words = sample['text'].split()
                unique_words = len(set(words))
                total_words = len(words)
                complexity_score = unique_words / total_words if total_words > 0 else 0
                
                # Filter by complexity score
                max_complexity = self.get_current_difficulty()
                if complexity_score <= max_complexity:
                    filtered_samples.append(sample)
            else:
                # Keep non-text samples
                filtered_samples.append(sample)
        
        return filtered_samples
    
    def _filter_by_topic(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter samples by topic based on curriculum stage.
        
        Args:
            samples: List of data samples
            
        Returns:
            Filtered list of samples
        """
        if not samples:
            return samples
        
        # Define topic progression (simplified)
        topic_stages = [
            ['simple', 'basic'],  # Stage 1: Simple topics
            ['intermediate', 'medium'],  # Stage 2: Intermediate topics
            ['complex', 'advanced']  # Stage 3: Complex topics
        ]
        
        # Get allowed topics for current stage
        allowed_topics = []
        for i in range(min(self.current_stage + 1, len(topic_stages))):
            allowed_topics.extend(topic_stages[i])
        
        filtered_samples = []
        for sample in samples:
            # Check if sample has a topic label
            if 'topic' in sample:
                if sample['topic'] in allowed_topics:
                    filtered_samples.append(sample)
            else:
                # Keep samples without topic labels
                filtered_samples.append(sample)
        
        return filtered_samples
    
    def update_sample_count(self, count: int):
        """
        Update the sample count for curriculum progression.
        
        Args:
            count: Number of samples processed
        """
        self.samples_seen += count


class AdaptiveBatchSampler:
    """Adaptive batch sampler that adjusts batch size based on curriculum stage."""
    
    def __init__(self, config: BangkongConfig):
        """
        Initialize adaptive batch sampler.
        
        Args:
            config: Bangkong configuration
        """
        self.config = config
        self.base_batch_size = getattr(config.training, 'batch_size', 1)
        self.curriculum_controller = CurriculumLearning(config)
    
    def get_adaptive_batch_size(self) -> int:
        """
        Get adaptive batch size based on current curriculum stage.
        
        Returns:
            Adaptive batch size
        """
        if isinstance(self.base_batch_size, int):
            base_size = self.base_batch_size
        else:
            # For "auto" batch size, use a default
            base_size = 1
        
        # Reduce batch size for earlier curriculum stages to allow more frequent updates
        difficulty = self.curriculum_controller.get_current_difficulty()
        adaptive_size = max(1, int(base_size * difficulty))
        
        return adaptive_size


def create_curriculum_controller(config: BangkongConfig) -> CurriculumLearning:
    """
    Create a curriculum learning controller.
    
    Args:
        config: Bangkong configuration
        
    Returns:
        Curriculum learning controller
    """
    return CurriculumLearning(config)


def create_adaptive_sampler(config: BangkongConfig) -> AdaptiveBatchSampler:
    """
    Create an adaptive batch sampler.
    
    Args:
        config: Bangkong configuration
        
    Returns:
        Adaptive batch sampler
    """
    return AdaptiveBatchSampler(config)