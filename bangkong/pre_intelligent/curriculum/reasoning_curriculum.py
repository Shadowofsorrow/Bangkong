#!/usr/bin/env python3
"""
Curriculum learning with synthetic reasoning traces for pre-intelligent LLMs
"""

import torch
from torch.utils.data import Dataset, DataLoader
import json
import random
from typing import List, Dict, Tuple, Any, Optional
import math
from abc import ABC, abstractmethod
from .config_loader import get_config


class SyntheticReasoningTask(ABC):
    """Base class for synthetic reasoning tasks."""
    
    def __init__(self, task_type: str, difficulty: int = 1):
        """
        Initialize synthetic reasoning task.
        
        Args:
            task_type: Type of reasoning task
            difficulty: Difficulty level (1-5)
        """
        self.task_type = task_type
        self.difficulty = difficulty
        self.config = get_config()
    
    @abstractmethod
    def generate_sample(self) -> Dict[str, Any]:
        """Generate a single sample for this task."""
        pass
    
    @abstractmethod
    def get_prompt_template(self) -> str:
        """Get prompt template for this task."""
        pass

class ArithmeticChainTask(SyntheticReasoningTask):
    """Arithmetic chain reasoning task."""
    
    def __init__(self, difficulty: int = 1):
        super().__init__("arithmetic_chain", difficulty)
        arithmetic_config = self.config.get("curriculum.arithmetic_chain", {})
        self.operations = ["+", "-", "*"]
        base_min_operations = arithmetic_config.get("base_min_operations", 3)
        max_operations_limit = arithmetic_config.get("max_operations_limit", 8)
        self.max_numbers = min(base_min_operations + difficulty, max_operations_limit)
        self.max_number_value = arithmetic_config.get("max_number_value", 20)
    
    def generate_sample(self) -> Dict[str, Any]:
        """Generate arithmetic chain sample."""
        # Generate chain of operations
        num_operations = random.randint(2, self.max_numbers - 1)
        numbers = [random.randint(1, self.max_number_value) for _ in range(num_operations + 1)]
        operations = [random.choice(self.operations) for _ in range(num_operations)]
        
        # Build reasoning chain
        chain_steps = []
        current_result = numbers[0]
        chain_steps.append(f"Start with {current_result}")
        
        for i, (num, op) in enumerate(zip(numbers[1:], operations)):
            if op == "+":
                current_result += num
                chain_steps.append(f"Step {i+1}: {current_result - num} + {num} = {current_result}")
            elif op == "-":
                current_result -= num
                chain_steps.append(f"Step {i+1}: {current_result + num} - {num} = {current_result}")
            elif op == "*":
                current_result *= num
                chain_steps.append(f"Step {i+1}: {current_result // num} × {num} = {current_result}")
        
        # Create prompt and response
        prompt = f"Solve this step by step: {' '.join([str(numbers[0])] + [f'{op} {num}' for num, op in zip(numbers[1:], operations)])}"
        reasoning_chain = "\n".join(chain_steps)
        final_answer = f"The answer is {current_result}"
        
        return {
            "task_type": self.task_type,
            "difficulty": self.difficulty,
            "prompt": prompt,
            "reasoning_chain": reasoning_chain,
            "final_answer": final_answer,
            "ground_truth": current_result
        }
    
    def get_prompt_template(self) -> str:
        return "Solve the following arithmetic problem step by step: {problem}"

class LogicChainTask(SyntheticReasoningTask):
    """Logic chain reasoning task."""
    
    def __init__(self, difficulty: int = 1):
        super().__init__("logic_chain", difficulty)
        logic_config = self.config.get("curriculum.logic_chain", {})
        base_min_statements = logic_config.get("base_min_statements", 3)
        max_statements_limit = logic_config.get("max_statements_limit", 6)
        self.max_statements = min(base_min_statements + difficulty, max_statements_limit)
        task_data = self.config.get_task_data("logic_chain")
        self.variables = task_data.get("variables", ["A", "B", "C", "D", "E", "F"])
    
    def generate_sample(self) -> Dict[str, Any]:
        """Generate logic chain sample."""
        # Generate logical statements
        statements = []
        conclusions = []
        
        # Simple logical chain
        variables = ["A", "B", "C", "D", "E", "F"]
        num_statements = random.randint(2, self.max_statements)
        
        # Create implication chain
        chain_steps = []
        chain_steps.append("Given the following logical statements:")
        
        for i in range(num_statements - 1):
            premise = variables[i]
            conclusion = variables[i + 1]
            statements.append(f"If {premise} then {conclusion}")
            chain_steps.append(f"{i+1}. If {premise} then {conclusion}")
        
        # Add initial fact
        initial_var = variables[0]
        statements.append(f"{initial_var} is true")
        chain_steps.append(f"{num_statements}. {initial_var} is true")
        
        # Derive conclusions
        chain_steps.append("Deriving conclusions:")
        current_fact = initial_var
        for i in range(num_statements - 1):
            next_var = variables[i + 1]
            conclusions.append(f"Therefore {next_var} is true")
            chain_steps.append(f"{num_statements + i + 1}. Since {current_fact} is true and 'If {current_fact} then {next_var}', therefore {next_var} is true")
            current_fact = next_var
        
        # Create prompt and response
        prompt = "Given: " + ", ".join(statements[:-1]) + f". {statements[-1]}. What can we conclude?"
        reasoning_chain = "\n".join(chain_steps)
        final_answer = f"Therefore {current_fact} is true"
        
        return {
            "task_type": self.task_type,
            "difficulty": self.difficulty,
            "prompt": prompt,
            "reasoning_chain": reasoning_chain,
            "final_answer": final_answer,
            "ground_truth": current_fact
        }
    
    def get_prompt_template(self) -> str:
        return "Analyze the following logical statements: {statements}"

class PlanningTask(SyntheticReasoningTask):
    """Planning reasoning task."""
    
    def __init__(self, difficulty: int = 1):
        super().__init__("planning", difficulty)
        planning_config = self.config.get("curriculum.planning", {})
        base_min_steps = planning_config.get("base_min_steps", 3)
        max_steps_limit = planning_config.get("max_steps_limit", 7)
        self.max_steps = min(base_min_steps + difficulty, max_steps_limit)
        task_data = self.config.get_task_data("planning")
        self.actions = task_data.get("actions", ["move", "pick", "place", "open", "close", "turn_on", "turn_off"])
        self.objects = task_data.get("objects", ["box", "key", "door", "light", "book", "cup", "phone"])
        self.locations = task_data.get("locations", ["kitchen", "bedroom", "living_room", "office", "bathroom"])
    
    def generate_sample(self) -> Dict[str, Any]:
        """Generate planning sample."""
        # Generate goal and initial state
        goal_object = random.choice(self.objects)
        goal_location = random.choice(self.locations)
        initial_location = random.choice([loc for loc in self.locations if loc != goal_location])
        
        # Generate action sequence
        num_steps = random.randint(2, self.max_steps)
        action_sequence = []
        current_location = initial_location
        has_object = False
        
        chain_steps = []
        chain_steps.append(f"Goal: Move {goal_object} to {goal_location}")
        chain_steps.append(f"Current state: {goal_object} is in {initial_location}")
        chain_steps.append("Plan:")
        
        for i in range(num_steps):
            if i == 0 and not has_object:
                # Pick up object
                action = f"pick {goal_object}"
                has_object = True
                chain_steps.append(f"Step {i+1}: {action} - now holding {goal_object}")
            elif i == num_steps - 1 and has_object:
                # Place object at goal location
                action = f"place {goal_object} in {goal_location}"
                chain_steps.append(f"Step {i+1}: {action} - {goal_object} is now in {goal_location}")
            else:
                # Move to different location
                new_location = random.choice([loc for loc in self.locations if loc != current_location])
                action = f"move to {new_location}"
                current_location = new_location
                chain_steps.append(f"Step {i+1}: {action} - now in {current_location}")
            
            action_sequence.append(action)
        
        # Create prompt and response
        prompt = f"How to move {goal_object} from {initial_location} to {goal_location}?"
        reasoning_chain = "\n".join(chain_steps)
        final_answer = f"Plan executed successfully: {goal_object} moved to {goal_location}"
        
        return {
            "task_type": self.task_type,
            "difficulty": self.difficulty,
            "prompt": prompt,
            "reasoning_chain": reasoning_chain,
            "final_answer": final_answer,
            "ground_truth": f"{goal_object} in {goal_location}"
        }
    
    def get_prompt_template(self) -> str:
        return "Create a plan to achieve the following goal: {goal}"

class SyntheticReasoningDataset(Dataset):
    """Dataset of synthetic reasoning traces."""
    
    def __init__(self, 
                 num_samples: int = None,
                 difficulty_range: Tuple[int, int] = None,
                 task_distribution: Optional[Dict[str, float]] = None):
        """
        Initialize synthetic reasoning dataset.
        
        Args:
            num_samples: Number of samples to generate
            difficulty_range: Range of difficulty levels (min, max)
            task_distribution: Distribution of task types
        """
        self.config = get_config()
        self.num_samples = num_samples if num_samples is not None else self.config.get("curriculum.default_num_samples", 1000)
        self.difficulty_range = difficulty_range if difficulty_range is not None else tuple(self.config.get("curriculum.default_difficulty_range", [1, 3]))
        self.task_distribution = task_distribution or self.config.get_task_distribution("default")
        
        # Generate samples
        self.samples = self._generate_samples()
    
    def _generate_samples(self) -> List[Dict[str, Any]]:
        """Generate synthetic reasoning samples."""
        samples = []
        
        # Create task instances
        task_instances = []
        for task_type, proportion in self.task_distribution.items():
            num_task_samples = int(self.num_samples * proportion)
            for _ in range(num_task_samples):
                difficulty = random.randint(self.difficulty_range[0], self.difficulty_range[1])
                if task_type == "arithmetic_chain":
                    task = ArithmeticChainTask(difficulty)
                elif task_type == "logic_chain":
                    task = LogicChainTask(difficulty)
                elif task_type == "planning":
                    task = PlanningTask(difficulty)
                elif task_type == "causal_reasoning":
                    task = CausalReasoningTask(difficulty)
                elif task_type == "analogical_reasoning":
                    task = AnalogicalReasoningTask(difficulty)
                elif task_type == "spatial_reasoning":
                    task = SpatialReasoningTask(difficulty)
                else:
                    raise ValueError(f"Unknown task type: {task_type}")
                task_instances.append(task)
        
        # Generate samples
        for task in task_instances:
            sample = task.generate_sample()
            samples.append(sample)
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]

class CurriculumScheduler:
    """Curriculum scheduler for progressive learning."""
    
    def __init__(self, 
                 initial_difficulty: int = None,
                 max_difficulty: int = None,
                 difficulty_increment: float = None,
                 competence_threshold: float = None):
        """
        Initialize curriculum scheduler.
        
        Args:
            initial_difficulty: Starting difficulty level
            max_difficulty: Maximum difficulty level
            difficulty_increment: Amount to increase difficulty
            competence_threshold: Threshold for increasing difficulty
        """
        self.config = get_config()
        curriculum_config = self.config.get("curriculum", {})
        difficulty_config = curriculum_config.get("difficulty", {})
        
        self.current_difficulty = initial_difficulty if initial_difficulty is not None else difficulty_config.get("initial", 1)
        self.max_difficulty = max_difficulty if max_difficulty is not None else difficulty_config.get("max", 5)
        self.difficulty_increment = difficulty_increment if difficulty_increment is not None else difficulty_config.get("increment", 0.1)
        self.competence_threshold = competence_threshold if competence_threshold is not None else difficulty_config.get("competence_threshold", 0.8)
        self.max_performance_history = curriculum_config.get("max_performance_history", 10)
        self.performance_window_size = curriculum_config.get("performance_window_size", 5)
        self.performance_history = []
    
    def update_performance(self, accuracy: float):
        """
        Update performance history and adjust difficulty.
        
        Args:
            accuracy: Current accuracy on tasks
        """
        self.performance_history.append(accuracy)
        
        # Keep only recent performance (last N episodes)
        if len(self.performance_history) > self.max_performance_history:
            self.performance_history = self.performance_history[-self.max_performance_history:]
        
        # Check if we should increase difficulty
        if len(self.performance_history) >= self.performance_window_size:
            recent_avg = sum(self.performance_history[-self.performance_window_size:]) / self.performance_window_size
            if recent_avg >= self.competence_threshold:
                self.current_difficulty = min(
                    self.current_difficulty + self.difficulty_increment,
                    self.max_difficulty
                )
    
    def get_current_difficulty(self) -> int:
        """Get current difficulty level."""
        return int(self.current_difficulty)
    
    def get_task_weights(self) -> Dict[str, float]:
        """
        Get task weights based on current curriculum stage.
        
        Returns:
            Dictionary of task weights
        """
        # Adjust task distribution based on difficulty
        if self.current_difficulty <= 2:
            # Early stages: focus on basic reasoning
            return self.config.get_task_distribution("early_stage")
        elif self.current_difficulty <= 3:
            # Middle stages: balanced distribution
            return self.config.get_task_distribution("middle_stage")
        else:
            # Advanced stages: more complex reasoning
            return self.config.get_task_distribution("advanced_stage")

class ReasoningTraceGenerator:
    """Generator for synthetic reasoning traces."""
    
    def __init__(self, 
                 num_samples: int = None,
                 difficulty_range: Tuple[int, int] = None):
        """
        Initialize reasoning trace generator.
        
        Args:
            num_samples: Number of samples to generate
            difficulty_range: Range of difficulty levels
        """
        self.config = get_config()
        curriculum_config = self.config.get("curriculum", {})
        
        self.num_samples = num_samples if num_samples is not None else curriculum_config.get("default_num_samples", 1000)
        self.difficulty_range = difficulty_range if difficulty_range is not None else tuple(curriculum_config.get("default_difficulty_range", [1, 5]))
    
    def generate_traces(self, 
                       output_file: str,
                       curriculum_scheduler: Optional[CurriculumScheduler] = None) -> List[Dict[str, Any]]:
        """
        Generate reasoning traces and save to file.
        
        Args:
            output_file: Path to output file
            curriculum_scheduler: Optional curriculum scheduler
            
        Returns:
            List of generated traces
        """
        traces = []
        
        # Determine difficulty and task distribution
        if curriculum_scheduler:
            difficulty = curriculum_scheduler.get_current_difficulty()
            task_distribution = curriculum_scheduler.get_task_weights()
            difficulty_range = (difficulty, min(difficulty + 1, self.difficulty_range[1]))
        else:
            difficulty_range = self.difficulty_range
            task_distribution = None
        
        # Create dataset
        dataset = SyntheticReasoningDataset(
            num_samples=self.num_samples,
            difficulty_range=difficulty_range,
            task_distribution=task_distribution
        )
        
        # Generate traces
        for sample in dataset:
            trace = {
                "instruction": sample["prompt"],
                "input": "",
                "output": f"{sample['reasoning_chain']}\n\n{sample['final_answer']}",
                "task_type": sample["task_type"],
                "difficulty": sample["difficulty"]
            }
            traces.append(trace)
        
        # Create directory if it doesn't exist (only if there's a directory path)
        import os
        output_dir = os.path.dirname(output_file)
        if output_dir:  # Only create directory if there's a path
            os.makedirs(output_dir, exist_ok=True)
        
        # Save to file
        with open(output_file, 'w') as f:
            for trace in traces:
                f.write(json.dumps(trace) + '\n')
        
        print(f"Generated {len(traces)} reasoning traces and saved to {output_file}")
        return traces
    
    def generate_curriculum(self, 
                          base_output_dir: str,
                          num_stages: int = None) -> List[str]:
        """
        Generate curriculum with progressive difficulty.
        
        Args:
            base_output_dir: Base directory for output files
            num_stages: Number of curriculum stages
            
        Returns:
            List of generated file paths
        """
        # Create base output directory if it doesn't exist
        import os
        os.makedirs(base_output_dir, exist_ok=True)
        
        # Get number of stages from config if not provided
        if num_stages is None:
            demo_config = self.config.get("curriculum.demo", {})
            num_stages = demo_config.get("num_curriculum_stages", 5)
        
        file_paths = []
        scheduler = CurriculumScheduler()
        
        for stage in range(num_stages):
            output_file = f"{base_output_dir}/curriculum_stage_{stage+1}.jsonl"
            
            # Generate traces for this stage
            self.generate_traces(output_file, scheduler)
            file_paths.append(output_file)
            
            # Simulate performance improvement for next stage
            demo_config = self.config.get("curriculum.demo", {})
            base_accuracy = demo_config.get("scheduler_base_accuracy", 0.7)
            accuracy_increment = demo_config.get("scheduler_accuracy_increment", 0.05)
            scheduler.update_performance(base_accuracy + accuracy_increment * stage)  # Progressive improvement
        
        return file_paths

def demonstrate_curriculum_learning():
    """Demonstrate curriculum learning with synthetic reasoning traces."""
    print("Demonstrating curriculum learning with synthetic reasoning traces...")
    
    # Get configuration
    config = get_config()
    demo_config = config.get("curriculum.demo", {})
    num_samples = demo_config.get("num_samples", 100)
    difficulty_range = tuple(demo_config.get("difficulty_range", [1, 3]))
    
    # Create trace generator
    generator = ReasoningTraceGenerator(num_samples=num_samples, difficulty_range=difficulty_range)
    
    # Generate sample traces
    traces = generator.generate_traces("sample_traces.jsonl")
    
    print(f"Generated {len(traces)} sample traces")
    print("Sample trace:")
    print(f"Instruction: {traces[0]['instruction']}")
    print(f"Output preview: {traces[0]['output'][:100]}...")
    
    # Create curriculum
    num_stages = demo_config.get("num_curriculum_stages", 3)
    curriculum_files = generator.generate_curriculum("curriculum", num_stages=num_stages)
    print(f"Generated curriculum with {len(curriculum_files)} stages")
    for i, file_path in enumerate(curriculum_files):
        print(f"  Stage {i+1}: {file_path}")
    
    # Demonstrate curriculum scheduler
    scheduler = CurriculumScheduler()
    print(f"Initial difficulty: {scheduler.get_current_difficulty()}")
    
    # Simulate learning progress
    base_accuracy = demo_config.get("base_accuracy", 0.6)
    accuracy_increment = demo_config.get("accuracy_increment", 0.04)
    
    for episode in range(10):
        accuracy = base_accuracy + accuracy_increment * episode  # Progressive improvement
        scheduler.update_performance(accuracy)
        print(f"Episode {episode+1}: Accuracy = {accuracy:.2f}, Difficulty = {scheduler.get_current_difficulty()}")
    
    # Test the new task types
    print("\nTesting new task types:")
    
    # Test CausalReasoningTask
    causal_task = CausalReasoningTask(difficulty=2)
    causal_sample = causal_task.generate_sample()
    print(f"Causal reasoning task sample: {causal_sample['prompt']}")
    
    # Test AnalogicalReasoningTask
    analogy_task = AnalogicalReasoningTask(difficulty=2)
    analogy_sample = analogy_task.generate_sample()
    print(f"Analogical reasoning task sample: {analogy_sample['prompt']}")
    
    # Test SpatialReasoningTask
    spatial_task = SpatialReasoningTask(difficulty=2)
    spatial_sample = spatial_task.generate_sample()
    print(f"Spatial reasoning task sample: {spatial_sample['prompt']}")
    
    print("Curriculum learning demonstration completed!")


class CausalReasoningTask(SyntheticReasoningTask):
    """Causal reasoning task."""
    
    def __init__(self, difficulty: int = 1):
        super().__init__("causal_reasoning", difficulty)
        causal_config = self.config.get("curriculum.causal_reasoning", {})
        base_min_chain_length = causal_config.get("base_min_chain_length", 2)
        max_chain_length_limit = causal_config.get("max_chain_length_limit", 5)
        self.max_chain_length = min(base_min_chain_length + difficulty, max_chain_length_limit)
        task_data = self.config.get_task_data("causal_reasoning")
        self.causal_patterns = task_data.get("patterns", [
            ("rain", "wet ground"),
            ("studying", "better grades"),
            ("exercise", "fitness"),
            ("sleep", "energy"),
            ("practice", "improvement")
        ])
    
    def generate_sample(self) -> Dict[str, Any]:
        """Generate causal reasoning sample."""
        # Generate causal chain
        chain_length = random.randint(2, self.max_chain_length)
        selected_patterns = random.sample(self.causal_patterns, min(chain_length, len(self.causal_patterns)))
        
        # Build causal chain
        chain_steps = []
        for i, (cause, effect) in enumerate(selected_patterns):
            if i == 0:
                chain_steps.append(f"If {cause} happens, then {effect} occurs.")
            else:
                prev_cause = selected_patterns[i-1][1]  # Effect from previous step
                chain_steps.append(f"Since {prev_cause} occurred, and {cause} results from {prev_cause}, then {effect} occurs.")
        
        # Create question
        first_cause = selected_patterns[0][0]
        last_effect = selected_patterns[-1][1]
        prompt = f"If {first_cause} happens, what is the final outcome?"
        
        reasoning_chain = "\n".join(chain_steps)
        final_answer = f"The final outcome is: {last_effect}"
        
        return {
            "task_type": self.task_type,
            "difficulty": self.difficulty,
            "prompt": prompt,
            "reasoning_chain": reasoning_chain,
            "final_answer": final_answer,
            "ground_truth": last_effect
        }
    
    def get_prompt_template(self) -> str:
        return "Analyze the causal chain: {scenario}"


class AnalogicalReasoningTask(SyntheticReasoningTask):
    """Analogical reasoning task."""
    
    def __init__(self, difficulty: int = 1):
        super().__init__("analogical_reasoning", difficulty)
        analogy_config = self.config.get("curriculum.analogical_reasoning", {})
        base_min_analogies = analogy_config.get("base_min_analogies", 2)
        max_analogies_limit = analogy_config.get("max_analogies_limit", 4)
        self.max_analogies = min(base_min_analogies + difficulty, max_analogies_limit)
        task_data = self.config.get_task_data("analogical_reasoning")
        self.analogies = task_data.get("analogies", [
            ("eye", "camera", "see", "capture"),
            ("ear", "microphone", "hear", "record"),
            ("heart", "pump", "pump blood", "pump liquid"),
            ("brain", "computer", "think", "process"),
            ("stomach", "mixer", "digest food", "mix ingredients")
        ])
    
    def generate_sample(self) -> Dict[str, Any]:
        """Generate analogical reasoning sample."""
        # Select analogies
        num_analogies = random.randint(2, self.max_analogies)
        selected_analogies = random.sample(self.analogies, min(num_analogies, len(self.analogies)))
        
        # Build analogy explanation
        analogy_explanations = []
        for item1, item2, function1, function2 in selected_analogies:
            explanation = f"Just as a {item1} {function1}, a {item2} {function2}."
            analogy_explanations.append(explanation)
        
        # Create question from the last analogy
        target_item1, target_item2, target_function1, target_function2 = selected_analogies[-1]
        prompt = f"If {target_item1} is like {target_item2}, and {target_item1} can {target_function1}, what can {target_item2} do?"
        
        reasoning_chain = "\n".join(analogy_explanations)
        final_answer = f"Therefore, {target_item2} can {target_function2}."
        
        return {
            "task_type": self.task_type,
            "difficulty": self.difficulty,
            "prompt": prompt,
            "reasoning_chain": reasoning_chain,
            "final_answer": final_answer,
            "ground_truth": target_function2
        }
    
    def get_prompt_template(self) -> str:
        return "Complete the analogy: {analogy}"


class SpatialReasoningTask(SyntheticReasoningTask):
    """Spatial reasoning task."""
    
    def __init__(self, difficulty: int = 1):
        super().__init__("spatial_reasoning", difficulty)
        spatial_config = self.config.get("curriculum.spatial_reasoning", {})
        base_min_objects = spatial_config.get("base_min_objects", 3)
        max_objects_limit = spatial_config.get("max_objects_limit", 6)
        self.max_objects = min(base_min_objects + difficulty, max_objects_limit)
        task_data = self.config.get_task_data("spatial_reasoning")
        self.shapes = task_data.get("shapes", ["square", "circle", "triangle", "rectangle", "pentagon"])
        self.positions = task_data.get("positions", ["above", "below", "left of", "right of", "inside", "outside"])
    
    def generate_sample(self) -> Dict[str, Any]:
        """Generate spatial reasoning sample."""
        # Generate objects and positions
        num_objects = random.randint(3, self.max_objects)
        objects = random.sample(self.shapes, min(num_objects, len(self.shapes)))
        
        # Create spatial relationships
        relationships = []
        for i in range(1, len(objects)):
            position = random.choice(self.positions)
            relationships.append(f"The {objects[i]} is {position} the {objects[i-1]}.")
        
        # Build spatial description
        description = f"Imagine {objects[0]} is in the center.\n" + "\n".join(relationships)
        
        # Create question about relative positions
        obj1_idx = random.randint(0, len(objects)-1)
        obj2_idx = random.randint(0, len(objects)-1)
        while obj1_idx == obj2_idx:
            obj2_idx = random.randint(0, len(objects)-1)
        
        obj1 = objects[obj1_idx]
        obj2 = objects[obj2_idx]
        prompt = f"Based on the description, where is the {obj1} in relation to the {obj2}?"
        
        reasoning_chain = description
        final_answer = f"The {obj1} is positioned relative to the {obj2} according to the given relationships."
        
        return {
            "task_type": self.task_type,
            "difficulty": self.difficulty,
            "prompt": prompt,
            "reasoning_chain": reasoning_chain,
            "final_answer": final_answer,
            "ground_truth": "relative position based on relationships"
        }
    
    def get_prompt_template(self) -> str:
        return "Determine the spatial relationship: {description}"


if __name__ == "__main__":
    demonstrate_curriculum_learning()