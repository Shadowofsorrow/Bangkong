#!/usr/bin/env python3
"""
Data processing script for Bangkong LLM Training System
"""

import argparse
import sys
import os
import json
import random
from pathlib import Path
from typing import List, Dict
import mimetypes

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bangkong.utils.path_manager import PathManager


def get_file_type(file_path: str) -> str:
    """
    Determine file type based on extension.
    
    Args:
        file_path: Path to the file
        
    Returns:
        File type category
    """
    file_ext = Path(file_path).suffix.lower()
    
    # Categorize by file type
    text_files = {'.txt', '.md', '.rst', '.log'}
    code_files = {'.py', '.js', '.java', '.cpp', '.c', '.h', '.html', '.css', '.xml'}
    document_files = {'.pdf', '.doc', '.docx', '.odt', '.rtf'}
    data_files = {'.csv', '.json', '.yaml', '.yml', '.xml', '.tsv'}
    image_files = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.svg'}
    
    if file_ext in text_files:
        return 'text'
    elif file_ext in code_files:
        return 'code'
    elif file_ext in document_files:
        return 'document'
    elif file_ext in data_files:
        return 'data'
    elif file_ext in image_files:
        return 'image'
    else:
        # Try to guess based on MIME type
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type:
            if mime_type.startswith('text/'):
                return 'text'
            elif mime_type.startswith('image/'):
                return 'image'
            elif mime_type.startswith('application/'):
                return 'document'
        return 'unknown'


def categorize_files(raw_data_path: str) -> Dict[str, List[str]]:
    """
    Categorize files in the raw data directory by type.
    
    Args:
        raw_data_path: Path to raw data directory
        
    Returns:
        Dictionary with file types as keys and file paths as values
    """
    categorized_files = {
        'text': [],
        'code': [],
        'document': [],
        'data': [],
        'image': [],
        'unknown': []
    }
    
    # Get all files in the raw data directory
    raw_path = Path(raw_data_path)
    if not raw_path.exists():
        print(f"Warning: Raw data directory {raw_data_path} does not exist")
        return categorized_files
    
    # Iterate through all files
    for file_path in raw_path.rglob('*'):
        if file_path.is_file():
            file_type = get_file_type(str(file_path))
            categorized_files[file_type].append(str(file_path))
    
    # Print categorization summary
    print("File categorization summary:")
    for category, files in categorized_files.items():
        if files:
            print(f"  {category}: {len(files)} files")
    
    return categorized_files


def extract_text_content(file_path: str) -> str:
    """
    Extract text content from various file types.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Extracted text content
    """
    try:
        file_ext = Path(file_path).suffix.lower()
        
        # For text files, read directly
        if file_ext in {'.txt', '.md', '.rst', '.log', '.py', '.js', '.java', '.cpp', '.c', '.h', '.html', '.css', '.xml'}:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        
        # For other file types that might contain text, try to extract
        else:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
                
    except Exception as e:
        print(f"  Warning: Failed to extract text from {file_path}: {e}")
        return ""


def process_text_files(file_paths: List[str], processed_path: str) -> List[Dict]:
    """
    Process text files (cleaning, tokenization, etc.) and return training samples.
    
    Args:
        file_paths: List of text file paths
        processed_path: Path to save processed files
        
    Returns:
        List of training samples
    """
    training_samples = []
    processed_dir = Path(processed_path) / 'text'
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    for file_path in file_paths:
        try:
            # Extract text content
            content = extract_text_content(file_path)
            
            if not content.strip():
                continue
            
            # Split content into chunks for training samples
            lines = [line.strip() for line in content.splitlines() if line.strip()]
            
            # Create training samples
            filename = Path(file_path).stem
            chunk_size = 10  # Lines per sample
            
            for i in range(0, len(lines), chunk_size):
                chunk_lines = lines[i:i + chunk_size]
                if len(chunk_lines) >= 3:  # Minimum 3 lines for a meaningful sample
                    sample_text = '\n'.join(chunk_lines)
                    sample = {
                        "text": sample_text,
                        "source": f"{filename}_chunk_{i//chunk_size}",
                        "file_type": "text",
                        "metadata": {
                            "original_file": Path(file_path).name,
                            "chunk_index": i//chunk_size,
                            "line_count": len(chunk_lines)
                        }
                    }
                    training_samples.append(sample)
            
            print(f"  Processed: {Path(file_path).name} -> {len(range(0, len(lines), chunk_size))} samples")
            
        except Exception as e:
            print(f"  Warning: Failed to process {file_path}: {e}")
    
    return training_samples


def process_code_files(file_paths: List[str], processed_path: str) -> List[Dict]:
    """
    Process code files and return training samples.
    
    Args:
        file_paths: List of code file paths
        processed_path: Path to save processed files
        
    Returns:
        List of training samples
    """
    training_samples = []
    processed_dir = Path(processed_path) / 'code'
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    for file_path in file_paths:
        try:
            # Extract code content
            content = extract_text_content(file_path)
            
            if not content.strip():
                continue
            
            # Split content into functions/classes for training samples
            lines = [line.rstrip() for line in content.splitlines()]
            
            # Simple approach: split by functions or classes
            filename = Path(file_path).stem
            current_block = []
            block_name = "unknown"
            block_index = 0
            
            for line in lines:
                # Detect function or class definitions
                if line.strip().startswith(('def ', 'class ', 'function ', 'public ', 'private ', 'protected ')):
                    if current_block and len(current_block) > 3:
                        # Save previous block as a sample
                        sample_text = '\n'.join(current_block)
                        sample = {
                            "text": sample_text,
                            "source": f"{filename}_{block_name}_{block_index}",
                            "file_type": "code",
                            "metadata": {
                                "original_file": Path(file_path).name,
                                "block_type": block_name,
                                "block_index": block_index,
                                "line_count": len(current_block)
                            }
                        }
                        training_samples.append(sample)
                        block_index += 1
                    
                    # Start new block
                    block_name = line.strip().split()[1].split('(')[0] if '(' in line else line.strip().split()[1]
                    current_block = [line]
                else:
                    current_block.append(line)
            
            # Save last block
            if current_block and len(current_block) > 3:
                sample_text = '\n'.join(current_block)
                sample = {
                    "text": sample_text,
                    "source": f"{filename}_{block_name}_{block_index}",
                    "file_type": "code",
                    "metadata": {
                        "original_file": Path(file_path).name,
                        "block_type": block_name,
                        "block_index": block_index,
                        "line_count": len(current_block)
                    }
                }
                training_samples.append(sample)
            
            print(f"  Processed: {Path(file_path).name} -> {block_index + 1} samples")
            
        except Exception as e:
            print(f"  Warning: Failed to process {file_path}: {e}")
    
    return training_samples


def process_document_files(file_paths: List[str], processed_path: str) -> List[Dict]:
    """
    Process document files and return training samples.
    
    Args:
        file_paths: List of document file paths
        processed_path: Path to save processed files
        
    Returns:
        List of training samples
    """
    training_samples = []
    processed_dir = Path(processed_path) / 'documents'
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    for file_path in file_paths:
        try:
            # Extract text content (simplified approach)
            content = extract_text_content(file_path)
            
            if not content.strip():
                continue
            
            # Split content into paragraphs for training samples
            paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
            
            filename = Path(file_path).stem
            for i, paragraph in enumerate(paragraphs):
                if len(paragraph) > 50:  # Minimum length for a meaningful paragraph
                    sample = {
                        "text": paragraph,
                        "source": f"{filename}_paragraph_{i}",
                        "file_type": "document",
                        "metadata": {
                            "original_file": Path(file_path).name,
                            "paragraph_index": i,
                            "char_count": len(paragraph)
                        }
                    }
                    training_samples.append(sample)
            
            print(f"  Processed: {Path(file_path).name} -> {len(paragraphs)} samples")
            
        except Exception as e:
            print(f"  Warning: Failed to process {file_path}: {e}")
    
    return training_samples


def process_data_files(file_paths: List[str], processed_path: str) -> List[Dict]:
    """
    Process data files and return training samples.
    
    Args:
        file_paths: List of data file paths
        processed_path: Path to save processed files
        
    Returns:
        List of training samples
    """
    training_samples = []
    processed_dir = Path(processed_path) / 'data'
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    for file_path in file_paths:
        try:
            # Extract content
            content = extract_text_content(file_path)
            
            if not content.strip():
                continue
            
            # For structured data, create samples based on records/rows
            lines = [line.strip() for line in content.splitlines() if line.strip()]
            
            filename = Path(file_path).stem
            # If it looks like CSV or similar, treat each line as a sample
            if len(lines) > 1:
                for i, line in enumerate(lines[1:], 1):  # Skip header
                    if len(line) > 10:  # Minimum length
                        sample = {
                            "text": line,
                            "source": f"{filename}_record_{i}",
                            "file_type": "data",
                            "metadata": {
                                "original_file": Path(file_path).name,
                                "record_index": i,
                                "char_count": len(line)
                            }
                        }
                        training_samples.append(sample)
            
            print(f"  Processed: {Path(file_path).name} -> {len(lines)-1} samples")
            
        except Exception as e:
            print(f"  Warning: Failed to process {file_path}: {e}")
    
    return training_samples


def save_training_samples(samples: List[Dict], output_path: str, format: str = "jsonl"):
    """
    Save training samples in the specified format.
    
    Args:
        samples: List of training samples
        output_path: Path to save the samples
        format: Output format (jsonl, json, txt)
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    if format == "jsonl":
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    elif format == "json":
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)
    
    elif format == "txt":
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(sample["text"] + '\n' + '='*50 + '\n')
    
    print(f"Saved {len(samples)} training samples to {output_file}")


def create_sample_dataset(all_samples: List[Dict], sample_path: str, sample_size: int = 100, format: str = "jsonl"):
    """
    Create a sample dataset from processed files for training.
    
    Args:
        all_samples: List of all processed samples
        sample_path: Path to save sample dataset
        sample_size: Number of samples to include
        format: Output format
    """
    if not all_samples:
        print("No samples to create dataset from")
        return
    
    # Select a sample
    selected_samples = random.sample(all_samples, min(sample_size, len(all_samples)))
    
    # Save in requested format
    save_training_samples(selected_samples, sample_path, format)
    
    print(f"Created sample dataset with {len(selected_samples)} samples in {sample_path}")


def main():
    """Main data processing function."""
    parser = argparse.ArgumentParser(description="Process raw data for Bangkong LLM Training")
    parser.add_argument("--raw-path", type=str, default="./data/raw", 
                       help="Path to raw data directory")
    parser.add_argument("--processed-path", type=str, default="./data/processed",
                       help="Path to save processed data")
    parser.add_argument("--sample-path", type=str, default="./data/sample/training_samples.jsonl",
                       help="Path to save sample dataset")
    parser.add_argument("--sample-size", type=int, default=100,
                       help="Number of samples to include in sample dataset")
    parser.add_argument("--format", type=str, default="jsonl", choices=["jsonl", "json", "txt"],
                       help="Output format for training samples")
    
    args = parser.parse_args()
    
    print("Bangkong LLM Training System - Data Processing Pipeline")
    print("=" * 55)
    print()
    
    # Ensure directories exist
    path_manager = PathManager()
    raw_path = path_manager.resolve_path(args.raw_path)
    processed_path = path_manager.resolve_path(args.processed_path)
    sample_path = path_manager.resolve_path(args.sample_path)
    
    print(f"Raw data path: {raw_path}")
    print(f"Processed data path: {processed_path}")
    print(f"Sample dataset path: {sample_path}")
    print(f"Output format: {args.format}")
    print()
    
    # Check if raw data exists
    if not raw_path.exists():
        print("Creating raw data directory...")
        raw_path.mkdir(parents=True, exist_ok=True)
        
        # Create sample raw data if none exists
        sample_files = [
            ("sample_text1.txt", "This is a sample text file for training.\nIt contains multiple lines of text.\nThis is useful for language model training.\n\nThe quick brown fox jumps over the lazy dog.\nMachine learning models require diverse training data.\nNatural language processing benefits from large datasets."),
            ("sample_text2.txt", "Another sample text file with different content.\nThis demonstrates variety in the training data.\nMachine learning models benefit from diverse data.\n\nDeep learning has revolutionized artificial intelligence.\nNeural networks can learn complex patterns.\nTransformer architectures have enabled large language models."),
            ("sample_code.py", "# This is a sample Python file\nimport os\nimport sys\n\ndef hello_world():\n    \"\"\"Print a greeting message\"\"\"\n    print('Hello, World!')\n    return True\n\nclass DataProcessor:\n    def __init__(self, data):\n        self.data = data\n    \n    def process(self):\n        return [item.strip() for item in self.data if item.strip()]\n\nif __name__ == '__main__':\n    hello_world()"),
        ]
        
        for filename, content in sample_files:
            with open(raw_path / filename, 'w', encoding='utf-8') as f:
                f.write(content)
        
        print("Created sample raw data files for demonstration.")
        print()
    
    # Categorize files
    print("Step 1: Categorizing files by type...")
    categorized_files = categorize_files(str(raw_path))
    print()
    
    # Process files by category and collect training samples
    all_samples = []
    
    print("Step 2: Processing files...")
    
    # Process text files
    if categorized_files['text']:
        print("Processing text files...")
        samples = process_text_files(categorized_files['text'], str(processed_path))
        all_samples.extend(samples)
        print(f"  Generated {len(samples)} samples from text files")
    
    # Process code files
    if categorized_files['code']:
        print("Processing code files...")
        samples = process_code_files(categorized_files['code'], str(processed_path))
        all_samples.extend(samples)
        print(f"  Generated {len(samples)} samples from code files")
    
    # Process document files
    if categorized_files['document']:
        print("Processing document files...")
        samples = process_document_files(categorized_files['document'], str(processed_path))
        all_samples.extend(samples)
        print(f"  Generated {len(samples)} samples from document files")
    
    # Process data files
    if categorized_files['data']:
        print("Processing data files...")
        samples = process_data_files(categorized_files['data'], str(processed_path))
        all_samples.extend(samples)
        print(f"  Generated {len(samples)} samples from data files")
    
    print()
    print(f"Total samples generated: {len(all_samples)}")
    
    # Handle unknown files
    if categorized_files['unknown']:
        print(f"Warning: {len(categorized_files['unknown'])} files with unknown types were not processed")
    
    # Create sample dataset
    print()
    print("Step 3: Creating sample dataset for training...")
    create_sample_dataset(all_samples, str(sample_path), args.sample_size, args.format)
    
    # Also save all samples
    all_samples_path = Path(args.processed_path) / f"all_samples.{args.format}"
    save_training_samples(all_samples, str(all_samples_path), args.format)
    
    print()
    print("Data processing pipeline completed successfully!")
    print("Training samples are now available for model training.")
    print(f"Sample dataset: {sample_path}")
    print(f"All samples: {all_samples_path}")


if __name__ == "__main__":
    main()