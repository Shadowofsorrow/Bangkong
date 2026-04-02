#!/usr/bin/env python3
"""
Data cleaning script for Bangkong LLM Training System
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


def is_unwanted_file(filename: str) -> bool:
    """
    Check if a file is unwanted for training.
    
    Args:
        filename: Name of the file
        
    Returns:
        True if file is unwanted, False otherwise
    """
    unwanted_patterns = [
        'readme', 'license', 'copyright', 'changelog', 'contributing',
        'authors', 'contributors', 'thanks', 'credits', 'notice',
        'todo', 'roadmap', 'manifest', 'requirements', 'setup',
        'test', 'tests', 'spec', 'specs', 'example', 'examples',
        'sample', 'samples', 'demo', 'demos', 'benchmark', 'benchmarks'
    ]
    
    filename_lower = filename.lower()
    file_ext = Path(filename).suffix.lower()
    
    # Check for unwanted patterns in filename
    for pattern in unwanted_patterns:
        if pattern in filename_lower:
            return True
    
    # Check for unwanted file extensions
    unwanted_extensions = {'.md', '.rst', '.txt'}  # We'll be more selective with text files
    if file_ext in unwanted_extensions:
        # For text files, check if they're likely documentation
        doc_indicators = ['readme', 'license', 'changelog', 'contributing']
        for indicator in doc_indicators:
            if indicator in filename_lower:
                return True
    
    return False


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
        
        # For text-based files, read directly
        text_extensions = {
            '.txt', '.md', '.rst', '.log', '.py', '.js', '.java', '.cpp', '.c', '.h',
            '.html', '.css', '.xml', '.php', '.rb', '.go', '.rs', '.json', '.yaml', '.yml'
        }
        
        if file_ext in text_extensions:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        
        # For other file types that might contain text, try to extract
        else:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
                
    except Exception as e:
        print(f"  Warning: Failed to extract text from {file_path}: {e}")
        return ""


def process_text_files(file_paths: List[str]) -> List[Dict]:
    """
    Process text files and return training samples.
    
    Args:
        file_paths: List of text file paths
        
    Returns:
        List of training samples
    """
    training_samples = []
    
    for file_path in file_paths:
        try:
            filename = Path(file_path).name
            
            # Skip unwanted files
            if is_unwanted_file(filename):
                print(f"  Skipping unwanted file: {filename}")
                continue
            
            # Extract text content
            content = extract_text_content(file_path)
            
            if not content.strip():
                continue
            
            # Split content into chunks for training samples
            lines = [line.strip() for line in content.splitlines() if line.strip()]
            
            # Remove excessive padding/whitespace
            # Filter out lines that are mostly whitespace or very short
            filtered_lines = [line for line in lines if len(line) > 3 and not line.isspace()]
            
            if not filtered_lines:
                continue
            
            # Create training samples
            chunk_size = 15  # Lines per sample
            
            for i in range(0, len(filtered_lines), chunk_size):
                chunk_lines = filtered_lines[i:i + chunk_size]
                if len(chunk_lines) >= 3:  # Minimum 3 lines for a meaningful sample
                    sample_text = '\n'.join(chunk_lines)
                    # Remove excessive padding
                    sample_text = '\n'.join(line.strip() for line in sample_text.splitlines() if line.strip())
                    
                    if len(sample_text.strip()) > 10:  # Minimum content length
                        sample = {
                            "text": sample_text,
                            "source": f"{Path(file_path).stem}_chunk_{i//chunk_size}",
                            "file_type": "text",
                            "metadata": {
                                "original_file": filename,
                                "chunk_index": i//chunk_size,
                                "line_count": len(chunk_lines),
                                "char_count": len(sample_text)
                            }
                        }
                        training_samples.append(sample)
            
            if training_samples:
                print(f"  Processed: {filename} -> {len(range(0, len(filtered_lines), chunk_size))} samples")
            
        except Exception as e:
            print(f"  Warning: Failed to process {file_path}: {e}")
    
    return training_samples


def process_code_files(file_paths: List[str]) -> List[Dict]:
    """
    Process code files and return training samples.
    
    Args:
        file_paths: List of code file paths
        
    Returns:
        List of training samples
    """
    training_samples = []
    
    for file_path in file_paths:
        try:
            filename = Path(file_path).name
            
            # Skip unwanted files
            if is_unwanted_file(filename):
                print(f"  Skipping unwanted file: {filename}")
                continue
            
            # Extract code content
            content = extract_text_content(file_path)
            
            if not content.strip():
                continue
            
            # Split content into functions/classes for training samples
            lines = [line.rstrip() for line in content.splitlines()]
            
            # Remove excessive padding
            lines = [line for line in lines if line.strip() or line == '']
            
            # Simple approach: split by functions or classes
            current_block = []
            block_name = "unknown"
            block_index = 0
            
            for line in lines:
                # Detect function or class definitions
                stripped_line = line.strip()
                if stripped_line.startswith(('def ', 'class ', 'function ', 'public ', 'private ', 'protected ', 'fn ', 'impl ')):
                    if current_block and len([l for l in current_block if l.strip()]) > 3:
                        # Save previous block as a sample
                        sample_text = '\n'.join(current_block)
                        # Remove excessive empty lines
                        sample_text = '\n'.join(line for line in sample_text.splitlines() if line.strip() or (line == '' and current_block.index(line) > 0 and current_block[current_block.index(line)-1].strip()))
                        
                        sample = {
                            "text": sample_text.strip(),
                            "source": f"{Path(file_path).stem}_{block_name}_{block_index}",
                            "file_type": "code",
                            "metadata": {
                                "original_file": filename,
                                "block_type": block_name,
                                "block_index": block_index,
                                "line_count": len([l for l in current_block if l.strip()]),
                                "char_count": len(sample_text.strip())
                            }
                        }
                        training_samples.append(sample)
                        block_index += 1
                    
                    # Start new block
                    block_name = stripped_line.split()[1].split('(')[0] if '(' in stripped_line else stripped_line.split()[1]
                    current_block = [line]
                else:
                    current_block.append(line)
            
            # Save last block
            if current_block and len([l for l in current_block if l.strip()]) > 3:
                sample_text = '\n'.join(current_block)
                # Remove excessive empty lines
                sample_text = '\n'.join(line for line in sample_text.splitlines() if line.strip() or (line == '' and current_block.index(line) > 0 and current_block[current_block.index(line)-1].strip()))
                
                sample = {
                    "text": sample_text.strip(),
                    "source": f"{Path(file_path).stem}_{block_name}_{block_index}",
                    "file_type": "code",
                    "metadata": {
                        "original_file": filename,
                        "block_type": block_name,
                        "block_index": block_index,
                        "line_count": len([l for l in current_block if l.strip()]),
                        "char_count": len(sample_text.strip())
                    }
                }
                training_samples.append(sample)
            
            if block_index > 0 or (current_block and len([l for l in current_block if l.strip()]) > 3):
                print(f"  Processed: {filename} -> {block_index + 1} samples")
            
        except Exception as e:
            print(f"  Warning: Failed to process {file_path}: {e}")
    
    return training_samples


def process_document_files(file_paths: List[str]) -> List[Dict]:
    """
    Process document files and return training samples.
    
    Args:
        file_paths: List of document file paths
        
    Returns:
        List of training samples
    """
    training_samples = []
    
    for file_path in file_paths:
        try:
            filename = Path(file_path).name
            
            # Skip unwanted files
            if is_unwanted_file(filename):
                print(f"  Skipping unwanted file: {filename}")
                continue
            
            # Extract text content (simplified approach)
            content = extract_text_content(file_path)
            
            if not content.strip():
                continue
            
            # Split content into paragraphs for training samples
            paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
            
            # Remove excessive padding
            paragraphs = [p for p in paragraphs if len(p) > 10 and not p.isspace()]
            
            for i, paragraph in enumerate(paragraphs):
                if len(paragraph) > 20:  # Minimum length for a meaningful paragraph
                    sample = {
                        "text": paragraph,
                        "source": f"{Path(file_path).stem}_paragraph_{i}",
                        "file_type": "document",
                        "metadata": {
                            "original_file": filename,
                            "paragraph_index": i,
                            "char_count": len(paragraph)
                        }
                    }
                    training_samples.append(sample)
            
            if paragraphs:
                print(f"  Processed: {filename} -> {len(paragraphs)} samples")
            
        except Exception as e:
            print(f"  Warning: Failed to process {file_path}: {e}")
    
    return training_samples


def process_data_files(file_paths: List[str]) -> List[Dict]:
    """
    Process data files and return training samples.
    
    Args:
        file_paths: List of data file paths
        
    Returns:
        List of training samples
    """
    training_samples = []
    
    for file_path in file_paths:
        try:
            filename = Path(file_path).name
            
            # Skip unwanted files
            if is_unwanted_file(filename):
                print(f"  Skipping unwanted file: {filename}")
                continue
            
            # Extract content
            content = extract_text_content(file_path)
            
            if not content.strip():
                continue
            
            # For structured data, create samples based on records/rows
            lines = [line.strip() for line in content.splitlines() if line.strip()]
            
            # If it looks like CSV or similar, treat each line as a sample
            if len(lines) > 1:
                # Skip header line
                for i, line in enumerate(lines[1:], 1):
                    if len(line) > 10:  # Minimum length
                        sample = {
                            "text": line,
                            "source": f"{Path(file_path).stem}_record_{i}",
                            "file_type": "data",
                            "metadata": {
                                "original_file": filename,
                                "record_index": i,
                                "char_count": len(line)
                            }
                        }
                        training_samples.append(sample)
            
            if len(lines) > 1:
                print(f"  Processed: {filename} -> {len(lines)-1} samples")
            
        except Exception as e:
            print(f"  Warning: Failed to process {file_path}: {e}")
    
    return training_samples


def save_training_samples(samples: List[Dict], output_path: str):
    """
    Save training samples in JSONL format.
    
    Args:
        samples: List of training samples
        output_path: Path to save the samples
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"Saved {len(samples)} training samples to {output_file}")


def clean_and_process_data(organized_path: str, processed_path: str) -> int:
    """
    Clean and process organized data files.
    
    Args:
        organized_path: Path to organized data directory
        processed_path: Path to processed data directory
        
    Returns:
        Number of samples generated
    """
    path_manager = PathManager()
    organized_dir = path_manager.resolve_path(organized_path)
    processed_dir = path_manager.resolve_path(processed_path)
    
    if not organized_dir.exists():
        print(f"Error: Organized data directory {organized_path} does not exist")
        return 0
    
    # Collect all files by category
    categories = ['text', 'code', 'document', 'data']
    all_samples = []
    
    print("Processing organized files...")
    
    for category in categories:
        category_dir = organized_dir / category
        if category_dir.exists():
            print(f"\nProcessing {category} files...")
            file_paths = [str(f) for f in category_dir.iterdir() if f.is_file()]
            
            if file_paths:
                if category == 'text':
                    samples = process_text_files(file_paths)
                elif category == 'code':
                    samples = process_code_files(file_paths)
                elif category == 'document':
                    samples = process_document_files(file_paths)
                elif category == 'data':
                    samples = process_data_files(file_paths)
                else:
                    samples = []
                
                all_samples.extend(samples)
                print(f"  Generated {len(samples)} samples from {category} files")
    
    if not all_samples:
        print("No samples generated. Check if organized data exists.")
        return 0
    
    # Save all samples
    samples_file = processed_dir / "training_samples.jsonl"
    save_training_samples(all_samples, str(samples_file))
    
    # Also save a sample dataset
    sample_size = min(100, len(all_samples))
    sample_samples = random.sample(all_samples, sample_size)
    sample_file = processed_dir / "sample_training_data.jsonl"
    save_training_samples(sample_samples, str(sample_file))
    
    print(f"\nSample dataset saved to {sample_file} ({sample_size} samples)")
    
    return len(all_samples)


def main():
    """Main data cleaning function."""
    parser = argparse.ArgumentParser(description="Clean and process organized data for Bangkong LLM Training")
    parser.add_argument("--organized-path", type=str, default="./data/organized", 
                       help="Path to organized data directory")
    parser.add_argument("--processed-path", type=str, default="./data/processed",
                       help="Path to processed data directory")
    
    args = parser.parse_args()
    
    print("Bangkong LLM Training System - Data Cleaning and Processing")
    print("=" * 58)
    print()
    
    print(f"Organized data path: {args.organized_path}")
    print(f"Processed data path: {args.processed_path}")
    print()
    
    # Clean and process data
    sample_count = clean_and_process_data(args.organized_path, args.processed_path)
    
    print()
    print("Data cleaning and processing completed successfully!")
    print(f"Generated {sample_count} training samples.")
    print("Training-ready JSONL files are now available in the processed directory.")
    print()
    print("Output files:")
    print(f"  {args.processed_path}/training_samples.jsonl (all samples)")
    print(f"  {args.processed_path}/sample_training_data.jsonl (100 random samples)")


if __name__ == "__main__":
    main()