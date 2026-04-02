#!/usr/bin/env python3
"""
Data organization script for Bangkong LLM Training System
"""

import argparse
import sys
import os
import shutil
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
    code_files = {'.py', '.js', '.java', '.cpp', '.c', '.h', '.html', '.css', '.xml', '.php', '.rb', '.go', '.rs'}
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


def organize_files(raw_data_path: str, organized_path: str) -> Dict[str, int]:
    """
    Organize files from raw data directory into categorized folders.
    
    Args:
        raw_data_path: Path to raw data directory
        organized_path: Path to organized data directory
        
    Returns:
        Dictionary with file type counts
    """
    # Ensure directories exist
    path_manager = PathManager()
    raw_path = path_manager.resolve_path(raw_data_path)
    organized_dir = path_manager.resolve_path(organized_path)
    
    # Create organized directory structure
    categories = ['text', 'code', 'document', 'data', 'image', 'unknown']
    for category in categories:
        (organized_dir / category).mkdir(parents=True, exist_ok=True)
    
    # Count organized files
    file_counts = {category: 0 for category in categories}
    
    # Get all files in the raw data directory (including subdirectories)
    if not raw_path.exists():
        print(f"Warning: Raw data directory {raw_data_path} does not exist")
        return file_counts
    
    print(f"Scanning {raw_path} for files...")
    
    # Iterate through all files
    for file_path in raw_path.rglob('*'):
        if file_path.is_file():
            # Skip hidden files and directories
            if any(part.startswith('.') for part in file_path.parts):
                continue
                
            file_type = get_file_type(str(file_path))
            target_dir = organized_dir / file_type
            
            try:
                # Copy file to organized directory
                target_path = target_dir / file_path.name
                # Handle duplicate filenames
                counter = 1
                original_target_path = target_path
                while target_path.exists():
                    stem = original_target_path.stem
                    suffix = original_target_path.suffix
                    target_path = target_dir / f"{stem}_{counter}{suffix}"
                    counter += 1
                
                shutil.copy2(file_path, target_path)
                file_counts[file_type] += 1
                print(f"  Organized: {file_path.name} -> {file_type}/")
                
            except Exception as e:
                print(f"  Warning: Failed to organize {file_path}: {e}")
    
    # Print organization summary
    print("\nOrganization summary:")
    total_files = 0
    for category, count in file_counts.items():
        if count > 0:
            print(f"  {category}: {count} files")
            total_files += count
    
    print(f"\nTotal files organized: {total_files}")
    
    return file_counts


def main():
    """Main data organization function."""
    parser = argparse.ArgumentParser(description="Organize raw data for Bangkong LLM Training")
    parser.add_argument("--raw-path", type=str, default="./data/raw", 
                       help="Path to raw data directory")
    parser.add_argument("--organized-path", type=str, default="./data/organized",
                       help="Path to organized data directory")
    
    args = parser.parse_args()
    
    print("Bangkong LLM Training System - Data Organization")
    print("=" * 50)
    print()
    
    print(f"Raw data path: {args.raw_path}")
    print(f"Organized data path: {args.organized_path}")
    print()
    
    # Check if raw data exists
    path_manager = PathManager()
    raw_path = path_manager.resolve_path(args.raw_path)
    
    if not raw_path.exists():
        print("Creating raw data directory...")
        raw_path.mkdir(parents=True, exist_ok=True)
        
        # Create sample raw data if none exists
        sample_files = [
            ("sample_text1.txt", "This is a sample text file for training.\nIt contains multiple lines of text.\nThis is useful for language model training."),
            ("sample_text2.md", "# Sample Markdown\n\nThis is a sample markdown file.\n\n## Section\n\nContent here."),
            ("sample_code.py", "# This is a sample Python file\nimport os\n\ndef hello_world():\n    print('Hello, World!')\n\nif __name__ == '__main__':\n    hello_world()"),
            ("README.md", "# Project README\n\nThis is a README file that might not be wanted for training."),
            ("LICENSE", "MIT License\n\nCopyright (c) 2025 Bangkong AI Team"),
        ]
        
        for filename, content in sample_files:
            with open(raw_path / filename, 'w', encoding='utf-8') as f:
                f.write(content)
        
        print("Created sample raw data files for demonstration.")
        print()
    
    # Organize files
    file_counts = organize_files(args.raw_path, args.organized_path)
    
    print()
    print("Data organization completed successfully!")
    print("Files are now organized in categorized folders.")
    print("You can review and filter files in the organized directory")
    print("before proceeding to the cleaning step.")
    print()
    print("Organized data structure:")
    print(f"  {args.organized_path}/")
    for category, count in file_counts.items():
        if count > 0:
            print(f"    ├── {category}/ ({count} files)")


if __name__ == "__main__":
    main()