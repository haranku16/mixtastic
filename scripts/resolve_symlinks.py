#!/usr/bin/env python3
"""
Script to resolve symbolic links in the processed directory's train/test splits,
replacing each link with a copy of the actual linked file.
"""

import os
import shutil
import argparse
from pathlib import Path


def is_hidden(path):
    """Check if a path (file or directory) is hidden."""
    return any(part.startswith('.') for part in Path(path).parts)


def resolve_links(processed_dir='data/processed', dry_run=False):
    """
    Resolve all symbolic links in the train/test splits by replacing them with actual copies.
    
    Args:
        processed_dir: Path to the processed directory
        dry_run: If True, only print what would be done without making changes
    """
    base_dir = Path(processed_dir)
    
    # Make sure the processed directory exists
    if not base_dir.exists():
        print(f"Error: {base_dir} directory does not exist")
        return

    # Process train and test directories
    for split in ['train', 'test']:
        split_dir = base_dir / split
        
        if not split_dir.exists():
            print(f"Skipping {split_dir} as it does not exist")
            continue
            
        print(f"Processing {split_dir}...")
        
        # Walk through the directory tree
        for root, dirs, files in os.walk(split_dir):
            root_path = Path(root)
            
            # Skip hidden directories
            dirs[:] = [d for d in dirs if not is_hidden(root_path / d)]
            
            for file in files:
                file_path = root_path / file
                
                # Skip hidden files
                if is_hidden(file_path):
                    continue
                
                # Check if it's a symbolic link
                if file_path.is_symlink():
                    target_path = file_path.resolve()
                    print(f"Found symlink: {file_path} -> {target_path}")
                    
                    if not dry_run:
                        # Remove the symlink
                        file_path.unlink()
                        
                        # Copy the actual file
                        shutil.copy2(target_path, file_path)
                        print(f"  Replaced with copy of {target_path}")
                    else:
                        print(f"  Would replace with copy of {target_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resolve symbolic links in the processed directory's train/test splits")
    parser.add_argument("--dir", default="data/processed", help="Path to the processed directory (default: data/processed)")
    parser.add_argument("--dry-run", action="store_true", help="Only print what would be done without making changes")
    args = parser.parse_args()
    
    resolve_links(processed_dir=args.dir, dry_run=args.dry_run)
    
    if args.dry_run:
        print("\nThis was a dry run. No changes were made.")
    else:
        print("\nAll symbolic links have been resolved and replaced with actual copies.") 