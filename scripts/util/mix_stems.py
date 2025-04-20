#!/usr/bin/env python3
import numpy as np
import soundfile as sf
import os
import argparse
from pathlib import Path
from typing import List, Union, Optional, Tuple

def load_audio_stems(stems_dir: Union[str, Path], stem_files: Optional[List[str]] = None) -> Tuple[List[np.ndarray], int]:
    """
    Load audio stems from a directory.
    
    Args:
        stems_dir: Directory containing stems
        stem_files: Optional list of stem filenames. If None, defaults to ['bass.wav', 'drums.wav', 'vocals.wav', 'other.wav']
        
    Returns:
        Tuple containing:
        - List of audio arrays 
        - Sample rate
    """
    if stem_files is None:
        stem_files = ['bass.wav', 'drums.wav', 'vocals.wav', 'other.wav']
    
    stems_dir = Path(stems_dir)
    audio_data = []
    shapes = []
    sample_rate = None
    
    for stem in stem_files:
        stem_path = stems_dir / stem
        if not stem_path.exists():
            print(f"Warning: {stem} not found at {stem_path}")
            continue
            
        data, sr = sf.read(stem_path)
        if sample_rate is None:
            sample_rate = sr
        elif sr != sample_rate:
            print(f"Warning: Sample rate mismatch in {stem}")
        
        print(f"Loaded {stem}: shape={data.shape}, dtype={data.dtype}")
        audio_data.append(data)
        shapes.append(data.shape)
    
    return audio_data, sample_rate

def process_stems(audio_data: List[np.ndarray], normalize: bool = True, normalize_factor: float = 0.9) -> np.ndarray:
    """
    Process audio stems into a mixed track.
    
    Args:
        audio_data: List of audio arrays
        normalize: Whether to normalize the output
        normalize_factor: Factor to normalize to (between 0 and 1)
        
    Returns:
        Mixed audio array
    """
    if not audio_data:
        raise ValueError("No valid stems found!")
    
    # Find minimum length and truncate if needed
    min_length = min(shape[0] for shape in [arr.shape for arr in audio_data])
    audio_arrays = [data[:min_length] for data in audio_data]
    
    # Mix stems by taking the mean
    mixed = np.mean(audio_arrays, axis=0)
    
    # Clip to prevent distortion
    max_val = np.max(np.abs(mixed))
    if max_val > 1.0:
        print(f"Clipping audio that exceeds bounds (max value: {max_val})")
        mixed = np.clip(mixed, -1.0, 1.0)
    
    # Normalize audio
    if normalize:
        max_val = np.max(np.abs(mixed))
        if max_val > 0:
            mixed = mixed / max_val * normalize_factor
    
    return mixed

def mix_stems(stems_dir: Union[str, Path], 
              output_path: Union[str, Path], 
              stem_files: Optional[List[str]] = None,
              normalize: bool = True, 
              normalize_factor: float = 0.9) -> None:
    """
    Load, mix and save stems from a directory.
    
    Args:
        stems_dir: Directory containing stems
        output_path: Path to save mixed audio to
        stem_files: Optional list of stem filenames. If None, defaults to ['bass.wav', 'drums.wav', 'vocals.wav', 'other.wav']
        normalize: Whether to normalize the output
        normalize_factor: Factor to normalize to (between 0 and 1)
    """
    print(f"Loading stems from {stems_dir}...")
    audio_data, sample_rate = load_audio_stems(stems_dir, stem_files)
    
    print("Mixing stems...")
    mixed = process_stems(audio_data, normalize, normalize_factor)
    
    # Save mixed audio
    print(f"Saving mixed audio to {output_path}...")
    sf.write(output_path, mixed, sample_rate)
    print("Done!")

def main():
    parser = argparse.ArgumentParser(description='Mix audio stems and save as WAV')
    parser.add_argument('--stems_dir', default='data/processed/test/Al James - Schoolboy Facination AUG1',
                      help='Directory containing stems')
    parser.add_argument('--output', default='test.wav',
                      help='Output file path')
    parser.add_argument('--no-normalize', action='store_true',
                      help='Disable audio normalization')
    parser.add_argument('--normalize-factor', type=float, default=0.9,
                      help='Normalization factor (0-1)')
    args = parser.parse_args()
    
    mix_stems(
        stems_dir=args.stems_dir, 
        output_path=args.output, 
        normalize=not args.no_normalize,
        normalize_factor=args.normalize_factor
    )

if __name__ == "__main__":
    main()
