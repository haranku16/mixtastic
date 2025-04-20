#!/usr/bin/env python3
import os
import sys
import numpy as np
import soundfile as sf
from pathlib import Path
import argparse

# Add project root to sys.path to enable imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from scripts.modeling.naive.model import NaiveModel
from scripts.types import Stems

def main():
    parser = argparse.ArgumentParser(description='Test naive mixing model')
    parser.add_argument('--stems_dir', default='data/processed/test/Al James - Schoolboy Facination AUG1',
                      help='Directory containing stems')
    parser.add_argument('--output', default='test.wav',
                      help='Output file path')
    args = parser.parse_args()
    
    # Path to test stems
    test_stem_path = args.stems_dir
    output_path = args.output
    
    print(f"Loading stems from {test_stem_path}...")
    
    # Create Stems object for the test song
    stems = Stems(path=test_stem_path, description="Test stems")
    
    # Initialize model
    model = NaiveModel()
    
    # Run prediction
    print("Running prediction with NaiveModel...")
    predicted_mixes = model.predict([stems])
    
    # Get the predicted mix (first item in the list)
    mix = predicted_mixes[0]
    
    # Determine sample rate
    sample_rate = getattr(stems, '_sample_rate', 44100)
    
    # Save mix to output file
    print(f"Saving mix to {output_path}...")
    
    # Write directly to wav file without conversion
    sf.write(output_path, mix, sample_rate)
    
    print(f"Done! Mix saved to {output_path}")

if __name__ == "__main__":
    main()
