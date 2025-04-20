import os
import sys
import torch
import torchaudio
import numpy as np
import glob
import random
from pathlib import Path
from .model import DeepModel
from scripts.types import Stems

def main():
    # Set device for PyTorch operations
    #device = torch.device("cuda" if torch.cuda.is_available() else 
    #                      "mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Load the trained model
    model_path = 'models/demucs_finetuned.pt'
    print(f"Loading model from {model_path}...")
    model = DeepModel.load(model_path)
    model.eval()  # Set to evaluation mode
    print("Model loaded successfully!")
    
    # Get list of all test song directories
    test_dir = 'data/processed/test'
    song_dirs = [d for d in glob.glob(os.path.join(test_dir, '*')) if os.path.isdir(d)]
    
    if not song_dirs:
        print(f"No song directories found in {test_dir}")
        return
    
    # Select a random song
    random_song_dir = random.choice(song_dirs)
    print(f"Selected song: {os.path.basename(random_song_dir)}")
    
    # Load the stems
    stems = Stems(random_song_dir)
    print(f"Loaded stems from {random_song_dir}")
    
    # Run inference
    print("Running inference...")
    with torch.no_grad():
        # The predict method expects a list of Stems
        result_mixes = model.predict([stems])
    
    if not result_mixes:
        print("No results returned from model")
        return
    
    # Get the processed mix (first and only result)
    processed_mix = result_mixes[0]
    
    # Save the result
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    output_filename = os.path.join(output_dir, f"{os.path.basename(random_song_dir)}_processed.wav")
    
    # Convert the processed mix to int16 format for saving
    if processed_mix.ndim == 2:  # Stereo
        # Save stereo audio
        torchaudio.save(
            output_filename,
            torch.tensor(processed_mix).t(),  # Transpose to [channels, samples]
            44100,
            encoding="PCM_S", 
            bits_per_sample=16
        )
    else:  # Mono
        # Save mono audio
        torchaudio.save(
            output_filename,
            torch.tensor(processed_mix).unsqueeze(0),  # Add channel dimension [1, samples]
            44100,
            encoding="PCM_S", 
            bits_per_sample=16
        )
    
    print(f"Processed mix saved to {output_filename}")
    
    # Also save the original mixture for comparison
    original_mix = stems.mixture_audio
    original_filename = os.path.join(output_dir, f"{os.path.basename(random_song_dir)}_original.wav")
    
    if original_mix.ndim == 2:  # Stereo
        torchaudio.save(
            original_filename,
            torch.tensor(original_mix).t(),  # Transpose to [channels, samples]
            44100,
            encoding="PCM_S", 
            bits_per_sample=16
        )
    else:  # Mono
        torchaudio.save(
            original_filename,
            torch.tensor(original_mix).unsqueeze(0),  # Add channel dimension [1, samples]
            44100,
            encoding="PCM_S", 
            bits_per_sample=16
        )
    
    print(f"Original mix saved to {original_filename}")
    print("Done!")

if __name__ == "__main__":
    main()
