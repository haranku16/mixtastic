#!/usr/bin/env python3

import argparse
import os
import soundfile as sf
import torch

from scripts.types import Stems
from scripts.util.audio_utils import pad_to_length
from scripts.modeling.naive.model import NaiveModel
from scripts.modeling.deep.model import DeepModel
from scripts.modeling.traditional.model import TraditionalModel

def load_model(model_name):
    """
    Load the specified model.
    
    Args:
        model_name (str): Name of the model to load. Can be 'naive' or a path to a DeepModel checkpoint.
        
    Returns:
        Model: An instance of a model that can predict mixes.
    """
    if model_name.lower() == 'naive':
        print("Loading NaiveModel...")
        return NaiveModel()
    elif model_name.lower() == 'traditional':
        print("Loading TraditionalModel...")
        return TraditionalModel.load('models/traditional_simple.pkl')
    else:
        # Assume it's a path to a DeepModel checkpoint
        print(f"Loading DeepModel from {model_name}...")
        return DeepModel.load(model_name)

def run_inference(model, stem_path, output_path):
    """
    Run inference with the model on the given stems and save the result.
    
    Args:
        model: The model to use for inference
        stem_path (str): Path to the directory containing stems
        output_path (str): Path to save the output mix
    """
    print(f"Loading stems from {stem_path}...")

    # Create Stems object for the test song
    stems = Stems(path=stem_path, description="Test stems")

    # Run prediction
    print(f"Running prediction with {model.__class__.__name__}...")
    predicted_mixes = model.predict([stems])

    # Get the predicted mix (first item in the list)
    mix = predicted_mixes[0]

    # Determine sample rate
    sample_rate = getattr(stems, '_sample_rate', 44100)

    # Pad to 30 seconds if needed
    mix = pad_to_length(mix, 30 * sample_rate, True)

    # Save mix to output file
    print(f"Saving mix to {output_path}...")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Write directly to wav file
    print(f'Writing with sample rate {sample_rate}')
    sf.write(output_path, mix, sample_rate)

    print(f"Done! Mix saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Run inference with a MixTastic model')
    parser.add_argument('--model', type=str, default='naive', 
                        help='Model to use for inference. Can be "naive" or a path to a DeepModel checkpoint')
    parser.add_argument('--stems', type=str, default='data/processed/test/Al James - Schoolboy Facination AUG1',
                        help='Path to the directory containing stems')
    parser.add_argument('--output', type=str, default='output/inference_output.wav',
                        help='Path to save the output mix')
    
    args = parser.parse_args()
    
    # Load the model
    model = load_model(args.model)
    
    # Run inference
    run_inference(model, args.stems, args.output)

if __name__ == '__main__':
    main() 