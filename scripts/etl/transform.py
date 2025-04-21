import os
import random
import numpy as np
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
import audiomentations as A
import logging
import traceback
from typing import List, Dict, Optional, Union
import shutil

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('transform.log'),
        logging.StreamHandler()
    ]
)

def calculate_rms(samples: np.ndarray) -> float:
    """Calculate the RMS value of the audio samples."""
    return np.sqrt(np.mean(np.square(samples)))

def convert_decibels_to_amplitude_ratio(decibels: float) -> float:
    """Convert decibels to amplitude ratio."""
    return 10 ** (decibels / 20)

def add_short_noises_multi_channel(
    samples: np.ndarray,
    sample_rate: int,
    sounds_path: str = "noises/",
    min_snr_db: float = 3.0,
    max_snr_db: float = 30.0,
    noise_rms: str = "relative_to_whole_input",
    min_time_between_sounds: float = 2.0,
    max_time_between_sounds: float = 8.0,
    p: float = 0.5
) -> np.ndarray:
    """Custom implementation of AddShortNoises that supports multi-channel audio."""
    if random.random() > p:
        return samples

    # Convert input to float32 to avoid audiomentations warning
    samples = samples.astype(np.float32)

    # Load noise files
    noise_files = list(Path(sounds_path).glob("*.wav"))
    if not noise_files:
        return samples

    # Calculate number of sounds to add
    duration = samples.shape[-1] / sample_rate
    current_time = 0
    sounds: List[Dict] = []

    while current_time < duration:
        # Select a random noise file
        sound_file_path = str(random.choice(noise_files))
        
        # Load the noise file (soundfile format: samples, channels)
        noise_samples, noise_sample_rate = sf.read(sound_file_path)
        if len(noise_samples.shape) == 1:
            noise_samples = noise_samples[:, np.newaxis]  # Convert to 2D if mono
        
        # Convert noise to float32
        noise_samples = noise_samples.astype(np.float32)
        
        # Transpose to (channels, samples) format for audiomentations
        noise_samples = noise_samples.T
        
        # Ensure noise has same number of channels as input
        if noise_samples.shape[0] != samples.shape[0]:
            noise_samples = np.tile(noise_samples[0:1, :], (samples.shape[0], 1))
        
        sound_duration = noise_samples.shape[1] / noise_sample_rate
        
        # Add fade in/out
        fade_in_time = min(0.1, sound_duration / 2)
        fade_out_time = min(0.1, sound_duration / 2)
        
        # Calculate SNR
        snr_db = random.uniform(min_snr_db, max_snr_db)
        
        sounds.append({
            "fade_in_time": fade_in_time,
            "start": current_time,
            "end": current_time + sound_duration,
            "fade_out_time": fade_out_time,
            "samples": noise_samples,
            "sample_rate": noise_sample_rate,
            "snr_db": snr_db
        })
        
        # Add pause between sounds
        current_time += sound_duration + random.uniform(min_time_between_sounds, max_time_between_sounds)

    # Apply sounds to input
    output = samples.copy()
    for sound in sounds:
        start_sample = int(sound["start"] * sample_rate)
        end_sample = int(sound["end"] * sample_rate)
        
        if start_sample >= samples.shape[-1]:
            continue
            
        end_sample = min(end_sample, samples.shape[-1])
        noise = sound["samples"]
        
        # Resample if needed (noise is already in channels-first format)
        if sound["sample_rate"] != sample_rate:
            noise = A.Resample(
                min_sample_rate=sample_rate,
                max_sample_rate=sample_rate,
                p=1.0
            )(noise, sound["sample_rate"])
        
        # Trim or pad noise to fit
        if noise.shape[1] > (end_sample - start_sample):
            noise = noise[:, :end_sample - start_sample]
        else:
            padding = np.zeros((noise.shape[0], end_sample - start_sample - noise.shape[1]))
            noise = np.hstack([noise, padding])
        
        # Apply fade in/out
        fade_in_samples = int(sound["fade_in_time"] * sample_rate)
        fade_out_samples = int(sound["fade_out_time"] * sample_rate)
        
        fade_in = np.linspace(0, 1, fade_in_samples)
        fade_out = np.linspace(1, 0, fade_out_samples)
        
        noise[:, :fade_in_samples] *= fade_in
        noise[:, -fade_out_samples:] *= fade_out
        
        # Calculate and apply gain based on SNR
        if noise_rms == "relative_to_whole_input":
            clean_rms = calculate_rms(samples)
        else:
            clean_rms = calculate_rms(samples[:, start_sample:end_sample])
            
        noise_rms_value = calculate_rms(noise)
        if noise_rms_value > 0:
            desired_noise_rms = calculate_rms(clean_rms) / convert_decibels_to_amplitude_ratio(sound["snr_db"])
            gain = desired_noise_rms / noise_rms_value
            noise *= gain
        
        # Add noise to output
        output[:, start_sample:end_sample] += noise

    return output

def create_augmentation_pipeline():
    """Create an augmentation pipeline with the specified transforms."""
    return A.Compose([
        A.AddGaussianSNR(min_snr_db=15.0, max_snr_db=40.0, p=0.5),
        A.Lambda(
            transform=add_short_noises_multi_channel,
            min_snr_db=10.0,  # Increased minimum SNR for less noise
            max_snr_db=30.0,
            p=0.3  # Lower probability
        ),
        A.AirAbsorption(
            min_temperature=10.0,
            max_temperature=20.0,
            min_humidity=40.0,
            max_humidity=70.0,
            min_distance=5.0,
            max_distance=30.0,  # Reduced max distance
            p=0.3
        ),
        A.ClippingDistortion(
            min_percentile_threshold=5,
            max_percentile_threshold=20,  # Reduced from 40 for less distortion
            p=0.3
        ),
        A.Gain(
            min_gain_db=-6,  # Reduced range
            max_gain_db=6,
            p=0.4
        ),
        A.GainTransition(
            min_gain_db=-6,  # Reduced range
            max_gain_db=6,
            min_duration=0.3,
            max_duration=0.8,
            p=0.3
        ),
        A.Mp3Compression(
            min_bitrate=128,  # Higher minimum bitrate (better quality)
            max_bitrate=320,
            p=0.3
        ),
        A.RoomSimulator(
            min_size_x=3.0,
            max_size_x=6.0,  # Less extreme room size
            min_size_y=3.0,
            max_size_y=6.0,
            min_size_z=2.5,
            max_size_z=3.5,
            min_absorption_value=0.4,  # Higher minimum (less reverb)
            max_absorption_value=0.9,
            p=0.3
        )
    ])

def validate_directory(directory: Path) -> bool:
    """Check if a directory contains all required files."""
    # Required files for all directories
    required_files = {'bass.wav', 'drums.wav', 'mixture.wav', 'other.wav', 'vocals.wav'}
    
    # Only require augmentations.txt for augmented directories
    if "AUG" in directory.name:
        required_files.add('augmentations.txt')
    
    try:
        existing_files = set(f.name for f in directory.glob('*'))
        return required_files.issubset(existing_files)
    except Exception as e:
        logging.error(f"Error validating directory {directory}: {str(e)}")
        return False

def prune_invalid_directories(output_dir: Path):
    """Remove directories that don't contain all required files."""
    logging.info(f"Pruning invalid directories in {output_dir}")
    for directory in output_dir.iterdir():
        if directory.is_dir():
            if not validate_directory(directory):
                logging.warning(f"Removing invalid directory: {directory}")
                try:
                    shutil.rmtree(directory)
                except Exception as e:
                    logging.error(f"Error removing directory {directory}: {str(e)}")

def process_audio_file(input_path, output_path, augment, num_augmentations=1):  # Default to 1 augmentation since we handle indices in transform()
    """Process a single audio file and create augmented versions."""
    try:
        # Load the audio file (soundfile format: samples, channels)
        audio, sample_rate = sf.read(input_path)
        logging.debug(f"Input shape: {audio.shape}, sample_rate: {sample_rate}")
        
        # Convert to float32 to avoid audiomentations warning
        audio = audio.astype(np.float32)
        
        # Store original number of channels
        original_channels = audio.shape[1] if len(audio.shape) > 1 else 1
        logging.debug(f"Original channels: {original_channels}")
        
        # Transpose audio to (channels, samples) format for audiomentations
        if len(audio.shape) > 1:
            audio = audio.T
        else:
            audio = audio[np.newaxis, :]  # Add channel dimension for mono
        logging.debug(f"After transpose for augmentation: {audio.shape}")
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save original file (transpose back to soundfile format)
        if original_channels > 1:
            sf.write(output_path, audio.T, sample_rate)
        else:
            sf.write(output_path, audio[0], sample_rate)  # Remove channel dimension for mono
        
        # Create augmented versions
        for i in range(num_augmentations):
            # Get stem name from the output path
            stem_name = os.path.basename(output_path)
            
            # Get the augmentation directory from the output path
            aug_dir = os.path.dirname(output_path)
            
            # Save in the augmentation directory
            aug_output_path = os.path.join(aug_dir, stem_name)
            
            # Apply augmentations (audio is already in channels-first format)
            augmented_audio = augment(samples=audio, sample_rate=sample_rate)
            logging.debug(f"After augmentation: {augmented_audio.shape}")
            
            # Save augmented audio (transpose back to soundfile format)
            if original_channels > 1:
                sf.write(aug_output_path, augmented_audio.T, sample_rate)
            else:
                sf.write(aug_output_path, augmented_audio[0], sample_rate)  # Remove channel dimension for mono
            
            # Get the list of applied augmentations from the pipeline
            applied_augmentations = []
            for transform in augment.transforms:
                if hasattr(transform, 'p') and random.random() < transform.p:
                    # Get transform name and parameters
                    transform_name = transform.__class__.__name__
                    params = {k: v for k, v in transform.__dict__.items() 
                             if not k.startswith('_') and k != 'p'}
                    applied_augmentations.append(f"{transform_name}: {params}")
            
            # Write augmentations to file
            aug_file_path = os.path.join(aug_dir, 'augmentations.txt')
            with open(aug_file_path, 'a') as f:
                f.write(f"\nStem: {stem_name}\n")
                f.write("Applied augmentations:\n")
                for aug in applied_augmentations:
                    f.write(f"- {aug}\n")
                f.write("\n")
                
    except Exception as e:
        logging.error(f"Error processing {input_path}: {str(e)}")
        logging.error(traceback.format_exc())
        raise

def transform(force=False):
    """Transform the extracted data into a format suitable for training.
    
    Args:
        force (bool): If True, overwrite existing data. If False, only process if output directory is empty.
    """
    try:
        # Set random seed for reproducibility
        random.seed(42)
        np.random.seed(42)
        
        # Create augmentation pipeline
        augment = create_augmentation_pipeline()
        
        # Process train and test splits
        for split in ['train', 'test']:
            input_dir = Path(f'data/musdb18hq/{split}')
            output_dir = Path(f'data/processed/{split}')
            
            # Check if output directory exists and has files
            if output_dir.exists() and any(output_dir.iterdir()):
                if not force:
                    logging.warning(f"Output directory {output_dir} is not empty. Use --force to overwrite.")
                    continue
            
            # Get all song directories
            song_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
            
            for song_dir in tqdm(song_dirs, desc=f'Processing {split} split'):
                try:
                    # Create song directory in output (ORIGINAL for original)
                    song_name = song_dir.name
                    original_dir = output_dir / f'{song_name} ORIGINAL'
                    os.makedirs(original_dir, exist_ok=True)
                    
                    # First process all stems for the original version
                    for stem_file in song_dir.glob('*.wav'):
                        try:
                            # Create output path
                            output_path = original_dir / stem_file.name
                            
                            # Copy original files instead of symbolic linking
                            if not output_path.exists():
                                shutil.copy(stem_file, output_path)
                                
                        except Exception as e:
                            logging.error(f"Error copying original stem {stem_file}: {str(e)}")
                            logging.error(traceback.format_exc())
                            continue
                    
                    # Then process all stems for each augmentation index
                    for aug_idx in range(1, 3):  # 1 to 2 augmentations
                        try:
                            aug_dir = output_dir / f'{song_name} AUG{aug_idx}'
                            os.makedirs(aug_dir, exist_ok=True)
                            
                            # Process each stem for this augmentation index
                            for stem_file in song_dir.glob('*.wav'):
                                try:
                                    if stem_file.name == 'mixture.wav':
                                        # Copy mixture.wav from original instead of symbolic linking
                                        mixture_path = original_dir / 'mixture.wav'
                                        aug_mixture_path = aug_dir / 'mixture.wav'
                                        if not aug_mixture_path.exists():
                                            shutil.copy(mixture_path, aug_mixture_path)
                                        continue
                                    
                                    # Process the audio file with augmentation
                                    process_audio_file(
                                        input_path=str(stem_file),
                                        output_path=str(aug_dir / stem_file.name),
                                        augment=augment,
                                        num_augmentations=1  # Only create one augmentation
                                    )
                                except Exception as e:
                                    logging.error(f"Error processing augmented stem {stem_file} for augmentation {aug_idx}: {str(e)}")
                                    logging.error(traceback.format_exc())
                                    continue
                        except Exception as e:
                            logging.error(f"Error processing augmentation {aug_idx} for song {song_name}: {str(e)}")
                            logging.error(traceback.format_exc())
                            continue
                except Exception as e:
                    logging.error(f"Error processing song {song_dir}: {str(e)}")
                    logging.error(traceback.format_exc())
                    continue
        
        # After processing all songs, prune invalid directories
        for split in ['train', 'test']:
            output_dir = Path(f'data/processed/{split}')
            prune_invalid_directories(output_dir)
            
    except Exception as e:
        logging.error(f"Error in transform function: {str(e)}")
        logging.error(traceback.format_exc())
        raise
