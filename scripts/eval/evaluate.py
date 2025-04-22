from scripts.modeling.model import Model
from scripts.types import Stems, Mix
from scripts.util.device import get_best_device
from scripts.util.audio_utils import pad_to_length
from typing import List
import auraloss
import torch
import numpy as np
from tqdm import tqdm

# Initialize MultiResolutionSTFTLoss with parameters suitable for 10-minute audio
mrstft = auraloss.freq.MultiResolutionSTFTLoss(
    fft_sizes=[8192, 16384, 32768],  # Much larger FFT sizes for long audio
    hop_sizes=[2048, 4096, 8192],     # Larger hop sizes
    win_lengths=[8192, 16384, 32768]  # Larger window lengths
)

def clear_device_memory(device: str):
    """Clear memory for the specified device."""
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        # MPS doesn't have an explicit cache clearing mechanism,
        # but we can try to force garbage collection
        import gc
        gc.collect()

def evaluate(model: Model, X: List[Stems], y: List[Mix]) -> float:
    '''
    Evaluate the model.
    '''
    # Get the best available device
    device = get_best_device()
    
    # Set fixed length for evaluation (30 seconds at 44.1kHz)
    target_length = 44100 * 30
    
    # Process predictions one at a time to save memory
    total_loss = 0.0
    num_chunks = 0
    
    for i, (stems, target) in enumerate(tqdm(zip(X, y), desc="Evaluating", total=len(X))):
        # Get prediction for current stems
        pred = model.predict([stems])[0]  # Process one stem at a time
        
        # Pad/trim both prediction and target to exactly 30 seconds
        pred = pad_to_length(pred, target_length, stereo=True)
        target = pad_to_length(target, target_length, stereo=True)
        
        # Convert to tensors and ensure correct shape
        pred_tensor = torch.tensor(pred).float().to(device)
        target_tensor = torch.tensor(target).float().to(device)
        
        # Normalize target to match prediction scale
        target_tensor = target_tensor / 32768.0  # Convert from 16-bit to [-1, 1]
        
        # Ensure tensors have shape (batch, channels, time)
        if len(pred_tensor.shape) == 1:  # Mono audio
            pred_tensor = pred_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, time)
            target_tensor = target_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, time)
        elif len(pred_tensor.shape) == 2:  # Stereo audio
            pred_tensor = pred_tensor.permute(1, 0).unsqueeze(0)  # (1, channels, time)
            target_tensor = target_tensor.permute(1, 0).unsqueeze(0)  # (1, channels, time)
        
        # Calculate loss directly (no need to chunk for 30 seconds)
        loss = mrstft(pred_tensor, target_tensor)
        total_loss += loss.item()
        num_chunks += 1
            
        # Clear memory after processing each prediction
        del pred_tensor, target_tensor
        clear_device_memory(device)
    
    return total_loss / num_chunks if num_chunks > 0 else 0.0
