from scripts.modeling.model import Model
from scripts.types import Stems, Mix
from typing import List
import auraloss
import torch
import numpy as np

# Initialize MultiResolutionSTFTLoss with parameters suitable for 10-minute audio
mrstft = auraloss.freq.MultiResolutionSTFTLoss(
    fft_sizes=[8192, 16384, 32768],  # Much larger FFT sizes for long audio
    hop_sizes=[2048, 4096, 8192],     # Larger hop sizes
    win_lengths=[8192, 16384, 32768]  # Larger window lengths
)

def evaluate(model: Model, X: List[Stems], y: List[Mix]) -> float:
    '''
    Evaluate the model.
    '''
    
    pred = model.predict(X)
    # Convert predictions and targets to torch tensors
    pred_tensor = torch.tensor(np.array(pred))
    y_tensor = torch.tensor(np.array(y))
    
    # Ensure tensors are float32 and properly shaped
    pred_tensor = pred_tensor.float()
    y_tensor = y_tensor.float()
    
    # Reshape tensors to (batch_size, channels, sequence_length)
    if len(pred_tensor.shape) == 2:
        pred_tensor = pred_tensor.unsqueeze(0)  # Add batch dimension if missing
    if len(y_tensor.shape) == 2:
        y_tensor = y_tensor.unsqueeze(0)  # Add batch dimension if missing
    
    # Ensure tensors are contiguous in memory
    pred_tensor = pred_tensor.contiguous()
    y_tensor = y_tensor.contiguous()
    
    # Process in chunks if needed
    chunk_size = 44100 * 60  # 1 minute chunks
    total_loss = 0.0
    num_chunks = 0
    
    for i in range(0, pred_tensor.shape[2], chunk_size):  # Changed from shape[1] to shape[2]
        chunk_pred = pred_tensor[:, :, i:i+chunk_size]    # Changed indexing to match new shape
        chunk_target = y_tensor[:, :, i:i+chunk_size]     # Changed indexing to match new shape
        
        if chunk_pred.shape[2] < chunk_size:  # Changed from shape[1] to shape[2]
            continue  # Skip incomplete chunks
            
        loss = mrstft(chunk_pred, chunk_target)
        total_loss += loss.item()
        num_chunks += 1
    
    return total_loss / num_chunks if num_chunks > 0 else 0.0
