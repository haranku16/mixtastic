from scripts.modeling.model import Model
from scripts.types import Stems, Mix
from typing import List, Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import os
import torchaudio
from scripts.util.audio_utils import pad_to_length
import gc
import glob
import random
from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS
import psutil
import warnings

# Set global memory constants
MAX_AUDIO_DURATION_SECONDS = 30  # 30 seconds
SAMPLE_RATE = 44100
MAX_SAMPLES = SAMPLE_RATE * MAX_AUDIO_DURATION_SECONDS

# Set MPS fallback environment variable to handle unsupported operations
# This will allow PyTorch to fall back to CPU for unsupported operations like angle()
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Check if MPS (Metal Performance Shaders) is available on macOS
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
        
# Get the device to use throughout the model
DEVICE = get_device()
print(f"Using device: {DEVICE}")
if DEVICE.type == "mps":
    print(f"MPS device detected. CPU fallback enabled for unsupported operations.")

def free_memory():
    """Aggressively free memory"""
    print("Freeing memory...")
    # Run garbage collection
    gc.collect()
    
    # Empty torch cache based on available device
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"Warning: Could not clear CUDA cache: {e}")
    
    # Empty MPS cache with error handling
    if torch.backends.mps.is_available():
        try:
            # Just call empty_cache without setting watermark ratios
            torch.mps.empty_cache()
        except Exception as e:
            print(f"Warning: Could not clear MPS cache: {e}")
            
    # Empty general PyTorch memory
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                if obj.device.type != 'cpu':
                    # Move tensor to CPU to release GPU memory
                    obj.data = obj.data.to('cpu')
                    if hasattr(obj, 'grad') and obj.grad is not None:
                        obj.grad.data = obj.grad.data.to('cpu')
        except Exception:
            pass

def get_duration_str(samples):
    """Convert sample count to duration string"""
    # For numpy arrays and other sequences
    if hasattr(samples, 'shape'):
        # Get the time dimension for different tensor shapes
        if len(samples.shape) == 1:  # 1D array [time]
            sample_length = samples.shape[0]
        elif len(samples.shape) == 2:  # 2D array [batch/channel, time] or [time, channels]
            sample_length = max(samples.shape)
        elif len(samples.shape) == 3:  # 3D array with time typically in middle
            sample_length = samples.shape[1]
        elif len(samples.shape) == 4:  # 4D array with time typically 3rd dimension
            sample_length = samples.shape[2]
        else:
            sample_length = samples.size if hasattr(samples, 'size') else len(samples)
    else:
        # Simple sequence
        sample_length = len(samples)
    
    # Calculate duration in seconds and format
    duration_sec = sample_length / SAMPLE_RATE
    return f"{duration_sec:.2f}s ({sample_length} samples)"

class StemMixDataset(Dataset):
    def __init__(self, stems_list: List[Stems]):
        self.stems_list = stems_list
        self.target_length = MAX_SAMPLES  # 30 seconds at 44.1kHz

    def __len__(self):
        return len(self.stems_list)

    def __getitem__(self, idx):
        try:
            stems = self.stems_list[idx]
            mix = self.stems_list[idx].mixture_audio
            
            # Determine if audio is stereo or mono
            is_stereo = mix.ndim == 2 and mix.shape[1] == 2
            
            # Pad/truncate stems and mix to target length
            if is_stereo:
                # Handle stereo audio (shape: samples, 2)
                stems_tensor = torch.stack([
                    torch.from_numpy(pad_to_length(stems.bass_audio, self.target_length, stereo=True)).float(),
                    torch.from_numpy(pad_to_length(stems.drums_audio, self.target_length, stereo=True)).float(),
                    torch.from_numpy(pad_to_length(stems.other_audio, self.target_length, stereo=True)).float(),
                    torch.from_numpy(pad_to_length(stems.vocals_audio, self.target_length, stereo=True)).float()
                ])
                
                mix_tensor = torch.from_numpy(pad_to_length(mix, self.target_length, stereo=True)).float()
                
                # Reshape to [4, time, 2] for stems and [time, 2] for mix
                if stems_tensor.dim() == 3:  # [4, time*2]
                    stems_tensor = stems_tensor.view(4, self.target_length, 2)
                if mix_tensor.dim() == 1:  # [time*2]
                    mix_tensor = mix_tensor.view(self.target_length, 2)
            else:
                # Handle mono audio
                stems_tensor = torch.stack([
                    torch.from_numpy(pad_to_length(stems.bass_audio, self.target_length)).float(),
                    torch.from_numpy(pad_to_length(stems.drums_audio, self.target_length)).float(),
                    torch.from_numpy(pad_to_length(stems.other_audio, self.target_length)).float(),
                    torch.from_numpy(pad_to_length(stems.vocals_audio, self.target_length)).float()
                ])
                
                mix_tensor = torch.from_numpy(pad_to_length(mix, self.target_length)).float()
            
            # Release memory immediately
            stems._bass_audio = None
            stems._drums_audio = None
            stems._other_audio = None
            stems._vocals_audio = None
            stems._mixture_audio = None
            # Force garbage collection to free memory
            gc.collect()
            
            return stems_tensor, mix_tensor
        except Exception as e:
            print(f"Error processing item {idx}: {e}")
            # Return a small dummy tensor in case of error
            return (torch.zeros(4, 1000, dtype=torch.float32), 
                   torch.zeros(1000, dtype=torch.float32))

def simple_mix_stems(stems):
    """Simple mean mixing of stems to create a naive mix"""
    # Check if we have stereo stems (last dimension is 2)
    if stems.dim() == 4 and stems.shape[-1] == 2:
        # Handle stereo stems - preserve stereo channels
        return stems.mean(dim=1)  # This will return [batch_size, time, 2]
    else:
        # Handle mono stems
        return stems.mean(dim=1)  # This will return [batch_size, time]

class ResourceMonitorCallback(pl.Callback):
    """Callback to monitor and log system resource usage"""
    def __init__(self, log_frequency=5):
        super().__init__()
        self.log_frequency = log_frequency
        self.last_logged_batch = -log_frequency  # Log on first batch
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Only log every N batches to reduce overhead
        if batch_idx - self.last_logged_batch >= self.log_frequency:
            self._log_resources(pl_module)
            self.last_logged_batch = batch_idx
            
            # Free up memory periodically
            free_memory()
    
    def on_validation_epoch_end(self, trainer, pl_module):
        self._log_resources(pl_module)
        
        # Force cleanup after validation
        free_memory()
    
    def _log_resources(self, pl_module):
        # Log CPU usage
        cpu_percent = psutil.cpu_percent()
        pl_module.log('system/cpu_percent', cpu_percent, prog_bar=True)
        
        # Log RAM usage
        mem = psutil.virtual_memory()
        ram_used_gb = mem.used / (1024 ** 3)
        ram_free_gb = mem.available / (1024 ** 3)
        pl_module.log('system/ram_used_gb', ram_used_gb, prog_bar=True)
        pl_module.log('system/ram_free_gb', ram_free_gb, prog_bar=True)
        
        # Log GPU memory if available on CUDA
        if torch.cuda.is_available():
            gpu_memory_used = torch.cuda.memory_allocated() / (1024 ** 3)
            pl_module.log('system/gpu_memory_gb', gpu_memory_used, prog_bar=True)

class DeepModel(Model, pl.LightningModule):
    def __init__(self, config: Dict[str, Any] = None):
        Model.__init__(self)
        pl.LightningModule.__init__(self)
        
        # Default config with minimal memory usage for M2 Pro
        self.config = config or {
            'learning_rate': 1e-5,  # Lower learning rate for fine-tuning
            'batch_size': 1,  # Minimum batch size to minimize memory usage
            'num_workers': 1,  # Minimum workers to minimize memory usage
            'max_epochs': 50,
            'early_stopping_patience': 10,
            'model_save_path': 'models/demucs_finetuned.pt',
            'segment_size': 5,  # Process in 5-second segments
            'mixed_precision': True,  # Use mixed precision training
            'gradient_checkpointing': True,  # Enable gradient checkpointing to save memory
        }
        
        # Get the pretrained Demucs model - we will finetune this
        bundle = HDEMUCS_HIGH_MUSDB_PLUS
        self.demucs = bundle.get_model().to(DEVICE)
        self.sample_rate = bundle.sample_rate
        
        # Store the expected number of audio channels for the model
        self.expected_channels = getattr(self.demucs, 'audio_channels', 2)
        
        # Optimization: Enable gradient checkpointing to save memory
        if self.config.get('gradient_checkpointing', True) and hasattr(self.demucs, 'encoder'):
            print("Enabling gradient checkpointing to save memory")
            self.demucs.encoder.use_checkpoint = True
        
        # Don't freeze the parameters - we want to finetune them
        for param in self.demucs.parameters():
            param.requires_grad = True
        
        # Loss function - a combination of L1 and spectral loss
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        
        # We don't initialize STFT here anymore - we create it dynamically in _spectral_loss
        # to ensure device consistency

    def _spectral_loss(self, y_pred, y_true):
        """Compute loss in the frequency domain with CPU fallback for unsupported MPS operations"""
        try:
            # Create a CPU-based STFT to ensure window and input are on same device
            # Use smaller FFT size to reduce memory usage
            cpu_stft = torchaudio.transforms.Spectrogram(
                n_fft=2048,  # Reduced from 4096 to save memory
                hop_length=512,  # Reduced from 1024 to save memory
                power=None,
            ).to("cpu")
            
            # Move to CPU for operations not supported on MPS (specifically angle calculation)
            # Note: we don't use detach() so gradients can still flow back
            cpu_device = torch.device("cpu")
            # Use clone() to avoid modifying the original tensor
            y_pred_cpu = y_pred.clone().to(cpu_device)  
            y_true_cpu = y_true.clone().to(cpu_device)
            
            # Handle multi-channel audio (stereo)
            if y_pred_cpu.dim() == 3 and y_true_cpu.dim() == 3:
                # Check if the last dimension is the channel dimension (batch, time, channel)
                if y_pred_cpu.shape[-1] <= 2:  # Assuming 1 or 2 channels
                    # Compute loss separately for each channel then average
                    batch_size, time_len, channels = y_pred_cpu.shape
                    total_loss = 0
                    
                    for ch in range(channels):
                        # Extract single channel
                        y_pred_ch = y_pred_cpu[..., ch]
                        y_true_ch = y_true_cpu[..., ch]
                        
                        # Compute spectral loss for this channel - using local CPU STFT
                        spec_pred = cpu_stft(y_pred_ch)
                        spec_true = cpu_stft(y_true_ch)
                        
                        # Magnitude loss
                        mag_pred = torch.abs(spec_pred)
                        mag_true = torch.abs(spec_true)
                        mag_loss = F.mse_loss(mag_pred, mag_true)
                        
                        # Phase loss (less weight as it's harder to predict perfectly)
                        phase_pred = torch.angle(spec_pred)  # This operation uses CPU fallback
                        phase_true = torch.angle(spec_true)
                        phase_loss = 0.1 * F.mse_loss(phase_pred, phase_true)
                        
                        # Add to total
                        total_loss += mag_loss + phase_loss
                        
                        # Clear variables to save memory
                        del spec_pred, spec_true, mag_pred, mag_true, phase_pred, phase_true
                    
                    # Average across channels
                    total_loss = total_loss / channels
                    
                    # Move result back to original device for backward pass
                    return total_loss.to(y_pred.device)
                
            # Handle mono or other tensor shapes
            spec_pred = cpu_stft(y_pred_cpu)
            spec_true = cpu_stft(y_true_cpu)
            
            # Magnitude loss
            mag_pred = torch.abs(spec_pred)
            mag_true = torch.abs(spec_true)
            mag_loss = F.mse_loss(mag_pred, mag_true)
            
            # Phase loss (less weight as it's harder to predict perfectly)
            phase_pred = torch.angle(spec_pred)
            phase_true = torch.angle(spec_true)
            phase_loss = 0.1 * F.mse_loss(phase_pred, phase_true)
            
            # Clear large variables to save memory
            del spec_pred, spec_true, mag_pred, mag_true, phase_pred, phase_true
            
            # Move the final loss back to the original device for backward pass
            total_loss = (mag_loss + phase_loss).to(y_pred.device)
            
            return total_loss
            
        except Exception as e:
            # If spectral loss fails, log the error and fall back to L1 loss
            print(f"Error in spectral loss calculation: {e}")
            print("Falling back to L1 loss only")
            return self.l1_loss(y_pred, y_true)
        
    def process_segment(self, stems_segment):
        """
        Process one segment through the finetuned Demucs pipeline:
        1. Simple mix stems
        2. Separate with Demucs
        3. Mix the Demucs output stems
        """
        # Create a simple mix from stems
        naive_mix = simple_mix_stems(stems_segment)  # [batch_size, time]
        
        # Check if the naive_mix is stereo (has an extra dimension)
        if naive_mix.dim() == 3:  # [batch_size, time, 2] (stereo)
            # Convert stereo to mono by averaging channels
            naive_mix = naive_mix.mean(dim=-1)  # [batch_size, time]
        
        # Add channel dimension for Demucs
        naive_mix = naive_mix.unsqueeze(1)  # [batch_size, 1, time]
        
        # Check if we need to adjust the number of channels for Demucs
        if naive_mix.shape[1] != self.expected_channels:
            # If we have 1 channel but need more, repeat the channel
            if naive_mix.shape[1] == 1 and self.expected_channels > 1:
                naive_mix = naive_mix.repeat(1, self.expected_channels, 1)
        
        # Clean up memory before heavy computation
        free_memory()
        
        # Get the improved stems through Demucs
        # Demucs outputs a tensor of shape [batch_size, sources=4, time]
        improved_stems = self.demucs(naive_mix)
        
        # Mix the improved stems to get the final output
        final_mix = simple_mix_stems(improved_stems)  # [batch_size, time]
        
        # Ensure the final_mix is exactly MAX_SAMPLES long
        if final_mix.shape[-1] != MAX_SAMPLES:
            # Convert to numpy, pad/truncate, then back to tensor
            final_mix_np = final_mix.cpu().numpy()
            final_mix_np = np.array([pad_to_length(mix, MAX_SAMPLES) for mix in final_mix_np])
            final_mix = torch.from_numpy(final_mix_np).to(final_mix.device).float()
        
        # Also ensure naive_mix is standardized
        naive_mix = naive_mix.squeeze(1)
        if naive_mix.shape[-1] != MAX_SAMPLES:
            naive_mix_np = naive_mix.cpu().numpy()
            naive_mix_np = np.array([pad_to_length(mix, MAX_SAMPLES) for mix in naive_mix_np])
            naive_mix = torch.from_numpy(naive_mix_np).to(naive_mix.device).float()
        
        return final_mix, naive_mix, improved_stems
        
    def forward(self, stems):
        """
        Forward pass through the model
        
        Args:
            stems: Tensor of shape [batch_size, num_stems=4, time] or [batch_size, num_stems=4, time, 2] for stereo
        
        Returns:
            final_mix: Tensor of shape [batch_size, time] or [batch_size, time, 2] for stereo
            naive_mix: Tensor of shape [batch_size, time] or [batch_size, time, 2] for stereo
            improved_stems: Tensor of shape [batch_size, 4, time] or [batch_size, 4, time, 2] for stereo
        """
        # Check if input is stereo (has extra dimension)
        is_stereo = stems.dim() == 4 and stems.shape[-1] == 2
        
        # If stereo, we need to convert to mono for demucs processing
        if is_stereo:
            # Keep original stereo stems for later
            original_stereo_stems = stems.clone()
            
            # Convert to mono by averaging channels
            stems = stems.mean(dim=-1)  # now [batch_size, num_stems=4, time]
        
        # Process segments with Demucs (which expects mono audio)
        final_mix, naive_mix, improved_stems = self.process_segment(stems)
        
        # From the logs, we see these shapes are already correct:
        # final_mix: [batch, channels, time]
        # naive_mix: [batch, channels, time]
        # improved_stems: [batch, sources, channels, time]
        
        return final_mix, naive_mix, improved_stems
        
    def training_step(self, batch, batch_idx):
        """Training step for Lightning"""
        stems, target_mix = batch
        
        # Process stems through our pipeline
        final_mix, naive_mix, improved_stems = self(stems)
        
        # Reshape if necessary to match dimensions
        # final_mix shape is [batch, channels, time] while target_mix is [batch, time, channels]
        if final_mix.dim() == 3 and target_mix.dim() == 3:
            # Check if channels are in different positions
            if final_mix.shape[1] == target_mix.shape[2]:
                # Need to permute final_mix to match target_mix
                final_mix = final_mix.permute(0, 2, 1)  # [batch, time, channels]
                naive_mix = naive_mix.permute(0, 2, 1)  # [batch, time, channels]
        
        # Calculate losses
        # Time domain loss for final mix vs target
        time_loss = self.l1_loss(final_mix, target_mix)
        
        # Frequency domain loss for better spectral characteristics
        spec_loss = self._spectral_loss(final_mix, target_mix)
        
        # Combined loss
        total_loss = time_loss + spec_loss
        
        # Optional: Add regularization to keep improvements subtle
        if self.config.get('regularize_changes', True):
            # Add loss to ensure improved stems aren't too different from input stems
            # Need to reshape improved_stems to match original stems dimensions
            if improved_stems.dim() == 4 and stems.dim() == 4:
                # improved_stems is [batch, sources, channels, time]
                # stems is [batch, sources, time, channels]
                improved_stems_reshaped = improved_stems.permute(0, 1, 3, 2)  # [batch, sources, time, channels]
                stem_diff_loss = 0.1 * self.l1_loss(improved_stems_reshaped, stems)
            else:
                stem_diff_loss = 0.1 * self.l1_loss(improved_stems, stems)
                
            total_loss += stem_diff_loss
            self.log('train_stem_diff_loss', stem_diff_loss)
        
        # Calculate improvement over naive mix
        naive_loss = self.l1_loss(naive_mix, target_mix)
        improvement = naive_loss - time_loss
        
        # Log metrics
        self.log('train_loss', total_loss, prog_bar=True)
        self.log('train_time_loss', time_loss)
        self.log('train_spec_loss', spec_loss)
        self.log('train_naive_loss', naive_loss)
        self.log('train_improvement', improvement, prog_bar=True)
        
        # Log current learning rate
        opt = self.optimizers()
        if opt is not None:
            current_lr = opt.param_groups[0]['lr']
            self.log('learning_rate', current_lr, prog_bar=True)
        
        # Clean up memory after each step
        free_memory()
        
        # Clear references to large tensors to free memory
        del final_mix, naive_mix, improved_stems
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step for Lightning"""
        stems, target_mix = batch
        
        # Process stems through our pipeline
        final_mix, naive_mix, improved_stems = self(stems)
        
        # Reshape if necessary to match dimensions
        # final_mix shape is [batch, channels, time] while target_mix is [batch, time, channels]
        if final_mix.dim() == 3 and target_mix.dim() == 3:
            # Check if channels are in different positions
            if final_mix.shape[1] == target_mix.shape[2]:
                # Need to permute final_mix to match target_mix
                final_mix = final_mix.permute(0, 2, 1)  # [batch, time, channels]
                naive_mix = naive_mix.permute(0, 2, 1)  # [batch, time, channels]
        
        # Calculate losses
        time_loss = self.l1_loss(final_mix, target_mix)
        spec_loss = self._spectral_loss(final_mix, target_mix)
        total_loss = time_loss + spec_loss
        
        # Calculate improvement over naive mix
        naive_loss = self.l1_loss(naive_mix, target_mix)
        improvement = naive_loss - time_loss
        
        # Log metrics
        self.log('val_loss', total_loss, prog_bar=True)
        self.log('val_time_loss', time_loss)
        self.log('val_spec_loss', spec_loss)
        self.log('val_naive_loss', naive_loss)
        self.log('val_improvement', improvement, prog_bar=True)
        
        # Clean up memory after validation step
        free_memory()
        
        # Clear references to large tensors to free memory
        del final_mix, naive_mix, improved_stems
        
        return total_loss
        
    def configure_optimizers(self):
        """Configure optimizers for Lightning"""
        # Use AdamW optimizer for better regularization
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.config.get('learning_rate', 1e-5),
            weight_decay=1e-5
        )
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5, 
            verbose=False  # Disable verbose output to avoid warnings
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }
        
    def fit(self, stems: List[Stems]):
        """Custom fit method that sets up Lightning trainer"""
        # Create dataset
        dataset = StemMixDataset(stems)
        
        # Split into train and validation sets
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        # Create data loaders with appropriate worker settings
        batch_size = self.config.get('batch_size', 1)
        use_persistent_workers = batch_size > 1  # Only use persistent workers if batch size > 1
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.config.get('num_workers', 1),
            pin_memory=True,
            persistent_workers=use_persistent_workers  # Use persistent workers if batch size > 1
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.config.get('num_workers', 1),
            pin_memory=True,
            persistent_workers=use_persistent_workers  # Use persistent workers if batch size > 1
        )
        
        # Set up trainer - use MPS for Apple Silicon GPU
        trainer_kwargs = {
            'max_epochs': self.config.get('max_epochs', 50),
            'callbacks': [
                pl.callbacks.EarlyStopping(
                    monitor='val_improvement',
                    patience=self.config.get('early_stopping_patience', 10),
                    mode='max'  # We want to maximize improvement
                ),
                pl.callbacks.ModelCheckpoint(
                    monitor='val_improvement',
                    dirpath='models/checkpoints/',
                    filename='demucs-{epoch:02d}-{val_improvement:.4f}',
                    save_top_k=3,
                    mode='max'
                ),
                # Add resource monitoring callback
                ResourceMonitorCallback(log_frequency=1)  # Check every batch for memory issues
            ],
            'accelerator': 'mps' if torch.backends.mps.is_available() else ('gpu' if torch.cuda.is_available() else 'cpu'),
            'devices': 1,
            'enable_progress_bar': True,
            'logger': pl.loggers.TensorBoardLogger('logs/', name='demucs'),
            'log_every_n_steps': 1,  # Log every batch since we have few batches
            'gradient_clip_val': 1.0,  # Prevent instability
            'accumulate_grad_batches': 8  # Increased from 4 to 8 to save memory
        }
        
        # Add mixed precision if requested (significantly speeds up training)
        if self.config.get('mixed_precision', True) and (torch.cuda.is_available() or torch.backends.mps.is_available()):
            # Note: MPS doesn't fully support mixed precision yet, but we'll keep it for future compatibility
            trainer_kwargs['precision'] = '16-mixed'
        
        trainer = pl.Trainer(**trainer_kwargs)
        
        print(f"Starting training with {train_size} training samples and {val_size} validation samples")
        print(f"Training with batch size {self.config.get('batch_size', 1)}, {self.config.get('num_workers', 1)} worker thread")
        print(f"Accumulating gradients over {trainer_kwargs['accumulate_grad_batches']} batches")
        print(f"Using accelerator: {trainer_kwargs['accelerator']}")
        print(f"Gradient checkpointing: {'enabled' if self.config.get('gradient_checkpointing', False) else 'disabled'}")
        
        # Clean up before training
        free_memory()
        
        # Try training with aggressive memory management
        try:
            # Train model
            trainer.fit(self, train_loader, val_loader)
            
            # Save final model
            self.save()
        except Exception as e:
            print(f"Error during training: {e}")
            # Try to save model anyway
            try:
                self.save()
                print("Saved model despite error")
            except:
                print("Could not save model")
        
        return self
        
    def predict(self, X: List[Stems]) -> List[Mix]:
        """Generate predictions for new stems"""
        result_mixes = []
        
        # Create a temporary dataset for prediction
        temp_dataset = StemMixDataset(X)
        
        predict_loader = DataLoader(
            temp_dataset,
            batch_size=1,  # Process one at a time to save memory
            shuffle=False,
            num_workers=1,  # Minimize workers for prediction
            pin_memory=True
        )
        
        # Set model to evaluation mode
        self.eval()
        
        # No gradient tracking needed for prediction
        with torch.no_grad():
            for i, (stems, target_mix) in enumerate(predict_loader):
                try:
                    # Move to same device as model
                    stems = stems.to(DEVICE)
                    
                    # Get model prediction
                    final_mix, _, _ = self(stems)
                    
                    # Handle the returned shape [batch, channels, time]
                    # Convert to numpy array
                    if final_mix.dim() == 3:  # [batch, channels, time]
                        # If stereo (2 channels), convert to expected format
                        if final_mix.shape[1] == 2:
                            # Permute to [batch, time, channels]
                            final_mix = final_mix.permute(0, 2, 1)
                    
                    # Convert to numpy and append to results
                    mix_array = final_mix[0].cpu().numpy()
                    
                    # Ensure all output mixes are exactly 30 seconds
                    if len(mix_array) != MAX_SAMPLES:
                        mix_array = pad_to_length(mix_array, MAX_SAMPLES)
                    
                    result_mixes.append(mix_array)
                    
                    # Clean up memory after each prediction
                    del final_mix, stems
                    free_memory()
                        
                except Exception as e:
                    print(f"Error processing item {i}: {e}")
                    # Add a silent audio segment as fallback
                    if target_mix.dim() == 2:  # Stereo
                        fallback = np.zeros((MAX_SAMPLES, 2), dtype=np.float32)
                    else:  # Mono
                        fallback = np.zeros(MAX_SAMPLES, dtype=np.float32)
                    result_mixes.append(fallback)
        
        return result_mixes
        
    def save(self):
        """Save model to disk"""
        save_path = self.config.get('model_save_path', 'models/demucs_finetuned.pt')
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Make sure model is on CPU before saving
        cpu_state_dict = {k: v.cpu() for k, v in self.state_dict().items()}
        
        # Save model state dict
        torch.save({
            'model_state_dict': cpu_state_dict,
            'config': self.config
        }, save_path)
        
        print(f"Model saved to {save_path}")
        
    @classmethod
    def load(cls, model_path: str) -> 'DeepModel':
        """Load model from disk"""
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Create model instance
        model = cls(config=checkpoint.get('config', {}))
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Move model to correct device
        model = model.to(DEVICE)
        
        return model

def load_data(data_dir, max_songs=None):
    """Load songs from a directory, optionally limiting the number"""
    stems_list = []
    
    # Find all song directories
    song_dirs = glob.glob(os.path.join(data_dir, "*"))
    
    if max_songs:
        # Randomly sample if max_songs is specified
        if len(song_dirs) > max_songs:
            song_dirs = random.sample(song_dirs, max_songs)
    
    for song_dir in song_dirs:
        try:
            stems = Stems(song_dir)
            stems_list.append(stems)
            
            # Release memory after appending
            gc.collect()
            
        except Exception as e:
            print(f"Error loading {song_dir}: {e}")
    
    print(f"Loaded {len(stems_list)} songs successfully")
    return stems_list

def main():
    """Standalone training script for the DeepModel.
    
    Loads songs from data/processed/train directory,
    uses 80 songs for training and 10 for validation.
    """
    # Configuration for training
    config = {
        'learning_rate': 1e-5,
        'batch_size': 1,
        'num_workers': 1,
        'max_epochs': 50,
        'early_stopping_patience': 10,
        'model_save_path': 'models/demucs_finetuned.pt',
        'segment_size': 5,
        'mixed_precision': True,
        'gradient_checkpointing': True
    }
    
    # Load songs data
    print("Loading song data from data/processed/train...")
    stems_list = load_data('data/processed/train', max_songs=90)
    
    if len(stems_list) == 0:
        print("Error: No songs found in data/processed/train directory")
        return
    
    print(f"Loaded {len(stems_list)} songs")
    
    # Create and train the model
    model = DeepModel(config)
    
    # Train the model using the fit method
    # The fit method already handles the train/validation split
    print("Starting model training...")
    model.fit(stems_list)
    
    print("Training complete! Model saved to:", config['model_save_path'])

if __name__ == "__main__":
    main()
