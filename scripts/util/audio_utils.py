import numpy as np

def pad_to_length(audio: np.ndarray, target_length: int, stereo: bool = False) -> np.ndarray:
    '''
    Pad or truncate audio array to reach target length.
    
    Args:
        audio: np.ndarray - The audio data to pad/truncate
        target_length: int - The target length to pad/truncate to
        stereo: bool - Whether the audio is stereo (has 2 channels)
        
    Returns:
        np.ndarray - The padded/truncated audio data
    '''
    current_length = len(audio)
    if current_length < target_length:
        # Pad if too short
        padding_length = target_length - current_length
        if stereo or (len(audio.shape) > 1 and audio.shape[1] == 2):
            # Handle stereo audio (samples, 2)
            return np.pad(audio, ((0, padding_length), (0, 0)))
        else:
            # Handle mono audio
            return np.pad(audio, (0, padding_length))
    elif current_length > target_length:
        # Truncate if too long
        if stereo or (len(audio.shape) > 1 and audio.shape[1] == 2):
            # Handle stereo audio
            return audio[:target_length, :]
        else:
            # Handle mono audio
            return audio[:target_length]
    return audio 