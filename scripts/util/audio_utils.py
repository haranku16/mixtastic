import numpy as np

def pad_to_length(audio: np.ndarray, target_length: int) -> np.ndarray:
    '''
    Pad audio array with zeros to reach target length.
    
    Args:
        audio: np.ndarray - The audio data to pad
        target_length: int - The target length to pad to
        
    Returns:
        np.ndarray - The padded audio data
    '''
    current_length = len(audio)
    if current_length < target_length:
        padding_length = target_length - current_length
        # Handle both mono and stereo cases
        if len(audio.shape) == 1:
            return np.pad(audio, (0, padding_length))
        else:
            return np.pad(audio, ((0, padding_length), (0, 0)))
    return audio 