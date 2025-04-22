import numpy as np

def blend(stem1, stem2, dB1, dB2):
    """Blend two audio stems with independent gain control
    
    Args:
        stem1: NumPy array of first audio stem
        stem2: NumPy array of second audio stem
        dB1: Gain for stem1 in decibels
        dB2: Gain for stem2 in decibels
    
    Returns:
        Normalized blended audio array
    """
    # Convert dB to linear amplitude
    lin_gain1 = 10 ** (dB1 / 20)
    lin_gain2 = 10 ** (dB2 / 20)
    
    # Apply gain scaling
    scaled_stem1 = stem1 * lin_gain1
    scaled_stem2 = stem2 * lin_gain2
    
    # Mix stems
    blended = scaled_stem1 + scaled_stem2
    
    # Prevent clipping with peak normalization
    peak = np.max(np.abs(blended))
    return blended / peak if peak > 1.0 else blended
