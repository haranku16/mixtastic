import numpy as np
from scipy.io.wavfile import read
from scripts.util.audio_utils import pad_to_length

class Stems:
    '''
    Set of stems to be used for training or testing.
    '''
    def __init__(self, path: str, description: str, bass_audio: np.ndarray = None, drums_audio: np.ndarray = None, other_audio: np.ndarray = None, vocals_audio: np.ndarray = None):
        '''
        Args:
            path: str - The path to the stems.
            description: str - The description of the processing steps applied to the wet stems to obtain the dry stems.
            bass_audio: np.ndarray - The bass audio data.
            drums_audio: np.ndarray - The drums audio data.
            other_audio: np.ndarray - The other audio data.
            vocals_audio: np.ndarray - The vocals audio data.
        '''
        self.path = path
        self.description = description
        if bass_audio is None or drums_audio is None or other_audio is None or vocals_audio is None:
            # Read the audio files
            sample_rate, self.bass_audio = read(f'{path}/bass.wav')
            _, self.drums_audio = read(f'{path}/drums.wav')
            _, self.other_audio = read(f'{path}/other.wav')
            _, self.vocals_audio = read(f'{path}/vocals.wav')
            
            # Calculate required length for 10 minutes
            target_length = sample_rate * 60 * 10  # 10 minutes in samples
            
            # Pad each stem with zeros to reach 10 minutes
            self.bass_audio = pad_to_length(self.bass_audio, target_length)
            self.drums_audio = pad_to_length(self.drums_audio, target_length)
            self.other_audio = pad_to_length(self.other_audio, target_length)
            self.vocals_audio = pad_to_length(self.vocals_audio, target_length)
        else:
            self.bass_audio = bass_audio
            self.drums_audio = drums_audio
            self.other_audio = other_audio
            self.vocals_audio = vocals_audio

    def __str__(self):
        '''String representation of the stems.'''
        return f"Stems(title={self.title}, description={self.description})"
