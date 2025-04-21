import numpy as np
import soundfile as sf
from scripts.util.audio_utils import pad_to_length
import os

class Stems:
    '''
    Set of stems to be used for training or testing.
    '''
    def __init__(self, path: str, description: str = "No Description", bass_audio: np.ndarray = None, drums_audio: np.ndarray = None, other_audio: np.ndarray = None, vocals_audio: np.ndarray = None, mixture_audio: np.ndarray = None):
        '''
        Args:
            path: str - The path to the stems.
            description: str - The description of the processing steps applied to the wet stems to obtain the dry stems.
            bass_audio: np.ndarray - The bass audio data (optional, for in-memory data).
            drums_audio: np.ndarray - The drums audio data (optional, for in-memory data).
            other_audio: np.ndarray - The other audio data (optional, for in-memory data).
            vocals_audio: np.ndarray - The vocals audio data (optional, for in-memory data).
            mixture_audio: np.ndarray - The mixture audio data (optional, for in-memory data).
        '''
        self.path = path
        self.description = description
        self._sample_rate = None
        self._target_length = None
        
        # Store in-memory data if provided
        if all(x is not None for x in [bass_audio, drums_audio, other_audio, vocals_audio, mixture_audio]):
            self.bass_audio = bass_audio
            self.drums_audio = drums_audio
            self.other_audio = other_audio
            self.vocals_audio = vocals_audio
            self.mixture_audio = mixture_audio
            self._sample_rate = 44100  # Default sample rate for in-memory data
            self._target_length = len(bass_audio)
        else:
            # Initialize paths for lazy loading
            self._bass_path = os.path.join(path, 'bass.wav')
            self._drums_path = os.path.join(path, 'drums.wav')
            self._other_path = os.path.join(path, 'other.wav')
            self._vocals_path = os.path.join(path, 'vocals.wav')
            self._mixture_path = os.path.join(path, 'mixture.wav')

            # Initialize properties to None
            self._bass_audio = None
            self._drums_audio = None
            self._other_audio = None
            self._vocals_audio = None
            self._mixture_audio = None
    @property
    def bass_audio(self) -> np.ndarray:
        if self._bass_audio is None:
            self._load_audio('bass')
        return self._bass_audio

    @property
    def drums_audio(self) -> np.ndarray:
        if self._drums_audio is None:
            self._load_audio('drums')
        return self._drums_audio

    @property
    def other_audio(self) -> np.ndarray:
        if self._other_audio is None:
            self._load_audio('other')
        return self._other_audio

    @property
    def vocals_audio(self) -> np.ndarray:
        if self._vocals_audio is None:
            self._load_audio('vocals')
        return self._vocals_audio

    @property
    def mixture_audio(self) -> np.ndarray:
        if self._mixture_audio is None:
            self._load_audio('mixture')
        return self._mixture_audio

    def _load_audio(self, stem_type: str):
        """Lazily load audio data for a specific stem type."""
        if self._sample_rate is None:
            # Load all stems first to determine their lengths
            stems_data = {}
            for stem in ['bass', 'drums', 'other', 'vocals']:
                path = getattr(self, f'_{stem}_path')
                if os.path.exists(path):
                    data, sr = sf.read(path)
                    stems_data[stem] = data
                    if self._sample_rate is None:
                        self._sample_rate = sr

            # Find the minimum length across all stems
            if stems_data:
                min_length = min(len(data) for data in stems_data.values())
                self._target_length = min_length
            else:
                # Fallback if no stems found
                self._target_length = 0

        path = getattr(self, f'_{stem_type}_path')
        if os.path.exists(path):
            audio, _ = sf.read(path)
            # Truncate to target length rather than padding
            if len(audio) > self._target_length:
                audio = audio[:self._target_length]
            setattr(self, f'_{stem_type}_audio', audio)
        else:
            # Handle missing stems by creating silence
            setattr(self, f'_{stem_type}_audio', np.zeros(self._target_length))

    def __str__(self):
        '''String representation of the stems.'''
        return f"Stems(path={self.path}, description={self.description})"
