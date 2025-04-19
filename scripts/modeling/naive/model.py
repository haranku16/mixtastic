from scripts.modeling.model import Model
from scripts.types import Stems, Mix
from typing import List
import numpy as np
from scipy.io.wavfile import write

class NaiveModel(Model):
    '''
    Naive model to predict the mix from the stems.
    '''
    def __init__(self):
        super().__init__()

    def fit(self, X: List[Stems], y: List[Mix]):
        pass

    def predict(self, X: List[Stems]) -> List[Mix]:
        mixes = []
        for x in X:
            # Process each stem one at a time and sum them
            mix = np.zeros_like(x.bass_audio)  # Initialize with zeros
            
            # Process and add each stem, then release it from memory
            mix += x.bass_audio
            x._bass_audio = None  # Release memory
            
            mix += x.drums_audio
            x._drums_audio = None  # Release memory
            
            mix += x.other_audio
            x._other_audio = None  # Release memory
            
            mix += x.vocals_audio
            x._vocals_audio = None  # Release memory
            
            # Normalize to prevent clipping
            mix = mix / np.max(np.abs(mix))
            
            mixes.append(mix)
                
        return mixes
