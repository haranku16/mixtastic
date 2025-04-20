from scripts.modeling.model import Model
from scripts.types import Stems, Mix
from typing import List
import numpy as np
from scripts.util.mix_stems import process_stems

class NaiveModel(Model):
    '''
    Naive model to predict the mix from the stems.
    '''
    def __init__(self):
        super().__init__()

    def fit(self, stems: List[Stems]):
        pass

    def predict(self, X: List[Stems]) -> List[Mix]:
        mixes = []
        for x in X:
            # Collect all the audio stems
            audio_arrays = []
            
            # Ensure all stems are loaded
            audio_arrays.append(x.bass_audio)
            audio_arrays.append(x.drums_audio)
            audio_arrays.append(x.other_audio)
            audio_arrays.append(x.vocals_audio)
            
            # Use process_stems from mix_stems.py to handle the mixing
            mixed = process_stems(audio_arrays, normalize=True, normalize_factor=0.9)
            
            # Add to results
            mixes.append(mixed)
            
            # Release memory
            x._bass_audio = None
            x._drums_audio = None
            x._other_audio = None
            x._vocals_audio = None
                
        return mixes
