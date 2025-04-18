from scripts.modeling.model import Model
from scripts.types import Stems, Mix
from typing import List
from scipy.io.wavfile import read
from scripts.util.audio_utils import pad_to_length
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
            sample_rate, mix = read(f'{x.path}/mixture.wav')
            padded_mix = pad_to_length(mix, sample_rate * 60 * 10)
            mixes.append(padded_mix)
        return mixes
