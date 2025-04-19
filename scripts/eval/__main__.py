from scripts.eval import evaluate
from scripts.modeling.naive import NaiveModel
from scripts.types import Stems, Mix
from scripts.util.audio_utils import pad_to_length
from typing import List
import os
from scipy.io.wavfile import read

def get_test_stems() -> List[Stems]:
    '''
    Get the stems for evaluation.
    '''
    test_dir = 'data/processed/test'
    stems = []
    
    for song_dir in os.listdir(test_dir):
        if song_dir.startswith('.'):  # Skip hidden files
            continue
            
        path = os.path.join(test_dir, song_dir)
        if os.path.isdir(path):
            print(f'Processing {path}...')
            stems.append(Stems(path=path, description="no effects"))
            
    return stems

def get_test_mixes() -> List[Mix]:
    '''
    Get the mixes for evaluation.
    '''
    test_dir = 'data/processed/test'
    mixes = []
    
    # Calculate target length for 10 minutes at 44.1kHz
    target_length = 44100 * 60 * 10  # 10 minutes in samples
    
    for song_dir in os.listdir(test_dir):
        if song_dir.startswith('.'):  # Skip hidden files
            continue
            
        path = os.path.join(test_dir, song_dir)
        if os.path.isdir(path):
            rate, audio = read(f'{path}/mixture.wav')
            # Pad the audio to 10 minutes
            padded_audio = pad_to_length(audio, target_length)
            mixes.append(padded_audio)
            
    return mixes

def main():
    '''
    Entrypoint for model evaluation.
    '''
    print('Starting evaluation...')
    models = [NaiveModel()]
    X = get_test_stems()
    y = get_test_mixes()
    for model in models:
        print(f'Evaluating {model.__class__.__name__}...')
        loss = evaluate(model, X, y)
        print(f'Loss: {loss}')

if __name__ == '__main__':
    main()
