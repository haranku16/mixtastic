from scripts.eval import evaluate
from scripts.modeling.naive import NaiveModel
from scripts.modeling.deep import DeepModel
from scripts.types import Stems
from typing import List
import os
from scripts.modeling.traditional.model import TraditionalModel

def get_test_stems(num_songs: int = None) -> List[Stems]:
    '''
    Get the stems for evaluation.
    
    Args:
        num_songs: Optional limit on the number of songs to process. If None, process all songs.
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
            
        if num_songs is not None and len(stems) >= num_songs:
            break
            
    return stems

def main():
    '''
    Entrypoint for model evaluation.
    '''
    print('Starting evaluation...')
    models = [
        NaiveModel(),
        DeepModel.load('models/demucs_finetuned.pt'),
        # TraditionalModel.load('models/traditional_simple.pkl') # RUN THIS WITH ONLY A FEW SONGS
    ]

    num_songs = None # Change to 6 to run traditional model
    
    stems = get_test_stems(num_songs)
    # Extract mixture_audio from stems to use as targets
    targets = [stem.mixture_audio for stem in stems]
    
    for model in models:
        print(f'Evaluating {model.__class__.__name__}...')
        loss = evaluate(model, stems, targets)
        print(f'Loss: {loss}')

if __name__ == '__main__':
    main()
