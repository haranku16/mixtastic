import os
import logging
from pathlib import Path
import shutil

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fix_links.log'),
        logging.StreamHandler()
    ]
)

def fix_links():
    """Fix broken symbolic links in the processed data directories."""
    # Process train and test splits
    for split in ['train', 'test']:
        input_dir = Path(f'data/musdb18hq/{split}').resolve()
        output_dir = Path(f'data/processed/{split}').resolve()
        
        if not output_dir.exists():
            logging.error(f"Output directory {output_dir} does not exist")
            continue
            
        # First, fix all ORIGINAL directories
        original_dirs = [d for d in output_dir.iterdir() if d.is_dir() and 'ORIGINAL' in d.name]
        for song_dir in original_dirs:
            try:
                # Get the original song name (remove ORIGINAL suffix)
                song_name = song_dir.name.split(' ORIGINAL')[0]
                original_song_dir = input_dir / song_name
                
                if not original_song_dir.exists():
                    logging.error(f"Original song directory {original_song_dir} does not exist")
                    continue
                
                # Process all .wav files in the directory
                for wav_file in song_dir.glob('*.wav'):
                    try:
                        # Check if the file is a broken symlink
                        if wav_file.is_symlink() and not wav_file.exists():
                            # Remove the broken symlink
                            wav_file.unlink()
                            logging.info(f"Removed broken symlink: {wav_file}")
                        
                        # If the file doesn't exist (either was a broken symlink or never existed)
                        if not wav_file.exists():
                            # Link to the original file in musdb18hq
                            original_file = original_song_dir / wav_file.name
                            if original_file.exists():
                                os.symlink(original_file.resolve(), wav_file)
                                logging.info(f"Created symlink for {wav_file.name} in ORIGINAL directory: {wav_file}")
                    except Exception as e:
                        logging.error(f"Error processing file {wav_file}: {str(e)}")
                        continue
                        
            except Exception as e:
                logging.error(f"Error processing directory {song_dir}: {str(e)}")
                continue
        
        # Then, fix all AUG directories
        aug_dirs = [d for d in output_dir.iterdir() if d.is_dir() and 'AUG' in d.name]
        for song_dir in aug_dirs:
            try:
                # Get the original song name (remove AUG suffix)
                song_name = song_dir.name.split(' AUG')[0]
                original_dir = output_dir / f"{song_name} ORIGINAL"
                
                if not original_dir.exists():
                    logging.error(f"Original directory {original_dir} does not exist")
                    continue
                
                # Ensure mixture.wav exists in AUG directory
                mixture_path = song_dir / 'mixture.wav'
                if not mixture_path.exists():
                    original_mixture = original_dir / 'mixture.wav'
                    if original_mixture.exists():
                        os.symlink(original_mixture.resolve(), mixture_path)
                        logging.info(f"Created missing mixture.wav symlink in AUG directory: {mixture_path}")
                
                # Process all .wav files in the directory
                for wav_file in song_dir.glob('*.wav'):
                    try:
                        # Check if the file is a broken symlink
                        if wav_file.is_symlink() and not wav_file.exists():
                            # Remove the broken symlink
                            wav_file.unlink()
                            logging.info(f"Removed broken symlink: {wav_file}")
                        
                        # If the file doesn't exist (either was a broken symlink or never existed)
                        if not wav_file.exists():
                            if wav_file.name == 'mixture.wav':
                                # For mixture.wav, link to the one in ORIGINAL directory
                                original_mixture = original_dir / 'mixture.wav'
                                if original_mixture.exists():
                                    os.symlink(original_mixture.resolve(), wav_file)
                                    logging.info(f"Created symlink for mixture.wav in AUG directory: {wav_file}")
                            else:
                                # For other stems, link to the original in musdb18hq
                                original_song_dir = input_dir / song_name
                                original_stem = original_song_dir / wav_file.name
                                if original_stem.exists():
                                    os.symlink(original_stem.resolve(), wav_file)
                                    logging.info(f"Created symlink for stem: {wav_file}")
                    except Exception as e:
                        logging.error(f"Error processing file {wav_file}: {str(e)}")
                        continue
                        
            except Exception as e:
                logging.error(f"Error processing directory {song_dir}: {str(e)}")
                continue

if __name__ == '__main__':
    fix_links() 