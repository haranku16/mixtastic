import os
import zipfile
import shutil

def extract(zip_path: str, force: bool = False):
    '''
    Extract data from the MUSDB18HQ zip file.
    
    Args:
        zip_path (str): Path to the MUSDB18HQ zip file
        force (bool): If True, force overwrite existing data
    '''
    output_dir = "data/musdb18hq"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if directory is empty or force flag is set
    if not force and os.path.exists(output_dir) and os.listdir(output_dir):
        print(f"Data already exists in {output_dir}. Use --force to overwrite.")
        return
    
    print(f"Extracting {zip_path} to {output_dir}...")
    
    # Extract the zip file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)
    
    print("Extraction complete!")
