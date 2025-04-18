import argparse
from scripts.etl.extract import extract

def main():
    '''
    Entry point for the workspace setup script.
    '''
    parser = argparse.ArgumentParser(description='Extract MUSDB18HQ dataset')
    parser.add_argument('zip_path', help='Path to the MUSDB18HQ zip file')
    parser.add_argument('-f', '--force', action='store_true', 
                       help='Force overwrite existing data')
    
    args = parser.parse_args()
    extract(args.zip_path, args.force)

if __name__ == "__main__":
    main()
