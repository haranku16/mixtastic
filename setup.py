import argparse
from scripts.etl.extract import extract
from scripts.etl.transform import transform

def main():
    '''
    Entry point for the workspace setup script.
    '''
    parser = argparse.ArgumentParser(description='Process MUSDB18HQ dataset')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Extract command
    extract_parser = subparsers.add_parser('extract', help='Extract MUSDB18HQ dataset')
    extract_parser.add_argument('zip_path', help='Path to the MUSDB18HQ zip file')
    extract_parser.add_argument('-f', '--force', action='store_true', 
                              help='Force overwrite existing data')
    
    # Transform command
    transform_parser = subparsers.add_parser('transform', help='Transform extracted data')
    transform_parser.add_argument('-f', '--force', action='store_true',
                                help='Force overwrite existing data')
    
    args = parser.parse_args()
    
    if args.command == 'extract':
        extract(args.zip_path, args.force)
    elif args.command == 'transform':
        transform(args.force)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
