# MixTastic
AI-powered audio engineering for beginner musicians

## Setup

### Prerequisites
- Python 3.12
- pip (Python package installer)

### Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/mixtastic.git
cd mixtastic
```

2. Create and activate a virtual environment:
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate
```

3. Install required Python packages:
```bash
pip install -r requirements.txt
```

4. Download the MUSDB18-HQ dataset:
   - Visit [MUSDB18-HQ on Zenodo](https://zenodo.org/records/3338373)
   - Download the `musdb18hq.zip` file (approximately 22.7 GB)

5. Extract and transform the dataset:
```bash
# Extract the dataset
python setup.py extract path/to/musdb18hq.zip

# Transform the extracted data (optional)
python setup.py transform
```

### Setup Script Commands

The setup script (`setup.py`) provides two main commands:

1. `extract` - Extract the MUSDB18HQ dataset:
   ```bash
   python setup.py extract path/to/musdb18hq.zip [-f|--force]
   ```
   - `zip_path` (required): Path to the MUSDB18HQ zip file
   - `-f, --force` (optional): Force overwrite existing data

2. `transform` - Transform the extracted data:
   ```bash
   python setup.py transform [-f|--force]
   ```
   - `-f, --force` (optional): Force overwrite existing data

The script will extract the dataset to `data/musdb18hq/` and transform it to `data/processed/`. If these directories already contain data, they will only be overwritten if the `--force` flag is used.

### Notes
- Remember to activate the virtual environment whenever you work on the project
- To deactivate the virtual environment when you're done, simply type `deactivate` in your terminal

## Data Processing

### Extraction
The extraction process:
- Unzips the MUSDB18HQ dataset
- Organizes the data into train/test splits
- Creates a data/musdb18hq directory with the following structure:
  ```
  data/musdb18hq/
  ├── train/
  │   ├── Artist - Song/
  │   │   ├── mixture.wav
  │   │   ├── vocals.wav
  │   │   ├── drums.wav
  │   │   ├── bass.wav
  │   │   └── other.wav
  │   └── ...
  └── test/
      └── ... (similar structure)
  ```

### Transformation
The transformation process applies various audio augmentations to create a more diverse training dataset. It:
- Processes both train and test splits
- Creates 50 augmented versions of each stem (excluding mixture.wav)
- Applies the following augmentations with 50% probability each:
  - Gaussian noise (SNR: 5-40 dB)
  - Short noises from the noises/ directory
  - Air absorption simulation
  - Impulse response convolution
  - Bit crushing (4-16 bits)
  - Clipping distortion
  - Gain adjustment (±12 dB)
  - Gain transitions
  - MP3 compression (32-320 kbps)
  - Pitch shifting (±4 semitones)
  - Polarity inversion
  - Room simulation
  - Time shifting

The transformed data is saved in data/processed with the same structure as the input data, with augmented files named as original_name_aug{i}.wav.

## Requirements
- Python 3.8+
- audiomentations
- soundfile
- tqdm
- numpy

## License
MIT
