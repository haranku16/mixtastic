# MixTastic
AI-powered audio engineering for beginner musicians

## Description
MixTastic offers affordable audio engineering solutions for beginner musicians who cannot afford professional services. This project aims to remove barriers to entry for new artists by providing AI-powered tools that assist with basic mixing and mastering tasks.

## Ethics Statement
This project is designed for **educational purposes only** and not for commercial use. MixTastic is not a replacement for professional audio engineers and will never reach that level of quality. It simply aims to help beginner musicians who cannot afford professional audio engineering services.

### Data Attribution
The example songs used in this project are sourced from [MUSDB18-HQ](https://zenodo.org/records/3338373).

Attribution:

```
@misc{MUSDB18HQ,
  author       = {Rafii, Zafar and
                  Liutkus, Antoine and
                  Fabian-Robert St{\"o}ter and
                  Mimilakis, Stylianos Ioannis and
                  Bittner, Rachel},
  title        = {{MUSDB18-HQ} - an uncompressed version of MUSDB18},
  month        = dec,
  year         = 2019,
  doi          = {10.5281/zenodo.3338373},
  url          = {https://doi.org/10.5281/zenodo.3338373}
}
```

These songs are covered by Creative Commons licenses (CC-BY-NC 3.0 or CC-BY-NC 4.0).

## Past Approaches
Some past approaches that inspired my deep learning approach are:

1. [FxNorm-Automix](https://github.com/sony/FxNorm-automix) (simulates dry stems from wet stems for its own mixing model)
2. [demucs](https://github.com/facebookresearch/demucs) (hybrid Wave U-Net/transformer architecture for source separation, available through torchaudio)
3. [denoiser](https://github.com/facebookresearch/denoiser) (based on demucs, specialized for denoising audio)

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
The transformation process applies targeted audio augmentations to create a more diverse training dataset. It:
- Processes both train and test splits
- Creates 2 augmented versions of each song
- Applies the following augmentations with varying probabilities:
  - Gaussian noise (SNR: 15-40 dB, p=0.5)
  - Short noises from the noises/ directory (SNR: 10-30 dB, p=0.3)
  - Air absorption simulation (temperature: 10-20°C, humidity: 40-70%, distance: 5-30m, p=0.3)
  - Clipping distortion (percentile threshold: 5-20%, p=0.3)
  - Gain adjustment (±6 dB, p=0.4)
  - Gain transitions (±6 dB, duration: 0.3-0.8s, p=0.3)
  - MP3 compression (128-320 kbps, p=0.3)
  - Room simulation (size: 3-6m x 3-6m x 2.5-3.5m, absorption: 0.4-0.9, p=0.3)

The transformed data is saved in data/processed with songs having ORIGINAL and AUG1-2 versions.

## Evaluation Strategy

The evaluation strategy employs a MultiResolutionSTFTLoss from the auraloss library:
- Uses large FFT sizes (8192, 16384, 32768) suitable for longer audio segments
- Evaluates mixes in 30-second chunks at 44.1kHz
- Compares predictions against the professionally mixed target stems
- Provides a comprehensive frequency-domain assessment of mix quality
- Considers both magnitude and phase information

This metric aligns with how humans perceive audio quality, capturing both time and frequency domain characteristics of the mixes.

To run evaluation, run the following command:

```bash
python -m scripts.eval
```

**Note:** In order to evaluate the `TraditionalModel`, modify the script to change `num_songs` to 6 and comment out the other two models.
Uncomment the `TraditionalModel`.

## Modeling Approaches

### Naive Approach
The Naive model implements a simple approach to mixing audio stems:
- Takes the raw stems (bass, drums, vocals, other)
- Creates a mix by simply averaging all stems together
- Applies normalization to avoid clipping (0.9 factor)
- No learning or parameters involved

This approach establishes a baseline performance level and helps identify the value added by more complex models.

**Evaluation Result:** Loss: 29,374.21427734375 (from all 50 test songs)

### Deep Learning Approach
The Deep Learning model leverages transfer learning with [Facebook Research's Demucs](https://github.com/facebookresearch/demucs):
- Fine-tunes the pre-trained Demucs model (HDEMUCS_HIGH_MUSDB_PLUS) on augmented MUSDB18-HQ data
- Uses the model to process dry stems and simulate wet stems (professionally mixed)
- Training with combined L1 and spectral loss functions
- Process flow:
  1. Input stems are processed through the Demucs model
  2. Model learns optimal mixing parameters
  3. Final prediction blends the deep learning result with the naive mix to reduce artifacts
  4. Uses gradient checkpointing and mixed precision to manage memory usage

**Evaluation Result:** Loss: 18,993.41712565104 (from all 50 test split songs)

Attribution:

```
@inproceedings{rouard2022hybrid,
  title={Hybrid Transformers for Music Source Separation},
  author={Rouard, Simon and Massa, Francisco and D{\'e}fossez, Alexandre},
  booktitle={ICASSP 23},
  year={2023}
}

@inproceedings{defossez2021hybrid,
  title={Hybrid Spectrogram and Waveform Source Separation},
  author={D{\'e}fossez, Alexandre},
  booktitle={Proceedings of the ISMIR 2021 Workshop on Music Source Separation},
  year={2021}
}
```

#### Training
To train the model, execute the following command. Alternatively, run the notebook from the notebooks/ directory in Google Colab.

```bash
python -m scripts.modeling.deep.model
```

### Traditional Approach
The Traditional model uses XGBoost with a window-based feature approach:
- Creates features from the surrounding audio context for each sample
- Applies polynomial feature transformation
- Uses Optuna for hyperparameter tuning with these parameters:
  - Window size: 512 samples
  - Number of estimators: 50-300
  - Learning rate: 0.01-0.3
  - Max depth: 2-10
- Training:
  - Limits to 5000 training examples for tractability
  - Parallel batch processing during prediction
  - Applies smoothing filters to final output
  
This approach is computationally expensive and slow, and does not produce valid outputs.

**Evaluation Result:** Loss: 44,652.16219075521 (only 6 songs due to model slowness)

#### Training

To train the model, execute the following command.

```bash
python -m scripts.modeling.traditional.model
```

## Requirements
- Python 3.12
- audiomentations >= 0.40.0
- auraloss >= 0.4.0
- matplotlib >= 3.10.1
- optuna >= 4.3.0
- pandas >= 2.2.3
- psutil >= 7.0.0
- pyAudioAnalysis >= 0.3.14
- pyroomacoustics >= 0.1.4
- pytorch-lightning >= 2.5.1
- scikit-learn >= 1.6.1
- scipy >= 1.15.2
- soundfile >= 0.13.1
- streamlit >= 1.44.1
- torch >= 2.6.0
- torchaudio >= 0.12.1
- tqdm >= 2.2.3
- tensorboard >= 2.19.0
- transformers >= 4.51.3
- xgboost >= 3.0.0

## Presentation
You can find the presentation for this project [here on YouTube](https://youtu.be/W47iex7USxg).

## Demo Application
You can find the Streamlit application, deployed to Google Cloud Platform, [here](https://mixtastic-518487429487.us-central1.run.app/).

## License
MIT

## Acknowledgement

This project was implemented with the assistance of Cursor AI using the Claude 3.7 Sonnet model.
