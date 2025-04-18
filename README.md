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

5. Extract the dataset:
```bash
python setup.py path/to/musdb18hq.zip
```

### Setup Script Options

The setup script (`setup.py`) provides the following command line options:

- `zip_path` (required): Path to the MUSDB18HQ zip file
- `-f, --force` (optional): Force overwrite existing data in the output directory

Example usage:
```bash
# Basic usage
python setup.py path/to/musdb18hq.zip

# Force overwrite existing data
python setup.py path/to/musdb18hq.zip --force
```

The script will extract the dataset to `data/musdb18hq/`. If the directory already contains data, it will only proceed if the `--force` flag is used.

### Notes
- Remember to activate the virtual environment whenever you work on the project
- To deactivate the virtual environment when you're done, simply type `deactivate` in your terminal
