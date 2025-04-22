from scripts.modeling.model import Model
from scripts.types import Stems, Mix
from typing import List
import numpy as np
import pickle
import xgboost as xgb
import os
import pandas as pd
import random
import optuna
import argparse
from pathlib import Path
import glob
from sklearn.model_selection import train_test_split
from tqdm import trange, tqdm
import multiprocessing
from functools import partial

class TraditionalModel(Model):
    '''
    A traditional model that uses XGBoost with a simple surrounding window of samples as features.
    '''
    def __init__(self, window_size=1024, n_estimators=100, learning_rate=0.1, max_depth=3):
        super().__init__()
        self.model = None
        self.window_size = window_size  # Size of window around each sample (must be even)
        self.max_training_examples = 5000  # Limit training examples
        self.min_examples_per_song = 1     # Minimum examples per song
        
        # XGBoost hyperparameters
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        
        # Ensure window size is even
        if self.window_size % 2 != 0:
            self.window_size += 1
        
    def _extract_sample_features(self, stem, sample_idx):
        """
        Extract features for a specific sample using its surrounding samples.
        
        Args:
            stem: Stems - The input stem
            sample_idx: int - The index of the sample
            
        Returns:
            np.ndarray - The extracted features
        """
        # Get half window size
        half_window = self.window_size // 2
        
        # Get audio data
        bass = stem.bass_audio
        drums = stem.drums_audio
        other = stem.other_audio
        vocals = stem.vocals_audio

        # Calculate start and end indices for the window
        start_idx = max(0, sample_idx - half_window)
        end_idx = min(len(bass), sample_idx + half_window)
        
        # Create a fixed-size window with padding if necessary
        bass_window = np.zeros(self.window_size)
        drums_window = np.zeros(self.window_size)
        other_window = np.zeros(self.window_size)
        vocals_window = np.zeros(self.window_size)
        
        # Fill windows with actual data
        window_start = max(0, half_window - sample_idx)
        window_end = min(self.window_size, window_start + (end_idx - start_idx))
        
        bass_window[window_start:window_end] = np.mean(bass[start_idx:end_idx], axis=1)
        drums_window[window_start:window_end] = np.mean(drums[start_idx:end_idx], axis=1)
        other_window[window_start:window_end] = np.mean(other[start_idx:end_idx], axis=1)
        vocals_window[window_start:window_end] = np.mean(vocals[start_idx:end_idx], axis=1)
        
        # Combine all features
        return np.concatenate([bass_window, drums_window, other_window, vocals_window])
    
    def fit(self, stems: List[Stems], force_new_features=False):
        '''
        Fit the model to the data using a limited number of random examples.
        
        Args:
            stems: List[Stems] - The input stems with mixture targets.
            force_new_features: bool - If True, regenerate features even if cached files exist
        '''
        if not stems:
            raise ValueError("No stems provided for training")
        
        # Create data directory if it doesn't exist
        data_dir = Path('data/traditional')
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Define file paths for saved features
        x_path = data_dir / f'X_train_simple_w{self.window_size}.csv'
        y_path = data_dir / f'y_train_simple_w{self.window_size}.csv'
        
        # Check if saved features exist and can be reused
        if not force_new_features and x_path.exists() and y_path.exists():
            print(f"Loading precomputed features from {data_dir}")
            X = pd.read_csv(x_path).values
            y = pd.read_csv(y_path).values.ravel()
            print(f"Loaded {len(X)} feature vectors for training")
        else:
            # Initialize lists for features and targets
            X = []
            y = []
            
            # Calculate total examples to sample from each stem
            num_stems = len(stems)
            
            # Ensure at least min_examples_per_song from each stem
            examples_per_stem = self.min_examples_per_song
            
            # Calculate remaining examples to distribute randomly
            remaining_examples = self.max_training_examples - (num_stems * examples_per_stem)
            if remaining_examples > 0:
                # Calculate additional examples per stem, ensuring fair distribution
                additional_per_stem = remaining_examples // num_stems
                examples_per_stem += additional_per_stem
            
            print(f"Training with {examples_per_stem} examples per stem, {num_stems} stems")
            
            # Sample examples from each stem
            half_window = self.window_size // 2
            for stem_idx, stem in tqdm(enumerate(stems)):
                # Get length of audio
                audio_length = len(stem.bass_audio)
                
                # Skip if audio is too short
                if audio_length < self.window_size:
                    print(f"Skipping stem {stem_idx}: audio too short ({audio_length} samples)")
                    continue
                
                # Sample random indices, at least half_window size away from edges
                valid_range = (half_window, audio_length - half_window)
                
                # If audio is too short, skip
                if valid_range[0] >= valid_range[1]:
                    print(f"Skipping stem {stem_idx}: valid range empty")
                    continue
                
                # Generate random indices
                num_samples = min(examples_per_stem, valid_range[1] - valid_range[0])
                if num_samples <= 0:
                    print(f"Skipping stem {stem_idx}: no valid samples")
                    continue
                    
                indices = random.sample(
                    range(valid_range[0], valid_range[1]), 
                    num_samples
                )
                
                # Extract features and target for each sample
                stem_features = []
                stem_targets = []
                
                for idx in indices:
                    # Extract features
                    features = self._extract_sample_features(stem, idx)
                    
                    # Get target (mixture value at this index)
                    target = stem.mixture_audio[idx]
                    
                    stem_features.append(features)
                    stem_targets.append(target)
                
                # Add features and targets
                X.extend(stem_features)
                y.extend(stem_targets)

                stem._bass_audio = None
                stem._drums_audio = None
                stem._other_audio = None
                stem._vocals_audio = None
                stem._mixture_audio = None
            
            if len(X) == 0:
                raise ValueError("No valid features extracted from any stems")
            
            # Convert to numpy arrays
            X = np.array(X)
            y = np.array(y)
            
            # Ensure y is the right shape (flattened if needed)
            if len(y.shape) > 1 and y.shape[1] > 1:
                print(f"Target shape before averaging channels: {y.shape}")
                # Average both stereo channels instead of taking just one
                y = np.mean(y, axis=1)
                print(f"Target shape after averaging channels: {y.shape}")
            
            print(f"Collected {len(X)} feature vectors for training")
            
            # Save features to CSV
            print(f"Saving features to {data_dir}")
            pd.DataFrame(X).to_csv(x_path, index=False)
            pd.DataFrame(y, columns=['target']).to_csv(y_path, index=False)
        
        print('Initializing XGBoost model...')
        # Train a single XGBoost model
        self.model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            n_jobs=-1,
            objective='reg:squarederror',
            verbose=1,
            tree_method='hist'
        )
        
        print(f"Training model with {len(X)} examples")
        self.model.fit(X, y)
        print("Model training complete")

    def _process_batch(self, stem, batch_start, batch_end):
        """
        Process a single batch of samples from a stem.
        
        Args:
            stem: Stems - The input stem
            batch_start: int - Start index of the batch
            batch_end: int - End index of the batch
            
        Returns:
            tuple - (batch_indices, batch_predictions)
        """
        # Collect data for this batch
        batch_features = []
        batch_indices = []
        
        # Process each sample in the batch
        for sample_idx in range(batch_start, batch_end):
            # Extract features
            features = self._extract_sample_features(stem, sample_idx)
            batch_features.append(features)
            batch_indices.append(sample_idx)
        
        # If no samples in batch, return empty results
        if not batch_features:
            return batch_indices, []
        
        # Convert to numpy array
        batch_features = np.array(batch_features)
        
        # Predict for batch
        batch_predictions = self.model.predict(batch_features)
        
        return batch_indices, batch_predictions

    def predict(self, stems: List[Stems]) -> List[Mix]:
        '''
        Predict the target variable for every sample in the input stems.
        Uses parallel processing for batch prediction within each stem.
        
        Args:
            stems: List[Stems] - The input stems.
            
        Returns:
            List[Mix] - The predicted target mix.
        '''
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        # Get the number of available CPU cores (leave one for the OS)
        num_cores = max(1, multiprocessing.cpu_count() - 1)
        print(f"Using {num_cores} CPU cores for parallel batch processing")
        
        # Initialize predictions
        predictions = []
        
        # Process each stem
        for stem_idx, stem in enumerate(stems):
            print(f"Predicting for stem {stem_idx+1}/{len(stems)}")
            
            # Get length of audio
            audio_length = len(stem.bass_audio)
            
            # Initialize output mix
            predicted_mix = np.zeros(audio_length)
            
            # Process in batches to save memory
            batch_size = 1000
            
            # Create batch ranges
            batch_ranges = [(i, min(i + batch_size, audio_length)) 
                           for i in range(0, audio_length, batch_size)]
            
            # Create a partial function with the stem parameter fixed
            process_batch_func = partial(self._process_batch, stem)
            
            # Use multiprocessing to process batches in parallel
            with multiprocessing.Pool(processes=num_cores) as pool:
                # Use tqdm to show progress
                batch_results = list(tqdm(
                    pool.starmap(process_batch_func, batch_ranges),
                    total=len(batch_ranges),
                    desc="Processing batches"
                ))
            
            # Apply predictions to output mix
            for batch_indices, batch_predictions in batch_results:
                for i, sample_idx in enumerate(batch_indices):
                    if i < len(batch_predictions):
                        # Extract scalar value
                        pred_value = batch_predictions[i]
                        if hasattr(pred_value, '__len__') and not isinstance(pred_value, (str, bytes)):
                            pred_value = pred_value[0]  # Take first element if it's a sequence
                        predicted_mix[sample_idx] = pred_value
            
            # Apply a smoothing filter to avoid abrupt changes
            window_size = 512  # Adjust as needed
            if audio_length > window_size:
                smoothing_window = np.hanning(window_size)
                smoothing_window = smoothing_window / np.sum(smoothing_window)
                # Use convolution for smooth filtering
                predicted_mix = np.convolve(predicted_mix, smoothing_window, mode='same')
            
            # Normalize to avoid potential clipping
            if np.max(np.abs(predicted_mix)) > 0:
                predicted_mix = predicted_mix / np.max(np.abs(predicted_mix))
            
            # Convert mono to stereo by duplicating the channel
            predicted_mix_stereo = np.column_stack((predicted_mix, predicted_mix))
            
            predictions.append(predicted_mix_stereo)
            print(f"Prediction for stem {stem_idx+1}/{len(stems)} complete")
        
        return predictions
    
    def save(self, path: str = 'models/traditional_simple.pkl'):
        '''
        Save the model to a file.
        
        Args:
            path: str - The path to save the model to.
        '''
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save the model
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'window_size': self.window_size,
                'max_training_examples': self.max_training_examples,
                'min_examples_per_song': self.min_examples_per_song,
                'n_estimators': self.n_estimators,
                'learning_rate': self.learning_rate,
                'max_depth': self.max_depth
            }, f)
        
        print(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str = 'models/traditional_simple.pkl'):
        '''
        Load the model from a file.
        
        Args:
            path: str - The path to load the model from.
            
        Returns:
            TraditionalModel - The loaded model.
        '''
        # Create a new instance
        model = cls()
        
        # Load the model
        with open(path, 'rb') as f:
            data = pickle.load(f)
            model.model = data['model']
            model.window_size = data.get('window_size', 1024)
            model.max_training_examples = data.get('max_training_examples', 5000)
            model.min_examples_per_song = data.get('min_examples_per_song', 1)
            model.n_estimators = data.get('n_estimators', 100)
            model.learning_rate = data.get('learning_rate', 0.1)
            model.max_depth = data.get('max_depth', 3)
        
        print(f"Model loaded from {path}")
        return model


def load_stems_from_directory(directory):
    """
    Load all stems from a directory.
    
    Args:
        directory: str - Path to directory containing stems
        
    Returns:
        List[Stems] - List of loaded stems
    """
    print(f"Loading stems from {directory}")
    stem_paths = glob.glob(os.path.join(directory, '*'))
    stems = []
    
    for path in stem_paths:
        try:
            # Assuming each directory contains one stem set
            if os.path.isdir(path):
                print(f"Loading stem from {path}")
                # Create Stems object - assuming constructor takes path
                stem = Stems(path)
                stems.append(stem)
        except Exception as e:
            print(f"Error loading stem from {path}: {e}")
    
    print(f"Loaded {len(stems)} stems")
    return stems


def objective(trial, train_stems, valid_stems):
    """
    Optuna objective function for hyperparameter tuning.
    
    Args:
        trial: optuna.Trial - Optuna trial object
        train_stems: List[Stems] - Training stems
        valid_stems: List[Stems] - Validation stems
        
    Returns:
        float - Validation error (MSE)
    """
    # Define hyperparameters to tune
    window_size = 512 # trial.suggest_categorical('window_size', [256, 512, 1024, 2048])
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
    max_depth = trial.suggest_int('max_depth', 2, 10)
    
    # Create and train model
    model = TraditionalModel(
        window_size=window_size,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth
    )
    
    # Adjust sample limits for faster tuning
    model.max_training_examples = 2000  # Use fewer examples for faster tuning
    model.min_examples_per_song = 1
    
    # Train model
    model.fit(train_stems)
    
    # For validation, check if validation features are already computed for this window size
    data_dir = Path('data/traditional')
    x_val_path = data_dir / f'X_val_simple_w{window_size}.csv'
    y_val_path = data_dir / f'y_val_simple_w{window_size}.csv'
    
    if x_val_path.exists() and y_val_path.exists():
        print(f"Loading precomputed validation features for window size {window_size}")
        
        # Load validation data
        X_val = pd.read_csv(x_val_path).values
        y_val = pd.read_csv(y_val_path).values.ravel()
        
        # Predict with model
        y_pred = model.model.predict(X_val)
        
        # Calculate MSE
        mse = np.mean((y_val - y_pred) ** 2)
        
        return mse
    
    # Calculate validation samples (20% of training samples)
    validation_max_examples = int(model.max_training_examples * 0.2)
    num_valid_stems = len(valid_stems)
    
    # Ensure at least min_examples_per_song from each stem
    examples_per_valid_stem = model.min_examples_per_song
    
    # Calculate remaining examples to distribute randomly
    remaining_examples = validation_max_examples - (num_valid_stems * examples_per_valid_stem)
    if remaining_examples > 0:
        # Calculate additional examples per stem, ensuring fair distribution
        additional_per_stem = remaining_examples // num_valid_stems
        examples_per_valid_stem += additional_per_stem
    
    print(f"Validating with {examples_per_valid_stem} examples per stem, {num_valid_stems} stems")
    
    # Store validation features for future use
    X_val = []
    y_val = []
    
    # Sample examples from each validation stem
    half_window = model.window_size // 2
    for stem_idx, stem in enumerate(valid_stems):
        # Get length of audio
        audio_length = len(stem.bass_audio)
        
        # Skip if audio is too short
        if audio_length < model.window_size:
            print(f"Skipping validation stem {stem_idx}: audio too short ({audio_length} samples)")
            continue
        
        # Sample random indices, at least half_window size away from edges
        valid_range = (half_window, audio_length - half_window)
        
        # If audio is too short, skip
        if valid_range[0] >= valid_range[1]:
            print(f"Skipping validation stem {stem_idx}: valid range empty")
            continue
        
        # Generate random indices
        num_samples = min(examples_per_valid_stem, valid_range[1] - valid_range[0])
        if num_samples <= 0:
            print(f"Skipping validation stem {stem_idx}: no valid samples")
            continue
            
        indices = random.sample(
            range(valid_range[0], valid_range[1]), 
            num_samples
        )
        
        # Extract features and target for each sample
        for idx in indices:
            # Extract features
            features = model._extract_sample_features(stem, idx)
            
            # Get target (mixture value at this index)
            target = stem.mixture_audio[idx]
            
            X_val.append(features)
            y_val.append(target)
    
    if not X_val:
        return float('inf')  # Return infinity if no validation samples
    
    # Convert to numpy arrays
    X_val = np.array(X_val)
    y_val = np.array(y_val)
    
    # Ensure y_val is the right shape (flattened if needed)
    if len(y_val.shape) > 1 and y_val.shape[1] > 1:
        print(f"Validation target shape before averaging channels: {y_val.shape}")
        # Average both stereo channels instead of taking just one
        y_val = np.mean(y_val, axis=1)
        print(f"Validation target shape after averaging channels: {y_val.shape}")
    
    # Save validation features for future trials - with window size in filename
    data_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(X_val).to_csv(x_val_path, index=False)
    pd.DataFrame(y_val, columns=['target']).to_csv(y_val_path, index=False)
    print(f"Saved {len(X_val)} validation features to {data_dir} for window size {window_size}")
    
    # Predict with model
    y_pred = model.model.predict(X_val)
    
    # Calculate MSE
    mse = np.mean((y_val - y_pred) ** 2)
    
    return mse


def main():
    """
    Main function to train the model using data from processed directories.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train a traditional model for music mixing")
    parser.add_argument('--use_validation', action='store_true', 
                        help='Use separate validation directory if it exists')
    parser.add_argument('--train_dir', default='data/processed/train',
                        help='Directory containing training stems')
    parser.add_argument('--valid_dir', default='data/processed/valid',
                        help='Directory containing validation stems (if exists)')
    parser.add_argument('--output_model', default='models/traditional_simple.pkl',
                        help='Path to save the trained model')
    parser.add_argument('--n_trials', type=int, default=20,
                        help='Number of optimization trials for Optuna')
    parser.add_argument('--max_examples', type=int, default=5000,
                        help='Maximum number of training examples')
    args = parser.parse_args()
    
    # Check if directories exist
    train_dir = Path(args.train_dir)
    valid_dir = Path(args.valid_dir)
    
    if not train_dir.exists():
        raise ValueError(f"Training directory {train_dir} does not exist")
    
    # Load training stems
    train_stems = load_stems_from_directory(train_dir)
    
    if len(train_stems) == 0:
        raise ValueError(f"No training stems found in {train_dir}")
    
    # Determine validation strategy
    use_separate_validation = args.use_validation and valid_dir.exists()
    
    if use_separate_validation:
        # Load validation stems from separate directory
        print(f"Using separate validation directory: {valid_dir}")
        valid_stems = load_stems_from_directory(valid_dir)
        
        if len(valid_stems) == 0:
            print(f"Warning: No validation stems found in {valid_dir}. Falling back to train-test split.")
            use_separate_validation = False
    
    if not use_separate_validation:
        # Split training data for validation
        print("Using 80-20 train-validation split")
        train_stems, valid_stems = train_test_split(train_stems, test_size=0.2, random_state=42)
        print(f"Split into {len(train_stems)} training stems and {len(valid_stems)} validation stems")
    
    # Create Optuna study for hyperparameter optimization
    print("Starting hyperparameter optimization with Optuna")
    study = optuna.create_study(direction='minimize')
    
    # Run optimization
    study.optimize(lambda trial: objective(trial, train_stems, valid_stems), n_trials=args.n_trials)
    
    # Get best parameters
    best_params = study.best_params
    print("Best hyperparameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    # Train final model with best hyperparameters
    print("Training final model with best hyperparameters")
    final_model = TraditionalModel(
        window_size=512, # best_params['window_size'],
        n_estimators=best_params['n_estimators'],
        learning_rate=best_params['learning_rate'],
        max_depth=best_params['max_depth']
    )
    
    # Set training examples limit
    final_model.max_training_examples = args.max_examples
    final_model.min_examples_per_song = 1
    
    # Train on all training stems - force regeneration of features to use the full dataset
    final_model.fit(train_stems, force_new_features=True)
    
    # Save the model
    final_model.save(args.output_model)
    print(f"Model training complete. Model saved to {args.output_model}")


if __name__ == "__main__":
    main()
