import os
import time
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf
import torch

from scripts.types import Stems
from scripts.modeling.naive.model import NaiveModel
from scripts.modeling.deep.model import DeepModel

# Set page configuration
st.set_page_config(page_title="🎸 MixTastic", layout="wide")

# App title
st.title("🎸 MixTastic")

# Description and ethics statement
st.markdown("""
**Affordable audio engineering for beginner musicians**

*Ethics Statement:* This project is designed for educational purposes only and not for commercial use. 
The example songs are sourced from MUSDB18-HQ (https://zenodo.org/records/3338373) with proper attribution.
The songs are covered by Creative Commons licenses (CC-BY-NC 3.0 or CC-BY-NC 4.0).

This tool is not a replacement for professional audio engineers and will never reach that level of quality.
MixTastic simply aims to remove a barrier for beginner musicians who cannot afford professional audio engineering.
""")

# Get list of example songs
examples_dir = "examples"
song_dirs = [d for d in os.listdir(examples_dir) if os.path.isdir(os.path.join(examples_dir, d))]
song_dirs.sort()

# Function to generate spectrogram
def plot_spectrogram(y, sr, title="Spectrogram"):
    fig, ax = plt.subplots(figsize=(10, 4))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    img = librosa.display.specshow(D, y_axis='log', x_axis='time', sr=sr, ax=ax)
    ax.set_title(title)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    return fig

# Function to generate mix with timing
def generate_mix_with_timing(model, stems):
    start_time = time.time()
    predicted_mixes = model.predict([stems])
    end_time = time.time()
    generation_time = end_time - start_time
    return predicted_mixes[0], generation_time

# Dropdown for song selection
selected_song = st.selectbox("Select a song:", song_dirs)

if selected_song:
    song_path = os.path.join(examples_dir, selected_song)
    
    # Create Stems object for the selected song
    stems = Stems(path=song_path, description="Selected song stems")
    
    # Display stem playback
    st.header("Stems")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.subheader("Bass")
        st.audio(stems._bass_path, format="audio/wav")
    
    with col2:
        st.subheader("Drums")
        st.audio(stems._drums_path, format="audio/wav")
    
    with col3:
        st.subheader("Vocals")
        st.audio(stems._vocals_path, format="audio/wav")
    
    with col4:
        st.subheader("Other")
        st.audio(stems._other_path, format="audio/wav")
    
    # Display reference mixture
    st.header("Reference Mixture")
    
    left_col, right_col = st.columns([1, 1])
    
    with left_col:
        st.audio(stems._mixture_path, format="audio/wav")
    
    # Reference mixture spectrogram
    mixture_audio, sr = sf.read(stems._mixture_path)
    if mixture_audio.ndim > 1:
        # Convert stereo to mono for spectrogram
        mixture_audio_mono = mixture_audio.mean(axis=1)
    else:
        mixture_audio_mono = mixture_audio
    
    with right_col:
        st.pyplot(plot_spectrogram(mixture_audio_mono, sr, "Reference Mixture Spectrogram"))
    
    # NaiveModel
    st.header("NaiveModel Prediction")
    left_col, right_col = st.columns([1, 1])
    
    with left_col:
        with st.spinner("Generating mix with NaiveModel..."):
            # Load model and generate mix
            naive_model = NaiveModel()
            naive_mix, naive_time = generate_mix_with_timing(naive_model, stems)
            
            # Save mix to temporary file for playback
            naive_mix_path = "output/naive_mix_temp.wav"
            os.makedirs(os.path.dirname(naive_mix_path), exist_ok=True)
            sf.write(naive_mix_path, naive_mix, sr)
        
        st.write(f"Generation time: {naive_time:.2f} seconds")
        st.audio(naive_mix_path, format="audio/wav")
    
    with right_col:
        # Create spectrogram for NaiveModel mix
        if naive_mix.ndim > 1:
            naive_mix_mono = naive_mix.mean(axis=1)
        else:
            naive_mix_mono = naive_mix
        
        st.pyplot(plot_spectrogram(naive_mix_mono, sr, "NaiveModel Mix Spectrogram"))
    
    # DeepModel
    st.header("DeepModel Prediction")
    left_col, right_col = st.columns([1, 1])
    
    with left_col:
        try:
            with st.spinner("Generating mix with DeepModel..."):
                # Load DeepModel
                deep_model = DeepModel.load("models/demucs_finetuned.pt")
                
                # Generate mix
                deep_mix, deep_time = generate_mix_with_timing(deep_model, stems)
                
                # Save mix to temporary file for playback
                deep_mix_path = "output/deep_mix_temp.wav"
                os.makedirs(os.path.dirname(deep_mix_path), exist_ok=True)
                sf.write(deep_mix_path, deep_mix, sr)
            
            st.write(f"Generation time: {deep_time:.2f} seconds")
            st.audio(deep_mix_path, format="audio/wav")
        except Exception as e:
            st.error(f"Error loading DeepModel: {str(e)}")
            st.write("Make sure the model file exists at models/demucs_finetuned.pt")
    
    with right_col:
        try:
            # Create spectrogram for DeepModel mix
            if deep_mix.ndim > 1:
                deep_mix_mono = deep_mix.mean(axis=1)
            else:
                deep_mix_mono = deep_mix
            
            st.pyplot(plot_spectrogram(deep_mix_mono, sr, "DeepModel Mix Spectrogram"))
        except:
            st.write("Spectrogram not available due to DeepModel error")
