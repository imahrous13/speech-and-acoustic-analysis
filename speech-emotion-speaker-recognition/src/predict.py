"""
Prediction module for Speech Emotion Detection.

This module provides functions to predict emotions from audio files using
a trained machine learning model.
"""

import librosa
import numpy as np
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# FEATURE EXTRACTION FUNCTIONS
# ============================================================================

def extract_mfcc(y, sr, n_mfcc=40):
    """Extract MFCC (Mel-Frequency Cepstral Coefficients) features."""
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfccs.T, axis=0)


def extract_chroma(y, sr):
    """Extract Chroma features (pitch class profile)."""
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    return np.mean(chroma.T, axis=0)


def extract_mel_spectrogram(y, sr, n_mels=128):
    """Extract Mel Spectrogram and flatten it."""
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_mean = np.mean(mel_spec.T, axis=0)
    return mel_mean


def extract_spectral_centroid(y, sr):
    """Extract Spectral Centroid (brightness of sound)."""
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    return np.mean(centroid)


def extract_spectral_bandwidth(y, sr):
    """Extract Spectral Bandwidth (spread of spectrum)."""
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    return np.mean(bandwidth)


def extract_spectral_contrast(y, sr):
    """Extract Spectral Contrast features."""
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    return np.mean(contrast.T, axis=0)


def extract_zero_crossing_rate(y):
    """Extract Zero-Crossing Rate (ZCR)."""
    zcr = librosa.feature.zero_crossing_rate(y)
    return np.mean(zcr)


def extract_rms_energy(y):
    """Extract RMS (Root Mean Square) Energy."""
    rms = librosa.feature.rms(y=y)
    return np.mean(rms)


def extract_all_features(audio_path, sr=22050):
    """Extract all features from an audio file."""
    try:
        # Load audio file
        y, sample_rate = librosa.load(audio_path, sr=sr)
        
        # Extract all features
        features = {}
        
        # MFCC (40 coefficients)
        mfcc = extract_mfcc(y, sample_rate, n_mfcc=40)
        for i, val in enumerate(mfcc, 1):
            features[f'mfcc_{i}'] = val
        
        # Chroma Features (12 coefficients)
        chroma = extract_chroma(y, sample_rate)
        for i, val in enumerate(chroma, 1):
            features[f'chroma_{i}'] = val
        
        # Mel Spectrogram (flattened, 128 features)
        mel_spec = extract_mel_spectrogram(y, sample_rate, n_mels=128)
        for i, val in enumerate(mel_spec, 1):
            features[f'mel_spec_{i}'] = val
        
        # Spectral features
        features['spectral_centroid'] = extract_spectral_centroid(y, sample_rate)
        features['spectral_bandwidth'] = extract_spectral_bandwidth(y, sample_rate)
        
        # Spectral Contrast (7 coefficients)
        contrast = extract_spectral_contrast(y, sample_rate)
        for i, val in enumerate(contrast, 1):
            features[f'spectral_contrast_{i}'] = val
        
        # Zero-Crossing Rate and RMS Energy
        features['zcr'] = extract_zero_crossing_rate(y)
        features['rms'] = extract_rms_energy(y)
        
        return features
    
    except Exception as e:
        print(f"Error processing {audio_path}: {str(e)}")
        return None


# ============================================================================
# MODEL LOADING
# ============================================================================

# Global variables to cache loaded models
_loaded_model = None
_loaded_label_encoder = None
_loaded_scaler = None
_loaded_feature_names = None
_models_dir = None


def _get_models_dir():
    """Get the models directory path."""
    global _models_dir
    if _models_dir is None:
        # Try to find models directory relative to this file
        current_file = Path(__file__).resolve()
        # models/ should be next to the project root. If this file is inside
        # a `src/` package, go up one level; otherwise use the parent folder.
        # Prefer project_root/models
        possible = current_file.parent
        if possible.name == 'src':
            possible = possible.parent
        _models_dir = possible / 'models'
    return _models_dir


def _load_model_artifacts():
    """Load model artifacts (model, scaler, label_encoder, feature_names)."""
    global _loaded_model, _loaded_label_encoder, _loaded_scaler, _loaded_feature_names
    
    if _loaded_model is None:
        models_dir = _get_models_dir()
        
        # Load model
        with open(models_dir / 'saved_model.pkl', 'rb') as f:
            _loaded_model = pickle.load(f)
        
        # Load label encoder
        with open(models_dir / 'label_encoder.pkl', 'rb') as f:
            _loaded_label_encoder = pickle.load(f)
        
        # Load scaler
        with open(models_dir / 'scaler.pkl', 'rb') as f:
            _loaded_scaler = pickle.load(f)
        
        # Load feature names
        with open(models_dir / 'feature_names.pkl', 'rb') as f:
            _loaded_feature_names = pickle.load(f)
    
    return _loaded_model, _loaded_label_encoder, _loaded_scaler, _loaded_feature_names


# ============================================================================
# PREDICTION FUNCTION
# ============================================================================

def predict_emotion(audio_path):
    """
    Predict emotion from an audio file.
    
    This function automatically loads the trained model, scaler, label encoder,
    and feature names from the models directory.
    
    Args:
        audio_path: Path to the audio file (str or Path object)
    
    Returns:
        dict: Dictionary containing:
            - 'audio_path': Path to the audio file
            - 'predicted_emotion': Predicted emotion label
            - 'probabilities': Dictionary mapping emotions to probabilities
        None if prediction fails
    """
    # Load model artifacts
    model, label_encoder, scaler, feature_names = _load_model_artifacts()
    
    # Extract features
    features = extract_all_features(audio_path)
    
    if features is None:
        return None
    
    # Arrange features in correct order
    feature_vector = np.array([features.get(feat, 0.0) for feat in feature_names])
    
    # Handle missing values
    if np.isnan(feature_vector).any():
        feature_vector = np.nan_to_num(feature_vector, nan=0.0)
    
    # Reshape for single sample
    feature_vector = feature_vector.reshape(1, -1)
    
    # Scale features
    feature_vector_scaled = scaler.transform(feature_vector)
    
    # Predict
    prediction_encoded = model.predict(feature_vector_scaled)[0]
    prediction = label_encoder.inverse_transform([prediction_encoded])[0]
    
    # Get prediction probabilities
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(feature_vector_scaled)[0]
        prob_dict = {
            label_encoder.inverse_transform([i])[0]: prob 
            for i, prob in enumerate(probabilities)
        }
    else:
        prob_dict = None
    
    return {
        'audio_path': str(audio_path),
        'predicted_emotion': prediction,
        'probabilities': prob_dict
    }


def predict_speaker(audio_path):
    """
    Predict speaker identity from an audio file.

    Expects these files in `models/`:
      - `speaker_model.pkl`
      - `speaker_label_encoder.pkl`
      - `speaker_scaler.pkl`
      - `speaker_feature_names.pkl`

    Raises FileNotFoundError with instructions when artifacts are missing.
    """
    models_dir = _get_models_dir()

    model_path = models_dir / 'speaker_model.pkl'
    le_path = models_dir / 'speaker_label_encoder.pkl'
    scaler_path = models_dir / 'speaker_scaler.pkl'
    feat_path = models_dir / 'speaker_feature_names.pkl'

    missing = [p.name for p in (model_path, le_path, scaler_path, feat_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing speaker model artifacts in {models_dir}: {missing}.\n"
            "Place `speaker_model.pkl`, `speaker_label_encoder.pkl`, `speaker_scaler.pkl`, "
            "and `speaker_feature_names.pkl` in the `models/` directory to enable speaker identification."
        )

    with open(model_path, 'rb') as f:
        speaker_model = pickle.load(f)
    with open(le_path, 'rb') as f:
        speaker_le = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        speaker_scaler = pickle.load(f)
    with open(feat_path, 'rb') as f:
        speaker_feature_names = pickle.load(f)

    features = extract_all_features(audio_path)
    if features is None:
        return None

    feature_vector = np.array([features.get(feat, 0.0) for feat in speaker_feature_names])
    if np.isnan(feature_vector).any():
        feature_vector = np.nan_to_num(feature_vector, nan=0.0)

    feature_vector = feature_vector.reshape(1, -1)
    feature_vector_scaled = speaker_scaler.transform(feature_vector)

    pred_encoded = speaker_model.predict(feature_vector_scaled)[0]
    pred = speaker_le.inverse_transform([pred_encoded])[0]

    prob_dict = None
    if hasattr(speaker_model, 'predict_proba'):
        probs = speaker_model.predict_proba(feature_vector_scaled)[0]
        prob_dict = {speaker_le.inverse_transform([i])[0]: p for i, p in enumerate(probs)}

    return {
        'audio_path': str(audio_path),
        'predicted_speaker': pred,
        'probabilities': prob_dict
    }


