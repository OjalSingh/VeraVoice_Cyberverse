
# features.py
import librosa
import numpy as np

def extract_mfcc_features(input_audio, sr=None, n_mfcc=13):
    """
    Extracts MFCC features from audio.

    Parameters:
    - input_audio: str (file path) OR numpy array (audio waveform)
    - sr: sampling rate (required if input_audio is array)
    - n_mfcc: number of MFCC coefficients

    Returns:
    - numpy array of mean MFCC features (size n_mfcc)
    """
    # Load audio if input is a file path
    if isinstance(input_audio, str):
        y, sr = librosa.load(input_audio, sr=None)
    else:
        y = input_audio
        if sr is None:
            raise ValueError("Sampling rate must be provided for numpy array input.")

    # Compute MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    # Average over time frames to get fixed-size feature vector
    mfcc_mean = np.mean(mfcc.T, axis=0)
    return mfcc_mean

