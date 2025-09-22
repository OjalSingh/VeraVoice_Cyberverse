import librosa
import numpy as np

def extract_mfcc_features(file_path, n_mfcc=13):
    try:
        audio, sample_rate = librosa.load(file_path, res_type="kaiser_fast")  
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
        
        # Take mean across time axis to get 1 value per coefficient
        mfccs_mean = np.mean(mfccs, axis=1)  # shape (13,)
        
        return mfccs_mean
    except Exception as e:
        print(f"Error extracting MFCCs from {file_path}: {e}")
        return np.zeros(n_mfcc)