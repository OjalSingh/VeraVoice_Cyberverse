import librosa
import numpy as np

def extract_mfcc_features(file_path, n_mfcc=13, max_pad_len=100):
    try:
        audio, sample_rate = librosa.load(file_path, res_type="kaiser_fast")  
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)

        # Pad or truncate to fixed length
        if mfccs.shape[1] < max_pad_len:
            pad_width = max_pad_len - mfccs.shape[1]
            mfccs = np.pad(mfccs, pad_width=((0,0),(0,pad_width)), mode="constant")
        else:
            mfccs = mfccs[:, :max_pad_len]

        return mfccs.flatten()  # make it 1D
    except Exception as e:
        print(f"Error extracting MFCCs from {file_path}: {e}")
        return np.zeros(n_mfcc * max_pad_len)