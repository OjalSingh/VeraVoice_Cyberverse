from datasets import load_dataset
import os
import numpy as np
from features import extract_mfcc_features
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.model_selection import train_test_split

# Load Hugging Face real dataset
ds_real = load_dataset("nguyenvulebinh/libris_clean_100", split="train")

X, y = [], []

# Real voices = label 0
for sample in ds_real:
    audio_array = sample["audio"]["array"]
    sr = sample["audio"]["sampling_rate"]
    feat = extract_mfcc_features(audio_array, sr)
    X.append(feat)
    y.append(0)

# Fake voices = label 1
FAKE_PATH = "data/fake"
for f in os.listdir(FAKE_PATH):
    fp = os.path.join(FAKE_PATH, f)
    import librosa
    y_audio, sr = librosa.load(fp, sr=None)
    feat = extract_mfcc_features(y_audio, sr)
    X.append(feat)
    y.append(1)

# Convert to arrays
X = np.array(X)
y = np.array(y)

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train SVM
clf = SVC(probability=True, kernel="linear")
clf.fit(X_train, y_train)

print("Training done. Test accuracy:", clf.score(X_test, y_test))

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(clf, "models/svm_mfcc.pkl")
print("Model saved at models/svm_mfcc.pkl")