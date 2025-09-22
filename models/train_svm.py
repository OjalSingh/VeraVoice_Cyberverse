# models/train_svm.py
import os
import joblib
import numpy as np
from sklearn.svm import SVC
from features import extract_mfcc_features
from glob import glob

# 1️⃣ Prepare dataset
real_files = glob("data/real/*.wav")
fake_files = glob("data/fake/*.wav")

X, y = [], []

for file in real_files:
    feat = extract_mfcc_features(file)
    X.append(feat)
    y.append(0)  # real label

for file in fake_files:
    feat = extract_mfcc_features(file)
    X.append(feat)
    y.append(1)  # fake label

X = np.array(X)
y = np.array(y)

# 2️⃣ Train SVM
clf = SVC(probability=True, kernel='linear')
clf.fit(X, y)

# 3️⃣ Save model
os.makedirs("models", exist_ok=True)
joblib.dump(clf, "models/svm_mfcc.pkl")
print("Model saved at models/svm_mfcc.pkl")



#NOT BEING USED
