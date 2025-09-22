# test_features.py
from features import extract_mfcc_features

# Test MFCC extraction on a sample file
feat = extract_mfcc_features("data/real/sample.wav")
print(feat.shape)  # should print (13,) if using n_mfcc=13
print(feat)        # optional: see the actual MFCC values