# utils.py
import librosa
import numpy as np

def extract_features_from_audio(file_path):
    try:
        # Load audio file using librosa
        y, sr = librosa.load(file_path, duration=30)  # limit to first 30 seconds if too long
        
        # Extract MFCCs
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        
        # Flatten the MFCC array
        mfcc = np.mean(mfcc.T, axis=0)
        
        return mfcc
    except Exception as e:
        print(f"Error during feature extraction: {e}")
        return None
