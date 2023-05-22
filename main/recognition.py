import numpy as np
import librosa

def extract_features(file_path):
    audio_data, sample_rate = librosa.load(file_path)
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
    mfccs_processed = np.mean(mfccs.T, axis=0)  # Gets the average MFCC values

    return mfccs_processed
