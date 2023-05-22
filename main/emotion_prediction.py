import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow import keras

# Path to the saved model file
model_file_path = "../model/emotion_prediction_model.h5"

# Audio file sampling rate
sampling_rate = 48000

# Emotion labels
emotions = ['Angry', 'Anxious', 'Beware', 'Bored', 'Cheerful', 'Frightened', 'Furious', 'Happy', 'Interrogator', 'Neutral', 'Peaceful', 'Reckless', 'Sad', 'Sarcastic', 'Shocked']

def extract_features(audio, sr):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    return np.concatenate((mfcc_mean, mfcc_std))

def predict_emotion(audio_file_path):
    # Load the audio file
    audio, sr = librosa.load(audio_file_path, sr=sampling_rate)
    
    # Extract features
    feature = extract_features(audio, sr)
    
    # Load the model
    model = keras.models.load_model(model_file_path)
    
    # Perform emotion prediction
    predictions = model.predict(np.expand_dims(feature, axis=0))
    predicted_index = np.argmax(predictions)
    predicted_emotion = emotions[predicted_index]
    
    return predicted_emotion

