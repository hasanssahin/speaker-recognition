from flask import Flask, request, jsonify
import speech_recognition as sr
from histogram import calculate_histogram, calculate_mel_spectrogram, plot_audio_signal
from recognition import extract_features
import h5py
import pickle
import numpy as np
import acc_fm as i3
import emotion_prediction as ep
import os
import base64

app = Flask(__name__)

@app.route('/api/histogram', methods=['POST'])
def calculate_histogram_api():
    
    if 'audio_file' not in request.files:
        return 'No audio file provided', 400
    audio_file = request.files['audio_file']
    # Save audio file to a temporary location
    temp_file_path = '../tmp/'+audio_file.filename
    audio_file.save(temp_file_path)

    # Calculate histogram
    histogram_path = calculate_histogram(temp_file_path)
    spectrogram_path=calculate_mel_spectrogram(temp_file_path)
    signal_path=plot_audio_signal(temp_file_path)
    

    # Encode images to Base64
    histogram_base64 = encode_image_to_base64(histogram_path)
    spectrogram_base64 = encode_image_to_base64(spectrogram_path)
    signal_base64 = encode_image_to_base64(signal_path)

    # Delete temporary files
    os.remove(temp_file_path)
    os.remove(histogram_path)
    os.remove(spectrogram_path)
    os.remove(signal_path)

    # Return images as JSON
    return jsonify({
        'histogram': histogram_base64,
        'spectrogram': spectrogram_base64,
        'signal': signal_base64
    })


def encode_image_to_base64(image_path):
    with open(image_path, 'rb') as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string


    
@app.route('/api/recognition', methods=['POST'])
def predict_speaker_api():
    
    if 'audio_file' not in request.files:
        return 'No audio file provided', 400
    audio_file = request.files['audio_file']
    # Save audio file to a temporary location
    temp_file_path = '../tmp/'+audio_file.filename
    audio_file.save(temp_file_path)
    
    
    # Extract the attributes by uploading the audio recording
    test_file_path = temp_file_path  # Path to the audio file to be tested
    test_features = extract_features(test_file_path)
    # Loading the model# Modeli yükleme
    model_file = "../model/dataset.h5"  # Path to sound model file

    # Making predictions using the sound model
    if model_file.endswith('.h5'):
        with h5py.File(model_file, 'r') as f:
            model_features = f['features'][:]
            model_labels = f['labels'][:]
    else: # If saved as .pkl file
        with open(model_file, 'rb') as f:
            model_data = pickle.load(f)
            model_features = model_data['features']
            model_labels = model_data['labels']
            
    # Guessing
    distances = np.linalg.norm(model_features - test_features, axis=1)
    closest_index = np.argmin(distances)
    predicted_label = model_labels[closest_index]
    predicted_label = predicted_label.decode('utf-8')
    
    # Delete temporary files
    os.remove(temp_file_path)
    
    return jsonify({'speaker': predicted_label})



@app.route('/api/accfm', methods=['GET'])
def acc_fm_api():
    return jsonify({'education_acc':i3.train_accuracy,
        'education_fm':i3.train_f1_score,
        'test_acc':i3.test_accuracy,
        'test_fm':i3.test_f1_score})

r = sr.Recognizer()
@app.route('/api/transcription', methods=['POST'])
def transcribe():
    if 'audio_file' not in request.files:
        return 'No audio file provided', 400
    audio_file = request.files['audio_file']
    # Save audio file to a temporary location
    temp_file_path = '../tmp/'+audio_file.filename
    audio_file.save(temp_file_path)
    
    with sr.AudioFile(temp_file_path) as source:
         audio = r.record(source)  # Save audio file
     
    # Convert speech to text
    try:
        text = r.recognize_google(audio, language="en-EN")
        # Calculate word count
        word_count = len(text.split())
        # Return response in JSON format
        response = {
        "transcription": text,
        "word_count": word_count
        }
        # Delete temporary files
        os.remove(temp_file_path)
        return jsonify(response)
    
    except sr.UnknownValueError:
        return jsonify({"error": "Audio not understood"})
    except sr.RequestError:
        return jsonify({"error": "Query failed"})
    

@app.route('/api/predict-emotion', methods=['POST'])
def predict_emotion_endpoint():
    if 'audio_file' not in request.files:
        return 'No audio file provided', 400
    audio_file = request.files['audio_file']
    # Save audio file to a temporary location
    temp_file_path = '../tmp/'+audio_file.filename
    audio_file.save(temp_file_path)
    
    
    # Perform emotion prediction
    prediction = ep.predict_emotion(temp_file_path)
    return jsonify({"prediction":prediction})
    
if __name__ == '__main__':
    app.run()
