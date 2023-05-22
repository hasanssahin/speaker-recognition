import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import librosa
import tensorflow as tf
from tensorflow import keras

# Data directory
data_directory = "../dataset2"

# Audio file sampling rate
sampling_rate = 48000

# Emotion labels
emotions = ['Angry', 'Anxious', 'Beware', 'Bored', 'Cheerful', 'Frightened', 'Furious', 'Happy', 'Interrogator', 'Neutral', 'Peaceful', 'Reckless', 'Sad', 'Sarcastic', 'Shocked']

# Lists to store features and labels
features = []
labels = []

def extract_features(audio, sr):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    return np.concatenate((mfcc_mean, mfcc_std))

# Iterate through the dataset
for emotion in emotions:
    for actor in range(1, 25):
        folder_path = os.path.join(data_directory, emotion, "Actor_" + str(actor))
        for file in os.listdir(folder_path):
            if file.endswith(".wav"):
                file_path = os.path.join(folder_path, file)
                audio, sr = librosa.load(file_path, sr=sampling_rate)
                feature = extract_features(audio, sr)  # Extract features from audio file
                features.append(feature)
                labels.append(emotion)

# Convert labels to numerical values
le = LabelEncoder()
labels = le.fit_transform(labels)

# Split the dataset into training and test subsets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Create the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate training accuracy
y_train_pred = model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
print("Training accuracy:", train_accuracy)

# Evaluate test accuracy
y_test_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print("Test accuracy:", test_accuracy)

# Save the test data
np.save("../model/test_data.npy", X_test)
np.save("../model/test_tags.npy", y_test)

# Convert to TensorFlow Keras model
tf_model = keras.models.Sequential()
tf_model.add(keras.layers.Dense(units=64, activation='relu', input_dim=len(features[0])))
tf_model.add(keras.layers.Dense(units=len(emotions), activation='softmax'))
tf_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
tf_model.fit(np.array(X_train), np.array(y_train), epochs=10, batch_size=32)

# Save the model
model_directory = "../model"
model_name = "emotion_prediction_model.h5"
model_path = os.path.join(model_directory, model_name)

if not os.path.exists(model_directory):
    os.makedirs(model_directory)

tf_model.save(model_path)
