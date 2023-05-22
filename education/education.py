import os
import numpy as np
from scipy.io import wavfile
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
import librosa

# Directory where the dataset is located
data_dir = "../dataset"

# Convert audio file to attributes
def extract_features(file_path):
    audio_data, sample_rate = librosa.load(file_path)
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
    mfccs_processed = np.mean(mfccs.T, axis=0)  # Gets the average MFCC values

    return mfccs_processed

# Loading and processing the dataset
actors = ['Actor_{}'.format(i) for i in range(1, 25)]
all_features = []
all_labels = []

for actor in actors:
    actor_dir = os.path.join(data_dir, actor)
    files = os.listdir(actor_dir)
    for file in files:
        file_path = os.path.join(actor_dir, file)
        features = extract_features(file_path)
        all_features.append(features)
        all_labels.append(actor)

# Convert tags to numeric values
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(all_labels)

# Creating training and test datasets
X_train, X_test, y_train, y_test = train_test_split(all_features, encoded_labels, test_size=0.2, random_state=42)

# Create and train the classifier model
classifier = SVC()
classifier.fit(X_train, y_train)

# Accuracy evaluation on the training dataset
train_predictions = classifier.predict(X_train)
train_accuracy = accuracy_score(y_train, train_predictions)
print("Training dataset accuracy:", train_accuracy)

# Accuracy evaluation on the test dataset
test_predictions = classifier.predict(X_test)
test_accuracy = accuracy_score(y_test, test_predictions)
print("Test dataset accuracy:", test_accuracy)

import h5py
save_dir = "../model"
save_path = os.path.join(save_dir, 'dataset.h5')
joblib.dump(classifier, save_path)

data = {'features': all_features, 'labels': all_labels}

with h5py.File(save_path, 'w') as f:
    f.create_dataset('features', data=data['features'])
    f.create_dataset('labels', data=data['labels'])