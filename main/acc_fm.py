import h5py
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

# Loading the dataset
with h5py.File('../model/dataset.h5', 'r') as f:
    all_features = np.array(f['features'])
    all_labels = np.array(f['labels'])

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

# Calculating F1 score on the training dataset
train_f1_score = f1_score(y_train, train_predictions, average='weighted')

# Accuracy evaluation on the test dataset
test_predictions = classifier.predict(X_test)
test_accuracy = accuracy_score(y_test, test_predictions)

# Calculating F1 score on test dataset
test_f1_score = f1_score(y_test, test_predictions, average='weighted')
