from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import tensorflow as tf
import pandas as pd
from imblearn.over_sampling import RandomOverSampler

# Data Scaling and Oversampling
def scale_dataset(dataframe, oversample=False):
    X = dataframe[dataframe.columns[:-1]].values
    y = dataframe[dataframe.columns[-1]].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    if oversample:
        ros = RandomOverSampler()
        X, y = ros.fit_resample(X, y)

    data = np.hstack((X, np.reshape(y, (-1, 1))))

    return data, X, y

# Neural Network Model
def NeuralNetworkClassifier(features, labels, prediction_set, epochs=10, batch_size=32):
    # Define the Neural Network
    nn_model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(features.shape[1],)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # Compile the Model
    nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the Model
    nn_model.fit(features, labels, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=1)

    # Predict the Classes for the Prediction Set
    predictions = (nn_model.predict(prediction_set) > 0.5).astype(int)

    return predictions
