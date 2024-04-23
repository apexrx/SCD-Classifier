from sklearn.ensemble import RandomForestClassifier as RFClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import math

def findShortestDistance(array, outputSize):
    result = [0] * outputSize
    for i in range(len(array)):
        curr = array[i]
        inserted = False
        j = 0
        while(j < len(result) and not inserted):
            if(curr < array[result[j]]):
                result.insert(j,i)
                if len(result) >= outputSize:
                    result.pop(outputSize)
                inserted = True
            j+=1
    return result

def RandomForestClassifier(features, labels, predictionSet, n_estimators=100):
    # Initialize the Random Forest Classifier
    rf_classifier = RFClassifier(n_estimators=n_estimators)

    # Train the Random Forest model
    rf_classifier.fit(features, labels)

    # Predict the classes for the prediction set
    predictions = rf_classifier.predict(predictionSet)

    return predictions
