import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from FeatureExtraction import *
from Tools import *
from LabelledData import *
from Classify import *
import os


def main():
    st.title("Sickle Cell Disease Classifier")

    # File uploader for image
    uploaded_file = st.file_uploader("Upload RBC Sample Image", type=["jpg", "png"])

    if uploaded_file is not None:
        # Save the uploaded file to a temporary location
        with open("temp_image.jpg", "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Read the uploaded image
        uploaded_image = plt.imread("temp_image.jpg")
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

        # Prepare the image for analysis
        result, num_features = image_prep("temp_image.jpg")

        # Display the prepped image
        st.image(result, caption="Prepped Image", use_column_width=True)

        # Extract the area and perimeter for each cell
        areaArray, perimArray = extract_area_perim(result, num_features)

        # Remove empty first elements
        areaArray.pop(0)
        perimArray.pop(0)

        # Get the circularity for each cell
        circularityArray = extract_circularity(areaArray, perimArray)

        # Get the relative area and perimeter of each cell
        relativeAreaArray, relativePerimArray = convert_to_relative(areaArray, perimArray)

        # Get preloaded data
        sickleData = getSickleData()
        healthyData = getHealthyData()

        # Combine the two training data arrays into one array
        combinedHealthySickle = sickleData + healthyData

        labels = [1] * len(sickleData) + [0] * len(healthyData)

        # Convert training data to numpy arrays
        features_array = np.array(combinedHealthySickle)
        labels_array = np.array(labels)

        # Extract features from the image
        features = convertTo3D(relativeAreaArray, relativePerimArray, circularityArray)

        # Classify each cell using Random Forest
        classified = RandomForestClassifier(features_array, labels_array, features, n_estimators=100)

        # Plot the classified cells
        fig = plt.figure("Classified Graph")
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('Area')
        ax.set_ylabel('Perimeter')
        ax.set_zlabel('Circularity')

        # Count the number of sickle cells & plot them in red
        numSickleCells = np.sum(classified)
        for i in range(len(classified)):
            if classified[i] == 1:
                ax.scatter(relativeAreaArray[i], relativePerimArray[i], circularityArray[i], c="red")

        # Display sickle cell image
        sickle_image = displaySickleImage(result, classified)
        st.image(sickle_image, caption="Sickle Cells Image", use_column_width=True)

        # Display the classified cells image
        st.pyplot(fig)
        st.caption("Feature visualization of Sickle Cells")

        # Display classification results
        st.subheader("Classification Results:")
        st.write("Sickle Cells:", np.sum(classified))

        # Remove the temporary file
        os.remove("temp_image.jpg")


if __name__ == "__main__":
    main()


