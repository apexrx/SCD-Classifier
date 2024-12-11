import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from FeatureExtraction import *
from Tools import *
from LabelledData import *
from Classify import NeuralNetworkClassifier, scale_dataset
import os

# Set Streamlit theme to light with red accents
st.set_page_config(page_title="Sickle Cell Disease Classifier", page_icon=":microscope:", layout="wide", initial_sidebar_state="expanded")

# Define colors
background_color = "#FFFFFF"  # White background
text_color = "#000000"  # Black text color
accent_color = "#FF0000"  # Red accent color

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

        # Prepare the image for analysis
        result, num_features = image_prep("temp_image.jpg")

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

        # Convert training data to a pandas DataFrame for scaling
        cols = ["area", "perimeter", "circularity", "class"]
        data = [list(data) + [label] for data, label in zip(combinedHealthySickle, labels)]
        df = pd.DataFrame(data, columns=cols)

        # Scale and preprocess training data
        _, X_train, y_train = scale_dataset(df, oversample=True)

        # Extract features from the image
        features = convertTo3D(relativeAreaArray, relativePerimArray, circularityArray)
        features = np.array(features)  # Ensure features are numpy arrays

        # Classify each cell using Neural Network
        classified = NeuralNetworkClassifier(X_train, y_train, features, epochs=10, batch_size=32)

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
                ax.scatter(relativeAreaArray[i], relativePerimArray[i], circularityArray[i], c=accent_color)

        # Display sickle cell image
        sickle_image = displaySickleImage(result, classified)

        # Create two columns for image display
        col1, col2, col3 = st.columns([1, 0.5, 1])

        # Display uploaded image
        with col1:
            st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

        # Display arrow pointing towards prepped image
        with col2:
            st.markdown('<p style="text-align: center; font-size: 24px; margin-top: 50%;">&#8594;</p>', unsafe_allow_html=True)

        # Display prepped image
        with col3:
            st.image(result, caption="Prepped Image", use_column_width=True)

        # Add padding between rows
        st.write("")  # Empty space
        st.write("---")  # Divider

        # Display classification results
        st.subheader("Classification Results:")
        st.write("Sickle Cells:", numSickleCells)

        # Display feature visualization plot and sickle image in the same row
        col4, col5 = st.columns([1, 1])

        # Display feature visualization plot
        with col4:
            st.subheader("Feature Visualization")
            st.pyplot(fig)
            st.caption("Feature visualization of Sickle Cells")

        # Display sickle cell image
        with col5:
            st.subheader("Sickle Cells Image")
            st.image(sickle_image, caption="Sickle Cells Image", use_column_width=True)

        # Remove the temporary file
        os.remove("temp_image.jpg")

if __name__ == "__main__":
    main()
