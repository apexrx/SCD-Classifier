from FeatureExtraction import *
from Tools import *
from LabelledData import *
from Classify import *
import sys

# Accept in image path argument
if len(sys.argv) < 2:
    print("Error: Please include image path in argument")
    exit()
if len(sys.argv) > 2:
    print("Error: Only one argument allowed")
    exit()
imagePath = str(sys.argv[1])

# Prepare the image for analysis
plt.figure("Original Image")
try:
    plt.imshow(cv2.imread(imagePath))
except:
    print("Error: Image path is not valid")
    exit()
result, num_features = image_prep(imagePath)
plt.figure("Prepped Image")
plt.imshow(result)

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
    else:
        ax.scatter(relativeAreaArray[i], relativePerimArray[i], circularityArray[i])

# Display the sickle cell image
displaySickleImage(result, classified)

# Print statistics
num_features = len(classified)
numHealthyCells = num_features - numSickleCells
print("Total Cells:", num_features)
print("Sickle Cells:", numSickleCells)
print("Healthy Cells:", numHealthyCells)
print("Percent Sickle:", (numSickleCells / num_features) * 100, "%")
print("Percent Healthy:", (numHealthyCells / num_features) * 100, "%")

# Display all images
plt.show()
