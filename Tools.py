
import matplotlib.pyplot as plt
import numpy as np

def printArray(message ,array):
    print(message, end= " ")
    for i in range(len(array)):
        print(i, array[i], end = ", ")
    print()
def highlightCell(img, number):
    result = img
    for i in range(len(img)):
        for j in range(len(img[i])):
            if (img[i][j] == number):
                result[i][j] = 240
    return result

def displayAllImages(img, num_features):
    for i in range(1, num_features + 1):
        findCell = highlightCell(img, i)
        plt.imshow(findCell)
        plt.title(i - 1)
        plt.show()
def displaySickleImage(img, array):
    result = np.copy(img)
    for z in range(len(array)):
        if (array[z] == 1):
            result = highlightCell(result, z + 1)
    return result
def convertTo3D(areaArray, perimArray, circularityArray):
    result = []
    for i in range(len(areaArray)):
        result.append([areaArray[i],perimArray[i],circularityArray[i]])
    return result