import math
import sys
import numpy as np
import scipy
from matplotlib import pyplot as plt


def load_data(filepath):
    # Takes in a string with a path to a CSV file and returns the data points as a list of dicts
    listToReturn = []
    with open(filepath) as f:
        # For each line in filepath: save it as a string, split up its elements by the , deliminator
        for line in f:
            string = line
            stringList = string.split(",")
            # Skip the first line in the list
            if (stringList[1] == 'Country'):
                continue
            # Edit the final index in the list so it doesn't have \n
            stringList[7] = stringList[7].replace("\n", "")
            # Create a dictionary object with the column headers as keys and the row elements as values
            dict = {'': stringList[0], 'Country': stringList[1], 'Population': stringList[2],
                    'Net migration': stringList[3], 'GDP ($ per capita)': stringList[4],
                    'Literacy (%)': stringList[5], 'Phones (per 1000)': stringList[6],
                    'Infant mortality (per 1000 births)': stringList[7]}
            # Add this dictionary to the list to return. Continue to next line
            listToReturn.append(dict)

    return listToReturn


def calc_features(row):
    # Calculates the Feature vector for the country, returns it as a NumPy array of shape (6,). Dtype should be
    # float64
    # Population, Net Migration, GDP, Literacy, Phones, Infant Mortality
    population = float(row['Population'])
    netMigration = float(row['Net migration'])
    gdp = float(row['GDP ($ per capita)'])
    literacy = float(row['Literacy (%)'])
    phones = float(row['Phones (per 1000)'])
    infant = float(row['Infant mortality (per 1000 births)'])
    returnList = np.array([population, netMigration, gdp, literacy, phones, infant], dtype="float64")
    return returnList


def hac(features):
    # Obtain the size from the feature vector
    size = len(features)
    # Initialize and create the distance matrix
    distanceMatrix = np.zeros((size, size))
    for i in range(len(features)):
        for j in range(len(features)):
            # TA said this was correct
            distance = np.linalg.norm(features[i] - features[j])
            distanceMatrix[i][j] = distance

    # 1. Number each of your starting points from 0 to n-1 with a dict
    dict = {
    }
    for i in range(len(features)):
        dict[i] = [i]

    # Create a (n-1) x 4 array or list (which will be returned)
    returnList = np.zeros(((size - 1), 4))
    # Iterate through each row of the returnList
    for l in range(len(returnList)):
        # Each row will have its own maximum value dictionary (check that this resets it), and key index for it
        key = -1
        maxValDict = {}
        # Next: go through each cluster (dict[0]...,etc) find the maximum distance from it to every other cluster
        # i is in range size + l because that is the range of dictionary cluster indices. Must do this for two clusters
        for i in range(size + l):
            for w in range(size + l):
                # We can't compare a cluster to itself
                if i == w:
                    continue
                # For the ith cluster, find all clusters that are a maximum distance from it
                # If i is a key in dictionary, And w is, then compare the two clusters and find the maximum distance
                # between them
                if i in dict.keys():
                    if w in dict.keys():
                        # Initialize the maxVal and indexes, increment key
                        maxVal = {
                            0: math.ulp(0.0),
                            1: [],  # 1 contains the list of og indices. 2/3 will contain the cluster indices
                        }
                        kIndex = [-1]
                        lIndex = [-1]
                        key = key + 1
                        for j in range(len(dict[i])):
                            for u in range(len(dict[w])):
                                # Go through each index stored at i and each index stored at w (all possible combos)
                                index = dict[i][j]
                                index2 = dict[w][u]
                                val = distanceMatrix[index][index2]
                                if (val > maxVal[0]):
                                    maxVal[0] = distanceMatrix[index][index2]
                                    # update the values for maxVal: distance, cluster indices. Use kIndex and lIndex
                                    # later for updating maxVal[1]
                                    kIndex = dict[i]
                                    lIndex = dict[w]
                                    maxVal[2] = i
                                    maxVal[3] = w
                        # list1 contains all the og indices that are within this merger
                        list1 = [lIndex + kIndex]
                        maxVal[1] = list1
                        maxValDict[key] = maxVal

        # Next: Find the minimum value within this maximum distance list
        minVal = {
            0: sys.float_info.max,  # 1 contains the list of og indices, 2 and 3 contain the cluster indices
        }
        for m in range(key):
            if m in maxValDict.keys():
                if maxValDict[m][0] < minVal[0]:
                    minVal[0] = maxValDict[m][0]
                    # Complete list of og indices
                    minVal[1] = maxValDict[m][1][0]
                    # Contains the dict indices that we merged (the cluster numbers)
                    minVal[2] = maxValDict[m][2]
                    minVal[3] = maxValDict[m][3]
                # Implement the tiebreaker
                if (maxValDict[m][0] == minVal[0]):
                    # If the index stored at 2 is less than the index stored at 3, it is the first index to check
                    if (minVal[2] < minVal[3]):
                        minIndex1 = minVal[2]
                        minIndex2 = minVal[3]
                    # Otherwise, it is the second index to check
                    else:
                        minIndex1 = minVal[3]
                        minIndex2 = minVal[2]
                    # If the index stored at 2 is less than the index stored at 3, it is the first index to check, vv
                    if (maxValDict[m][2] < maxValDict[m][3]):
                        maxValIndex1 = maxValDict[m][2]
                        maxValIndex2 = maxValDict[m][3]
                    else:
                        maxValIndex1 = maxValDict[m][3]
                        maxValIndex2 = maxValDict[m][2]
                    # First, check if the index of the maxValDict we are iterating thorugh is less than minIndex 1.
                    # If so, update the minValue
                    if (maxValIndex1 < minIndex1):
                        minVal[0] = maxValDict[m][0]
                        minVal[1] = maxValDict[m][1][0]
                        # Contains the dict indecies that we merged (the cluster numbers)
                        minVal[2] = maxValDict[m][2]
                        minVal[3] = maxValDict[m][3]
                    # If they are equal, check the same for the second indices
                    if (maxValIndex1 == minIndex1):
                        if (maxValIndex2 < minIndex2):
                            minVal[0] = maxValDict[m][0]
                            minVal[1] = maxValDict[m][1][0]
                            # Contains the dict indecies that we merged (the cluster numbers)
                            minVal[2] = maxValDict[m][2]
                            minVal[3] = maxValDict[m][3]
        # Next: add to the return list, and update dictionary by removing the previous values and adding the new cluster
        dict[size + l] = minVal[1]
        lenCluster = len(minVal[1])
        # ensure it is in increasing order (the cluster indices that are being merged)
        if (minVal[2] < minVal[3]):
            returnList[l] = [minVal[2], minVal[3], minVal[0], lenCluster]
        else:
            returnList[l] = [minVal[3], minVal[2], minVal[0], lenCluster]
        # Find the dict cluster index (i) that contains any of the indices being merged. If it does, remove it
        for i in range(size + l):
            for j in range(len(minVal[1])):
                if i in dict.keys():
                    if minVal[1][j] in dict[i]:
                        del dict[i]
    return returnList


def fig_hac(Z, names):
    # Input: NumPy array Z output from hac, list of string names corresponding to country name with length n
    # Output: A matplot lib figure with a graph that visualizes the hierarchical clustering
    fig = plt.figure()
    gram = scipy.cluster.hierarchy.dendrogram(Z, labels=names, leaf_rotation=90)
    fig.tight_layout()
    return fig


def normalize_features(features):
    # Output: Identical format to input. A list of NumPy arrays with shape (6,) and dtype float64
    # size = number of rows
    size = len(features)
    meanList = np.zeros(6)
    sdList = np.zeros(6)
    # Iterate through each column...each column has their own array
    for i in range(len(features[0])):
        array = np.zeros(size)
        # Iterate through each row of the column (column stays constant)
        for j in range(len(features)):
            # Store an array containing all column values
            array[j] = features[j][i]
        # Calculate the eman and standard deviation of that array, store the value in each list
        meanList[i] = np.mean(array)
        sdList[i] = np.std(array)

    returnList = []
    # Iterate through the rows, each row having their own array (the row now stays constant)
    for j in range(len(features)):
        array = np.zeros(6)
        # Iterate through each value stored at the lists. Each column corresponds to index of each list
        for i in range(len(meanList)):
            ogVal = features[j][i]
            mean = meanList[i]
            sd = sdList[i]
            finalVal = (ogVal - mean) / sd
            array[i] = finalVal

        returnList.append(array)
    return returnList

# Main method can be uncommented out to test project for various choices of n
'''''
if __name__=="__main__":
    data = load_data("countries.csv")
    country_names = [row["Country"] for row in data]
    features = [calc_features(row) for row in data]
    features_normalized = normalize_features(features)
    n = 50 # Vary n for testing
    Z_raw = hac(features[:n])
    fig = fig_hac(Z_raw, country_names[:n])
    plt.show()
'''

