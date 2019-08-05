# Md. Toufiqul Islam
# ID: 15-02-04-097
# Section: B2

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

trainDataset = pd.read_csv('train.txt', sep=" ", header=None, dtype='int64')

# ploting the train.txt data
class1X = []
class1Y = []

class2X = []
class2Y = []
for arr in trainDataset.values:
    if arr[2] == 1:
        class1X.append(arr[0])
        class1Y.append(arr[1])
    elif arr[2] == 2:
        class2X.append(arr[0])
        class2Y.append(arr[1])

plt.scatter(class1X, class1Y, marker="*", color='red')
plt.scatter(class2X, class2Y, marker="+", color='blue')

# plt.show()


# Mean Classifier
class1MeanMatrix = np.array([np.mean(class1X), np.mean(class1Y)])
class2MeanMatrix = np.array([np.mean(class2X), np.mean(class2Y)])

testDataset = pd.read_csv('test.txt', sep=" ", header=None, dtype="int64")

count = 0

testClass1X = []
testClass1Y = []

testClass2X = []
testClass2Y = []
for value in testDataset.values:
    x = np.array([value[0], value[1]])
    g1x = np.dot(np.transpose(class1MeanMatrix), x) - np.dot(np.transpose(class1MeanMatrix), class1MeanMatrix) / 2.0
    g2x = np.dot(np.transpose(class2MeanMatrix), x) - np.dot(np.transpose(class2MeanMatrix), class2MeanMatrix) / 2.0
    absg1x = (g1x)
    absg2x = (g2x)

    if absg1x < absg2x:
        if value[2] == 1:
            count += 1
        testClass1X.append(value[0])
        testClass1Y.append(value[1])
    else:
        if value[2] == 2:
            count += 1
        testClass2X.append(value[0])
        testClass2Y.append(value[1])

plt.scatter(testClass1X, testClass1Y, marker="o", color='red')
plt.scatter(testClass2X, testClass2Y, marker="v", color='blue')
# plt.show()

# Dicission Boundary
min = min(class1X.__add__(class2Y).__add__(class2X).__add__(class2Y))
xvalues = np.arange(min, 7, 0.25)

constant = (np.dot(np.transpose(class1MeanMatrix), class1MeanMatrix) - np.dot(np.transpose(class2MeanMatrix),
                                                                             class2MeanMatrix))/2.0
coeff1 = np.dot(np.transpose(class1MeanMatrix), np.array([1, 1]))
coeff2 = np.dot(np.transpose(class2MeanMatrix), np.array([1, 1]))

yValues = []
for x in xvalues:
    yValues.append((coeff1 * x + constant) / coeff2)

plt.scatter(xvalues, yValues, marker=".", color="k")
plt.show()

accuracy = count / len(testDataset.values) * 100
print("Accuracy Is: %.2f" % accuracy)
