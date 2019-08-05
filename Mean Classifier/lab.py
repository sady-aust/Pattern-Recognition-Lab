import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

trainDataset = pd.read_csv('train.txt', sep=" ", header=None, dtype="int64")
trainClass1X = []
trainClass1Y = []

trainClass2X = []
trainClass2Y = []

for data in trainDataset.values:
    if data[2] == 1:
        trainClass1X.append(data[0])
        trainClass1Y.append(data[1])
    elif data[2] == 2:
        trainClass2X.append(data[0])
        trainClass2Y.append(data[1])

plt.scatter(trainClass1X, trainClass1Y, marker="*", color="red")
plt.scatter(trainClass2X, trainClass2Y, marker="+", color="blue")
# plt.show()
class1MeanMatrix = np.array([np.mean(trainClass1X), np.mean(trainClass1Y)])
class2MeanMatrix = np.array([np.mean(trainClass2X), np.mean(trainClass2Y)])

testDataset = pd.read_csv('test.txt', sep=" ", header=None, dtype="int64")

count = 0
testClass1X = []
testClass1Y = []

testClass2X = []
testClass2Y = []

for data in testDataset.values:
    x = np.array([data[0], data[1]])
    g1x = np.dot(np.transpose(class1MeanMatrix), x) - np.dot(np.transpose(class1MeanMatrix), class1MeanMatrix) / 2
    g2x = np.dot(np.transpose(class2MeanMatrix), x) - np.dot(np.transpose(class2MeanMatrix), class2MeanMatrix) / 2

    absg1x = abs(g1x)
    absg2x = abs(g2x)

    if g1x > g2x:
        if data[2] == 1:
            count += 1
        testClass1X.append(data[0])
        testClass1Y.append(data[1])
    else:
        if data[2] == 2:
            count += 1
        testClass2X.append(data[0])
        testClass2Y.append(data[1])

plt.scatter(testClass1X, testClass1Y, marker="o", color="red")
plt.scatter(testClass2X, testClass2Y, marker="v", color="blue")
plt.show()

min = min(trainClass1X.__add__(trainClass1Y).__add__(trainClass2X).__add__(trainClass2Y))

xValues = np.arange(min, 7, 0.25)
constant = np.dot(np.transpose(class1MeanMatrix), class1MeanMatrix) - np.dot(np.transpose(class2MeanMatrix),
                                                                             class2MeanMatrix)
print(constant)
