# Md. Toufiqul Islam
# ID: 15-02-04-097
# Section: B2

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def getHignDimensionalvalue(x1,x2,isNormalized):
    if not isNormalized:
        return [x1*x1,x2*x2,x1*x2,x1,x2,1]
    else:
        return [x1*x1*-1,x2*x2*-1,x1*x2*-1,x1*-1,x2*-1,1*-1]

def gy(y,wT):
    ans = []
    for aY in y:
        ans.append(aY[0]*wT[0]+aY[1]*wT[1]+aY[2]*wT[2]+aY[3]*wT[3]+aY[4]*wT[4]+aY[5]*wT[5])
    return ans

def isAllClassified(gy):
    for i in gy:
        if i<=0:
            return False
    return True

def manyAtATime(initialWeightVector,ALPHA):
    counter = 1
    while counter <= 200:
        currentGy = gy(myHighDimensionalArrays, initialWeightVector)

        if isAllClassified(currentGy):
           return counter
        else:
            misclassified = []
            for i in range(0, len(currentGy)):
                if currentGy[i] <= 0:
                    misclassified.append(myHighDimensionalArrays[i])

            modifiedMissClassified = []
            for aMissClassified in misclassified:
                modifiedMissClassified.append([i * ALPHA for i in aMissClassified])

            modifiedMissClassified.append(initialWeightVector)

            initialWeightVector = np.sum(modifiedMissClassified, axis=0)

        counter += 1
    return -1

def oneAtATime(initialWeightVector,ALPHA):
    counter = 1
    while counter <= 200:
        currentGy = gy(myHighDimensionalArrays, initialWeightVector)

        if isAllClassified(currentGy):
           return counter
        else:
            misclassified = []
            for i in range(0, len(currentGy)):
                if currentGy[i] <= 0:
                    misclassified.append(myHighDimensionalArrays[i])

            modifiedMissClassified = []
            for aMissClassified in misclassified:
                modifiedMissClassified.append([i * ALPHA for i in aMissClassified])

            modifiedMissClassified.append(initialWeightVector)

            initialWeightVector = np.sum(modifiedMissClassified, axis=0)

        counter += 1
    return -1


trainDataset = pd.read_csv('Perceptrontrain.txt', sep=" ", header=None, dtype='float64')

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
#plt.show()


# MeanClassifier
class1MeanMatrix = np.array([np.mean(class1X), np.mean(class1Y)])
class2MeanMatrix = np.array([np.mean(class2X), np.mean(class2Y)])

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
#plt.show()

myHighDimensionalArrays = []
for i in range(len(class1X)):
    myHighDimensionalArrays.append(getHignDimensionalvalue(class1X[i],class1Y[i],False))

for i in range(len(class2X)):
    myHighDimensionalArrays.append(getHignDimensionalvalue(class2X[i],class2Y[i],True))

initialWeightVector = [1,1,1,1,1,1]
alphaValues = np.arange(0.1,1.1,0.1)

for ALPHA in alphaValues:
    count = manyAtATime(initialWeightVector,ALPHA)
    if count != -1:
        print("For Alpha ",ALPHA," Many at a time ",count)

print()
initialWeightVector = [0,0,0,0,0,0]

for ALPHA in alphaValues:
    count = manyAtATime(initialWeightVector,ALPHA)
    if count != -1:
        print("For Alpha ",ALPHA," Many at a time ",count)