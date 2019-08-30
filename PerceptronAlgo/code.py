# Md. Toufiqul Islam
# ID: 15-02-04-097
# Section: B2

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate



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

def signleWtY(y,w):
    return y[0]*w[0] + y[1]*w[1] + y[2]*w[2] + y[3]*w[3] + y[4]*w[4] + y[5]*w[5]

def isAllClassified(gy):
    for i in gy:
        if i<=0:
            return False
    return True

def manyAtATime(myHighDimensionalArrays,initialWeightVector,ALPHA):
    counter = 1
    while counter <= 200:
        currentGy = gy(myHighDimensionalArrays, initialWeightVector)

        if isAllClassified(currentGy):
            print("For Alpha ",ALPHA," in Many at a time weight Coefficient ",initialWeightVector)
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

def oneAtATime(myHighDimensionalArrays,initialWeightVector,ALPHA):
    updateFunction = []
    counter = 0

    while counter<200:
        counter +=1
        numberOfClassified = 0

        for aHighDimensionalArray in myHighDimensionalArrays:
           wTy = signleWtY(aHighDimensionalArray,initialWeightVector)
           if(wTy<=0):
               updateFunction = aHighDimensionalArray
               modifiedUpdateFunction = [i*ALPHA for i in updateFunction]
               temp = []

               temp.append(modifiedUpdateFunction)
               temp.append(initialWeightVector)
               initialWeightVector = np.sum(temp, axis=0)
           else:
               numberOfClassified +=1

        if numberOfClassified == 6:
            print("For Alpha ", ALPHA, " in One at a time weight Coefficient ", initialWeightVector)
            return counter
    return -1

def drowGroupChart(oneCountArray,manyCountArray,title):
    fig, ax = plt.subplots()
    plt.xlabel('Learning Rate')
    plt.ylabel('Numbers Of Iterations')
    plt.title(title)
    index = alphaValues
    bar_width = 0.025
    opacity = 0.8

    rects1 = plt.bar(index, oneCountArray, bar_width,
                     alpha=opacity,
                     color='b',
                     label='One at a time')

    rects2 = plt.bar(index + bar_width, manyCountArray, bar_width,
                     alpha=opacity,
                     color='y',
                     label='Many at a time')

    plt.xticks(index + bar_width, tuple(alphaValues))
    plt.legend()

    plt.tight_layout()
    plt.show()


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
alphaValues[2] = 0.30
alphaValues[6] = 0.70
table = []
oneCountArray = []
manyCountArray = []
for ALPHA in alphaValues:
    oneCount = oneAtATime(myHighDimensionalArrays,initialWeightVector,ALPHA)
    manyCount = manyAtATime(myHighDimensionalArrays,initialWeightVector,ALPHA)
    oneCountArray.append(oneCount)
    manyCountArray.append(manyCount)
    table.append([ALPHA,oneCount,manyCount])


name = ["Alpha(Learning Rate)","One at a time","Many at a time"]
print("Case 1: Initial Weight Vector All One")
print(tabulate(table, name, tablefmt="youtrack"))

drowGroupChart(oneCountArray,manyCountArray,"Case 1: Initial Weight Vector All One")

initialWeightVector = [0,0,0,0,0,0]
oneCountArray = []
manyCountArray = []

table = []
for ALPHA in alphaValues:
    oneCount = oneAtATime(myHighDimensionalArrays,initialWeightVector,ALPHA)
    manyCount = manyAtATime(myHighDimensionalArrays,initialWeightVector,ALPHA)
    oneCountArray.append(oneCount)
    manyCountArray.append(manyCount)
    table.append([ALPHA,oneCount,manyCount])


name = ["Alpha(Learning Rate)","One at a time","Many at a time"]
print("Case 2: Initial Weight Vector All Zero")
print(tabulate(table, name, tablefmt="youtrack"))

drowGroupChart(oneCountArray,manyCountArray,"Case 2: Initial Weight Vector All Zero")

initialWeightVector = np.random.random(6)
oneCountArray = []
manyCountArray = []
table = []
for ALPHA in alphaValues:
    oneCount = oneAtATime(myHighDimensionalArrays,initialWeightVector,ALPHA)
    manyCount = manyAtATime(myHighDimensionalArrays,initialWeightVector,ALPHA)
    oneCountArray.append(oneCount)
    manyCountArray.append(manyCount)
    table.append([ALPHA,oneCount,manyCount])


name = ["Alpha(Learning Rate)","One at a time","Many at a time"]
print("Case 3: Initial Weight Vector Random Numbers ",initialWeightVector)
print(tabulate(table, name, tablefmt="youtrack"))
drowGroupChart(oneCountArray,manyCountArray,"Case 3: Initial Weight Vector Random Numbers")