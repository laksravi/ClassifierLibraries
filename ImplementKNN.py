'''
Created on Feb 4, 2016
@author: Lakshmi Ravi
'''
import numpy as np
import matplotlib.pyplot as pl
from MaxHeap import MaxHeap


def readData():
    # reads data from external file and returns the input and output
    data = np.load("data.npy")
    row, column = (data.shape)
    print(row, column)
    # Initialize Input-matrix (with Bias-input set 1)
    X = [[1 for j in range(0, column)]for i in range(0, row)]
    for i in range(0, row):
        for j in range(1, column):
            X[i][j] = data[i][j - 1]
    # Set Output Matrix
    Y = [[data[j][column - 1] for i in range(1)] for j in range(0, row)]
    return X, Y


def KNearestPlot(K):
    data = np.load("data.npy")
    row, column = data.shape
    pl.figure()
    pl.title(str(K) + '-Nearest-Plot')
    pl.ylabel('Attribute 2')
    pl.xlabel('Attribute 1')
    # Found the data is of the range (-4,4)
    pl.xlim(-4, 4)
    pl.ylim(-4, 4)
    # Computed output - which is later used to mark boundary
    cmpOp = [[0 for i in range(-40, 50)] for j in range(-40, 50)]
    # Defined a range and classify the data in that region
    for i in range(-40, 50):
        x = i / 10
        for k in range(-40, 50):
            y = k / 10
            mh = MaxHeap(K)
            # Find the K nearest neighbor in the training data
            #Compute in Max-heap
            for j in range(0, row):
                # use sum of squared differences to find error
                distance = pow(abs(x - data[j][0]), 2) + \
                    pow(abs(y - data[j][1]), 2)
                if mh.isFull():
                    maxDis = mh.getMax()
                    if(distance < maxDis):
                        mh.removeMax()
                    else:
                        continue
                mh.insert(distance, data[j][2])
            Yc = mh.getAverage()
            if(Yc == 0.0):
                pl.plot(x, y, 'b.')
            else:
                pl.plot(x, y, 'r.')
            cmpOp[40 + i][40 + k] = Yc

    # Go through the matrix-space and see if the consecutive values differ.
    # If yes, plot a boundary mark
    for i in range(-39, 49):
        x = i / 10
        xPrev = 40 + i - 1
        xNext = 40 + i + 1
        for j in range(-39, 49):
            y = j / 10
            yPrev = 40 + j - 1
            yNext = 40 + j + 1
            if(cmpOp[i + 40][j + 40] != cmpOp[xPrev][j + 40]):
                pl.plot(x - 0.05, y, 'k.')
            elif (cmpOp[i + 40][j + 40] != cmpOp[xNext][j + 40]):
                pl.plot(x + 0.05, y, 'k.')
            elif(cmpOp[i + 40][j + 40] != cmpOp[i + 40][yPrev]):
                pl.plot(x, y - 0.05, 'k.')
            elif (cmpOp[i + 40][j + 40] != cmpOp[i + 40][yNext]):
                pl.plot(x, y + 0.05, 'k.')

    plotTrainData(K)


def plotTrainData(K):
    # Plot the training data in the same figure
    data = np.load("data.npy")
    row, column = data.shape
    for i in range(0, row):
        mh = MaxHeap(K)
        # Find the K nearest neighbors for training data
        for j in range(0, row):
            distance = pow(abs(data[i][0] - data[j][0]), 2) + \
                pow(abs(data[i][1] - data[j][1]), 2)
            if mh.isFull():
                maxDis = mh.getMax()
                if(distance < maxDis):
                    mh.removeMax()
                else:
                    continue
            mh.insert(distance, data[j][2])
        Yc[i] = mh.getAverage()
        # color class-O as blue and Class-1 as red
        if(data[i][2] == 0.0):
            pl.plot(data[i][0], data[i][1], 'bo')
        else:
            pl.plot(data[i][0], data[i][1], 'ro')
    confusionMatrix = [
        [0 for i in range(0, column - 1)]for j in range(0, column - 1)]
    for i in range(row):
        x = int(data[i][2])
        y = int(Yc[i])
        confusionMatrix[x][y] = confusionMatrix[x][y] + 1
    print("Confusion Matrix -", K, "-Nearest neighbor")
    print("Actual output as rows and computed-output as columns")
    for i in range(0, column - 1):
        for j in range(0, column - 1):
            print(confusionMatrix[i][j])
        print("")
    correctness = 0
    for i in range(column - 1):
        correctness += confusionMatrix[i][i]
    print("\tPercentage of correctness", (correctness * 100) / Yc.size)
    pl.show()


def LeastSquare():
    # Analytically computes the expected output and returns the computed output
    X, Y = readData()
    # Find Transpose
    XT = np.transpose(X)
    # Find inverse of square matrix
    XT_X_In = np.linalg.inv(np.dot(XT, X))
    Beta = np.dot(np.dot(XT_X_In, XT), Y)
    # Expected Output
    Yc = np.dot(X, Beta)
    return X, Y, Yc, Beta


def markLinearBoundary(X, Y, Yc, Beta):
    # Plots the Graph with given input, output and mark linear boundary
    pl.figure()
    pl.title('Least-Square')
    pl.ylabel('Attribute 2')
    pl.xlabel('Attribute 1')
    pl.figaspect(4)
    pl.xlim(-4, 4)
    pl.ylim(-4, 4)
    confusionMatrix = [
        [0 for i in range(Beta.size - 1)] for j in range(Beta.size - 1)]
    for i in range(Yc.size):
        x = int(Y[i][0])
        y = 0
        if(Yc[i] >= 0.5):
            y = 1
        confusionMatrix[x][y] += 1
    print("Confusion-Matrix after Least Square")
    print("Actual output as rows and computed-output as columns")
    for i in range(Beta.size - 1):
        for j in range(Beta.size - 1):
            print(confusionMatrix[i][j])
        print("")
    # Calculate percentage of correctness
    correctness = 0
    for i in range(Beta.size - 1):
        correctness += confusionMatrix[i][i]
    print("\tPercentage of correctness", (correctness * 100) / Yc.size)
    for i in range(0, Yc.size):
        if Y[i][0] == 0.0:
            pl.plot(X[i][1], X[i][2], 'ro')
        else:
            pl.plot(X[i][1], X[i][2], 'bo')
    # Plot the linear separator
    for i in range(-40, 50):
        for j in range(-40, 50):
            x = i / 10
            y = j / 10
            lineY = (0.5 - Beta[0] - Beta[1] * x) / Beta[2]
            if (y - lineY) > 0.0:
                pl.plot(x, y, 'b.')
            else:
                pl.plot(x, y, 'r.')
    Xrange = [i for i in range(-4, 5)]
    Yrange = [0 for i in range(-4, 5)]
    for i in range(-4, 5):
        lineY = (0.5 - Beta[0] - Beta[1] * i) / Beta[2]
        Yrange[i + 4] = lineY
    #pl.plot(Xrange, Yrange)
    pl.show()


print("\n\n********************Least-Square**************************")
X, Y, Yc, Beta = LeastSquare()
markLinearBoundary(X, Y, Yc, Beta)
print("\n\n*******************15-Nearest neighbor**********************")
KNearestPlot(15)
print("\n\n********************1-nearest neighbor*********************")
KNearestPlot(1)
