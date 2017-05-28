import numpy as np
import math
import matplotlib.pyplot as mpl
cost = [[-200000, 0.07, 0.07, 0.07], [0.07, -0.15, 0.07, 0.07],
        [0.07, 0.07, -0.05, 0.07], [0.03, 0.03, 0.03, 0.03]]


Impcost = [[-0.30, 0.07, 0.07, 0.07], [0.07, -0.225, 0.07, 0.07],
           [0.07, 0.07, -0.075, 0.07], [0.03, 0.03, 0.03, 0.03]]
scrapModifiedCost = [[-0.3326, 0.116, 0.116, 0.07], [0.116, -0.2495, 0.116, 0.07],
                     [0.116, 0.116, -0.0815, 0.07], [0.049, 0.049, 0.049, 0.03]]
# Attribute to change Prior : changes prior on set true
pIncrease = True
# File name to train the classifier
fileName = "nuts_bolts.csv"


def findLessRisky(p, cost):
    # for the defined cost function computes the less risky class
    if(cost == None):
        return findMaxClass(p)
    Risk = [0 for i in range(0, 4)]
    for i in range(0, 4):
        for j in range(0, 4):
            Risk[i] += cost[i][j] * p[j]
    R = min(Risk)
    for i in range(0, 4):
        if Risk[i] == R:
            return i + 1


def findMaxClass(P):
    M = max(P)
    for i in range(0, 4):
        if(P[i] == M):
            return i + 1


def findMean(X):
    # Find mean of all features
    f = len(X[0])
    mean = [[0 for i in range(0, 1)] for j in range(0, f)]
    for i in range(0, len(X)):
        for j in range(0, f):
            mean[j][0] += X[i][j]
    for j in range(0, f):
        mean[j][0] /= len(X)
    return mean


def findVariance(X, feature, mean):
    varnce = 0
    for i in range(0, len(X)):
        varnce += pow((X[i][feature] - mean), 2)
    varnce = varnce / X.length
    return varnce


def findCoVarinaceMatrix(X, mean):
    features = len(X[0])
    CoVar = [[0 for i in range(0, features)] for j in range(0, features)]
    # Find mean of every feature
    samCnt = len(X)
    for i in range(0, samCnt):
        # For each sample
        for j in range(0, features):
            for k in range(0, features):
                # Consider possible pairs of feature
                CoVar[j][k] += (X[i][j] - mean[j][0]) * (X[i][k] - mean[k][0])
    for j in range(0, features):
        for k in range(0, features):
            CoVar[j][k] /= samCnt
    return CoVar


def findPrior(Y):
    # for a 2-class problem, given the output
    count = [0 for i in range(0, 4)]
    P = [0.0 for i in range(0, 4)]
    for i in range(0, len(Y)):
        index = (int)(Y[i][0]) - 1
        count[index] += 1
        P[index] += 1.0
    X = len(Y)
    if(pIncrease == True):
        P[0] += 10000.0
        X += 10000.0
    for i in range(0, 4):
        P[i] /= X
    return count, P


def findClassConditionalProbablity(X, M, CoVar):
    # for a given sample, find the probablity of occurance
    # Assume the sample is normally distributed
    c = 1.0 / (math.sqrt(np.linalg.det(CoVar)))
    xi = np.transpose(np.matrix(X))
    mean = np.matrix(M)
    xi = xi - mean
    Cov_xi = (np.linalg.inv(CoVar))
    result = np.dot(np.transpose(xi), Cov_xi)
    result = np.dot(result, xi)
    val = math.exp(-0.5 * (result[0][0]))
    return c * val


def readData():
    file = open(fileName)
    # Get and set the samples and features
    N = len(file.readlines())
    columns = 3

    # Set the Input-X and Output X
    X = [[0 for j in range(0, columns - 1)]for i in range(0, N)]
    Y = [[0 for j in range(0, 1)] for i in range(0, N)]
    # read data from the data file
    i = 0
    for line in open(fileName):
        LineElements = line.split(",")
        X[i] = float(LineElements[0]), float(LineElements[1])
        Y[i][0] = int(LineElements[2])
        i = i + 1
    # holds the count of samples in each class
    classCount = [0 for i in range(0, 4)]
    ccIndex = [0 for i in range(0, 4)]
    for i in range(0, i):
        index = Y[i][0] - 1
        classCount[index] += 1
    # Create
    classInput = [
        [[0 for k in range(0, columns - 1)] for j in range(0, classCount[i])] for i in range(0, 4)]

    for i in range(0, N):
        index = Y[i][0] - 1
        for j in range(0, columns - 1):
            classInput[index][ccIndex[index]][j] = X[i][j]
        ccIndex[index] += 1
    return X, Y, classInput


def printMatrix(Matrix):
    rows = len(Matrix)
    cols = len(Matrix[0])
    for i in range(0, rows):
        for j in range(0, cols):
            print(Matrix[i][j])
        print("")


def executeCostMAP(Cost):
    X, Y, splitX = readData()
    N = len(X)
    f = len(X[0])
    cCount, prior = findPrior(Y)
    # Find the mean and Co-variance foe each class
    classMean = [[] for i in range(0, 4)]
    classCoVar = [[[]]for i in range(0, 4)]
    # Compute mean and covariance for each class
    for i in range(0, 4):
        classMean[i] = findMean(splitX[i])
        classCoVar[i] = findCoVarinaceMatrix(splitX[i], classMean[i])
    Yc = [0 for i in range(0, N)]
    for i in range(0, N):
        # for each class find the probablity
        P_Class_X = [0 for i in range(0, 4)]
        for j in range(0, 4):
            P_Class_X[j] = findClassConditionalProbablity(
                X[i], classMean[j], classCoVar[j]) * prior[j]
        Yc[i] = findLessRisky(P_Class_X, Cost)

    # From the Computed Output find the confusion matrix and accuracy
    confusionMatrix = [[0 for i in range(0, 4)]for j in range(0, 4)]
    accuracy = 0.0
    for i in range(0, N):
        confusionMatrix[Y[i][0] - 1][Yc[i] - 1] += 1
        if(Yc[i] == Y[i][0]):
            accuracy += 1.0
    accuracy /= N
    print("\n\n", accuracy, " is the accuracy ratio")
    print("\n*****************Confusion Matrix*******************")
    printMatrix(confusionMatrix)
    plotDecisionBoundary(X, Y, Yc, prior, classMean, classCoVar, Cost)


def plotDecisionBoundary(X, Y, Yc, prior, classMean, classCoVar, Cost):
    mpl.figure()
    mpl.title('Decision boundary')
    mpl.ylabel('Attribute 2')
    mpl.xlabel('Attribute 1')
    # Found the data is of the range (-4,4)
    mpl.xlim(-0.3, 1)
    mpl.ylim(-0.3, 1)
    count = len(X)
    # Mark the training samples
    for i in range(0, count):
        if(Y[i][0] == 1):
            mpl.plot(X[i][0], X[i][1], 'ro')
        elif(Y[i][0] == 2):
            mpl.plot(X[i][0], X[i][1], 'bo')
        elif(Y[i][0] == 3):
            mpl.plot(X[i][0], X[i][1], 'mo')
        elif(Y[i][0] == 4):
            mpl.plot(X[i][0], X[i][1], 'co')

    mat = [[0 for i in range(-30, 100)] for j in range(-30, 100)]
    for x in range(-30, 100):
        px = x / 100
        for y in range(-30, 100):
            py = y / 100
            point = [px, py]
            P_Class_X = [0 for i in range(0, 4)]
            for k in range(0, 4):
                P_Class_X[k] = findClassConditionalProbablity(
                    point, classMean[k], classCoVar[k]) * prior[k]
            mat[30 + x][30 + y] = findLessRisky(P_Class_X, Cost)
            if(mat[30 + x][30 + y] == 1):
                mpl.plot(px, py, 'r.')
            elif(mat[30 + x][30 + y] == 2):
                mpl.plot(px, py, 'b.')
            elif(mat[30 + x][30 + y] == 3):
                mpl.plot(px, py, 'm.')
            elif(mat[30 + x][30 + y] == 4):
                mpl.plot(px, py, 'c.')

    for i in range(-29, 99):
        x = i / 100
        xPrev = 30 + i - 1
        xNext = 30 + i + 1
        for j in range(-29, 99):
            y = j / 100
            yPrev = 30 + j - 1
            yNext = 30 + j + 1
            if(mat[30 + i][30 + j] != mat[xPrev][j + 30]):
                mpl.plot(x - 0.005, y, 'k.')
            elif (mat[30 + i][30 + j] != mat[xNext][j + 30]):
                mpl.plot(x + 0.005, y, 'k.')
            elif(mat[30 + i][30 + j] != mat[30 + i][yPrev]):
                mpl.plot(x, y - 0.005, 'k.')
            elif (mat[30 + i][30 + j] != mat[30 + i][yNext]):
                mpl.plot(x, y + 0.005, 'k.')
    mpl.show()


print("\nExecuting with Cost Matrix")
executeCostMAP(cost)
print("\nExecuting - Uniform Cost ")
executeCostMAP(None)
