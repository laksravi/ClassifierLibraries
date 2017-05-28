import numpy as np
import matplotlib.pyplot as mpl
import math


def findMean(X):
    # Find mean of all features, returns mean vector
    f = len(X[0])
    mean = [[0 for i in range(0, 1)] for j in range(0, f)]
    for i in range(0, len(X)):
        for j in range(0, f):
            mean[j][0] += X[i][j]
    for j in range(0, f):
        mean[j][0] /= len(X)
    return mean


def findVariance(X, feature, mean):
    # Find variance of the given feature
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
    w0 = 0
    w1 = 0
    p0 = 0.0
    for i in range(0, len(Y)):
        if(Y[i][0] == 1.0):
            w1 += 1
        else:
            w0 += 1
            p0 += 1.0
    p0 /= len(Y)
    #w1 /= Y.size
    return w0, w1, p0


def readData():
    data = np.load("data.npy")
    row, column = (data.shape)
    X = [[0 for j in range(0, column - 1)]for i in range(0, row)]
    for i in range(0, row):
        for j in range(0, column - 1):
            X[i][j] = data[i][j]
    # Set Output Matrix
    Y = [[data[j][column - 1] for i in range(1)] for j in range(0, row)]
    w0, w1, p0 = findPrior(Y)
    X0 = [[0 for j in range(0, column - 1)]for i in range(0, w0)]
    X1 = [[0 for j in range(0, column - 1)] for i in range(0, w1)]
    m1 = 0
    m2 = 0
    for i in range(0, row):
        if(Y[i][0] == 1.0):
            for j in range(0, column - 1):
                X1[m1][j] = X[i][j]
            m1 += 1
        else:
            for j in range(0, column - 1):
                X0[m2][j] = X[i][j]
            m2 += 1
    return X, Y, X0, X1


def findClassConditionalProbablity(X, M, CoVar):
    c = 1.0 / (math.sqrt(np.linalg.det(CoVar)))
    xi = np.transpose(np.matrix(X))
    mean = np.matrix(M)
    xi = xi - mean
    Cov_xi = (np.linalg.inv(CoVar))
    result = np.dot(np.transpose(xi), Cov_xi)
    result = np.dot(result, xi)
    val = math.exp(-0.5 * (result[0][0]))
    return c * val
    # print(xi)
    # print(mean)


def plotDecisionBoundary(X, Y, Yc, mean0, mean1, coVarnce0, coVarnce1):
    mpl.figure()
    mpl.title('Map Classifier')
    mpl.ylabel('Attribute 2')
    mpl.xlabel('Attribute 1')
    # Found the data is of the range (-4,4)
    mpl.xlim(-4, 4)
    mpl.ylim(-4, 4)
    count = len(X)
    for i in range(0, count):
        if(Y[i][0] == 0.0):
            mpl.plot(X[i][0], X[i][1], 'bo')
        else:
            mpl.plot(X[i][0], X[i][1], 'ro')

    w0, w1, pw0 = findPrior(Y)
    pw1 = 1 - pw0
    mat = [[1.0 for i in range(-40, 40)] for j in range(-40, 40)]
    for i in range(-40, 40):
        for j in range(-40, 40):
            pt = [i / 10.0, j / 10.0]
            p_X_w0 = findClassConditionalProbablity(pt, mean0, coVarnce0)
            p_X_w1 = findClassConditionalProbablity(pt, mean1, coVarnce1)
            p_w0_X = p_X_w0 * pw0
            p_w1_X = p_X_w1 * pw1
            if p_w0_X >= p_w1_X:
                mpl.plot(i / 10.0, j / 10.0, 'b.')
                mat[i + 40][j + 40] = 0.0
            else:
                mpl.plot(i / 10.0, j / 10.0, 'r.')

    # Boundary is where both the
    for i in range(-39, 39):
        x = i / 10
        xPrev = 40 + i - 1
        xNext = 40 + i + 1
        for j in range(-39, 39):
            y = j / 10
            yPrev = 40 + j - 1
            yNext = 40 + j + 1
            if(mat[i + 40][j + 40] != mat[xPrev][j + 40]):
                mpl.plot(x - 0.05, y, 'k.')
            elif (mat[i + 40][j + 40] != mat[xNext][j + 40]):
                mpl.plot(x + 0.05, y, 'k.')
            elif(mat[i + 40][j + 40] != mat[i + 40][yPrev]):
                mpl.plot(x, y - 0.05, 'k.')
            elif (mat[i + 40][j + 40] != mat[i + 40][yNext]):
                mpl.plot(x, y + 0.05, 'k.')
    mpl.show()


def computeOutput():
    X, Y, X0, X1 = readData()
    count = len(X)
    w0, w1, pw0 = findPrior(Y)
    pw1 = 1 - pw0
    # Find mean and covariance for class-0 and class-1
    mean0 = findMean(X0)
    mean1 = findMean(X1)
    coVarnce0 = findCoVarinaceMatrix(X0, mean0)
    coVarnce1 = findCoVarinaceMatrix(X1, mean1)
    Yc = [-1.0 for i in range(0, count)]
    # for every sample find the expected output
    for i in range(0, count):
        p_X_w0 = findClassConditionalProbablity(X[i], mean0, coVarnce0)
        p_X_w1 = findClassConditionalProbablity(X[i], mean1, coVarnce1)
        p_w0_X = p_X_w0 * pw0
        p_w1_X = p_X_w1 * pw1
        #print(p_X_w0, " ", p_X_w1)
        if p_w0_X < p_w1_X:
            Yc[i] = 1.0
        else:
            Yc[i] = 0.0

    print(
        "\n\nConfusion Matrix <Actual class as rows and Computes class as columns>")
    ConfusionMatrix = [[0 for i in range(0, 2)]for j in range(0, 2)]
    accuracy = 0
    for i in range(0, count):
        x = int(Y[i][0])
        y = int(Yc[i])
        ConfusionMatrix[x][y] += 1
        if(x == y):
            accuracy += 1.0
    print(ConfusionMatrix)
    accuracy = (accuracy * 100.0) / count
    print("\n The accuracy of this Algorithm is ", accuracy, " %")
    plotDecisionBoundary(X, Y, Yc, mean0, mean1, coVarnce0, coVarnce1)
computeOutput()
