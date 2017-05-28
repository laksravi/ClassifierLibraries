'''
Created on Mar 13, 2016

@author: Lakshmi Ravi
'''
from os.path import sys
import os
import math
directory = sys.argv[1]
totalCharacterFreq = {}
targetCharacterCount = {}
actualTrainingFreq = {}
fileCharacterMap = {}
selectedTrainingFiles = []
totalFiles = 0


def sumMergeDict(dict1, dict2):
    dict(dict1.items() + dict2.items() +
         [(k, (dict1[k] + dict2[k])) for k in set(dict1) & set(dict2)]
         )


def getCost(dict1, refDict, trainDict):
    cost = 0
    for item in refDict:
        freq = 0
        if item in dict1:
            freq = dict1[item]
        if item in trainDict:
            freq += trainDict[item]
        freqRef = refDict[item]
        if(freq > 0):
            cost += math.log(freqRef)
            cost -= math.log(freq)
    return cost


def getCharacterCount():
    # count the characters in
    # from the output directory count the
    global totalFiles
    opDir = directory + "\\output"
    for fileName in os.listdir(opDir):
        totalFiles += 1
        fileCharacterCount = {}
        f = os.path.join(opDir, fileName)
        for line in open(f):
            if "EO," in line:
                break
            if "E," in line:
                break
            if "O," in line or "N," in line:
                lineElements = line.split(",")
                lineElements[2] = lineElements[2].replace(' ', '')
                if lineElements[2] in totalCharacterFreq:
                    totalCharacterFreq[lineElements[2]] += 1
                else:
                    totalCharacterFreq[lineElements[2]] = 1
                # update for the file alone
                if lineElements[2] in fileCharacterCount:
                    fileCharacterCount[lineElements[2]] += 1
                else:
                    fileCharacterCount[lineElements[2]] = 1
        fileCharacterMap[f] = fileCharacterCount

    # compute target frequency
    for item in totalCharacterFreq:
        targetCharacterCount[item] = totalCharacterFreq[item]
    print(targetCharacterCount)
    '''
    for item in fileCharacterMap:
        print(getCost(fileCharacterMap[item], targetCharacterCount))
        print(fileCharacterMap[item])
    '''
    # print(fileCharacterMap)
    print("Files read in the Output directory", totalFiles)


def updateTarget(freq):
    for item in freq:
        #targetCharacterCount[item] -= freq[item]
        if item in actualTrainingFreq:
            actualTrainingFreq[item] += freq[item]
        else:
            actualTrainingFreq[item] = freq[item]


def findtrainFilesFromFreq():
    maxCost = -99999
    chosenFile = None

    for item in fileCharacterMap:
        ci = getCost(
            fileCharacterMap[item], targetCharacterCount, actualTrainingFreq)
        if ci > maxCost:
            maxCost = ci
            chosenFile = item
    if chosenFile == None:
        return
    selectedTrainingFiles.append(chosenFile)
    updateTarget(fileCharacterMap[chosenFile])
    del fileCharacterMap[chosenFile]


def isTargetReached():
    count = 0
    for item in targetCharacterCount:
        count += targetCharacterCount[item]
    if(count <= 0):
        return True
    return False


def freqDiff(dictTotal, dict1):
    dict2 = {}
    for item in dict1:
        dict2[item] = dictTotal[item] - dict1[item]
    return dict2


def findTotalSamples(dict1):
    count = 0
    for item in dict1:
        count += dict1[item]
    return count


def findTrainFiles():
    p = int(2 * totalFiles / 3)
    while(len(selectedTrainingFiles) < p):
        findtrainFilesFromFreq()
    print(len(selectedTrainingFiles), "Is the number of files for training")


def writeDistributions(trainFreq, testFreq):
    trainSamples = findTotalSamples(trainFreq)
    testSamples = findTotalSamples(testFreq)
    file = open("NewDistributions.csv", 'a+')
    file.write(
        "CLASSES,TrainSampleCount,TestSampleCount,Total\n")
    for item in trainFreq:
        file.write(item + "," + str(trainFreq[item]) + ",")
        #file.write(str(trainFreq[item] / trainSamples * 100) + ",")
        if item in testFreq:
            file.write(str(testFreq[item]) + ",")
            #file.write(str(testFreq[item] / testSamples * 100))
        file.write(str(totalCharacterFreq[item]) + ",")
        file.write("\n")


if(len(sys.argv) < 2):
    print(
        "Please Enter the File-Path as python a3.py <Folder_With_INKML_FILES>")
getCharacterCount()

print("Total Frequency", findTotalSamples(totalCharacterFreq))
print(totalCharacterFreq)
print(targetCharacterCount)

findTrainFiles()
print("Total Training Frequency", findTotalSamples(actualTrainingFreq))
print(actualTrainingFreq)
testFreq = freqDiff(totalCharacterFreq, actualTrainingFreq)
print("Test Frequency", findTotalSamples(testFreq))
print(testFreq)
writeDistributions(actualTrainingFreq, testFreq)
