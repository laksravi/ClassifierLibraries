'''
Created on Feb 6, 2016

@author: Lakshmi Ravi
'''


class Node():

    def __init__(self, distance, output):
        self.distance = distance
        self.output = output


class MaxHeap():

    def __init__(self, K):
        self.K = K
        self.n = [None for i in range(0, K)]
        self.count = -1

    def getMax(self):
        return self.n[0].distance

    def insert(self, distance, output):
        # print(self.count)
        self.count = self.count + 1
        self.n[self.count] = Node(distance, output)
        # percolate up
        k = self.count
        while(k > 0):
            par = int((k - 1) / 2)
            if self.n[k].distance > self.n[par].distance:
                temp = self.n[k]
                self.n[k] = self.n[par]
                self.n[par] = temp
                k = par
            else:
                break

    def removeMax(self):
        self.n[0] = self.n[self.count]
        self.n[self.count] = None
        self.count = self.count - 1
        # percolate down
        k = 0
        while(k < int(self.count / 2)):
            child1 = 2 * k
            child2 = 2 * k + 1
            par = 2 * k
            if (self.n[child1].distance > self.n[child2].distance):
                par = child1
            elif child2 < self.count:
                par = child2
            else:
                break
            if self.n[k].distance < self.n[par].distance:
                temp = self.n[k]
                self.n[k] = self.n[par]
                self.n[par] = temp
                k = par
            else:
                break

    def printHeap(self):
        for i in range(self.count + 1):
            print(self.n[i].distance)

    def isFull(self):
        return self.count == self.K - 1

    def getAverage(self):
        AvgComp = 0
        for i in range(0, self.count + 1):
            AvgComp += self.n[i].output
        AvgComp = AvgComp / (self.count + 1)
        if AvgComp >= 0.5:
            return 1.0
        return 0.0

'''
# testFunctions
mh = MaxHeap(5)
mh.insert(4, 0)
mh.insert(5, 0)
mh.removeMax()
mh.printHeap()
'''
