from enum import Enum
from Queue import Queue


def func():
    a = ["xx1", "abc", "abcd", "xxxcde"]
    print(a)
    for word in a:
        if ("xx") in word:
            a.remove(word)
    print(a)

# func()


class Test():

    @staticmethod
    def someStatic():
        Test.staticObj = 10

    def __init__(self):
        self.something = 1

    def incMyCount(self):
        Test.staticObj += 1

    def printS(self):
        print(Test.staticObj)


def X():
    # Y()
    obj = Test()
    obj.incMyCount()
    obj.printS()


def Y():
    Test.someStatic()


class Etest(Enum):
    One = 1
    Two = 2


def Etester():
    for items in Etest:
        print(items.name)
    print(Etest.One.value)
    c = [None for i in Etest]
    print(c)

# Etester()


def QueuTest():
    a = Queue()
    a.put("A")
    print(a.maxsize)


QueuTest()