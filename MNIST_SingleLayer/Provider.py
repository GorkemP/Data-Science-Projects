import numpy as np


def numToArray(aNumber):
    if aNumber == 0:
        return np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    elif aNumber == 1:
        return np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    elif aNumber == 2:
        return np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    elif aNumber == 3:
        return np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
    elif aNumber == 4:
        return np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
    elif aNumber == 5:
        return np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
    elif aNumber == 6:
        return np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
    elif aNumber == 7:
        return np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
    elif aNumber == 8:
        return np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
    elif aNumber == 9:
        return np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])

def numArrayToBinaryArray(anArray):
    v = np.zeros((np.shape(anArray)[1], 10))
    for i in range(np.shape(anArray)[1]):
        v[i, :] = numToArray(anArray.item(i)).reshape(1,10)
    return v