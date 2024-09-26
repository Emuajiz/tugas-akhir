from PIL import Image
import numpy as np
import random


def readImage(filename):
    image = Image.open(filename)
    return np.array(image).astype(dtype=np.uint8)


def createSubBlock(data, size):
    assert data.shape[0] % size == 0
    assert data.shape[1] % size == 0
    height = data.shape[0] // size
    width = data.shape[1] // size
    res = np.zeros((height, width, size, size), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            posx = x*size+size
            posy = y*size+size
            res[y, x] = data[y*size:posy, x*size:posx]
    return res


def mergeSubBlock(data):
    res = np.zeros((data.shape[0] * data.shape[2],
                   data.shape[1] * data.shape[2]), dtype=np.uint8)
    for y in range(data.shape[0]):
        for x in range(data.shape[1]):
            posx = x * data.shape[2]
            posy = y * data.shape[2]
            res[posy:posy+data.shape[2], posx:posx+data.shape[2]] = data[y, x]
    return res


def calculateAuthenticationBit(data, salt):
    data = data.flatten()
    data = data >> 2
    res = 0
    while salt > 0:
        res = res + salt % 2
        salt = salt >> 1
    for i in data:
        while i > 0:
            res = res + i % 2
            i = i >> 1
    return res % 4


def calculateRecoveryBit(data):
    data = data.flatten()
    data = data >> 2
    return np.average(data).astype(np.uint8)


def calculateWatermarkData(authenticationBit, recoveryBit, salt):
    stringAuthenticationBit = bin(authenticationBit)[2:]
    stringAuthenticationBit = "00" + stringAuthenticationBit
    stringAuthenticationBit = stringAuthenticationBit[-2:]
    stringRecoveryBit = bin(recoveryBit)[2:]
    stringRecoveryBit = "000000" + stringRecoveryBit
    stringRecoveryBit = stringRecoveryBit[-6:]
    res = stringAuthenticationBit + stringRecoveryBit
    random.seed(salt)
    res = list(res)
    random.shuffle(res)
    return ''.join(res)


def arnoldMap(x, y, width, height, iteration):
    resx = x
    resy = y
    while iteration > 0:
        tmpx = resx
        tmpy = resy
        resx = (2 * tmpx + tmpy) % width
        resy = (tmpx + tmpy) % height
        iteration = iteration - 1
    return (resx, resy)


def reverseArnoldMap(x, y, width, height, iteration):
    resx = x
    resy = y
    while iteration > 0:
        tmpx = resx
        tmpy = resy
        resx = (tmpx - tmpy) % width
        resy = (-tmpx + 2 * tmpy) % height
        iteration = iteration - 1
    return (resx, resy)


def embedWatermark(data, watermarkData):
    res = np.zeros(data.shape, dtype=np.uint8)
    counter = 0
    for y, _ in enumerate(data):
        for x, _ in enumerate(data[y]):
            pixelData = data[y, x] >> 2
            pixelData = pixelData << 2
            watermark = watermarkData[counter:counter+2]
            watermark = int(watermark, 2)
            res[y, x] = pixelData + watermark

            counter = counter + 2
    return res


if __name__ == "__main__":
    img = readImage("test1-marked.png")
    subBlock = createSubBlock(img, 2)

    size = (subBlock.shape[0], subBlock.shape[1])
    authenticationBits = np.zeros(size, dtype=np.uint)
    recoveryBits = np.zeros(size, dtype=np.uint8)
    counter = 0

    res = np.zeros(subBlock.shape, dtype=np.uint8)
    for y, _ in enumerate(subBlock):
        for x, _ in enumerate(subBlock[y]):
            tmpmap = arnoldMap(x, y, size[1], size[0], 10)
            salt = tmpmap[0] + tmpmap[1]
            recoveryBits[y, x] = calculateRecoveryBit(
                subBlock[tmpmap[0], tmpmap[1]])
            authenticationBits[y, x] = calculateAuthenticationBit(
                subBlock[y, x], salt)
            watermarkData = calculateWatermarkData(
                authenticationBits[y, x], recoveryBits[y, x], salt)
            res[y, x] = embedWatermark(subBlock[y, x], watermarkData)
            counter = counter + 1
    final = mergeSubBlock(res)
    Image.fromarray(final).show()
    print("complete")
