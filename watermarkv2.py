from PIL import Image
import numpy as np
import random
import math
import scipy

ARNOLD_MAP_N = 10


def psnr(A, B):
    mse = np.mean((A - B) ** 2)
    if (mse == 0):
        return 100
    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr


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
    stringConcatedBit = stringAuthenticationBit + stringRecoveryBit
    arrayPos = list(range(len(stringConcatedBit)))
    random.seed(salt)
    random.shuffle(arrayPos)
    res = ""
    for i in arrayPos:
        res += stringConcatedBit[i]
    return res


def getAuthenticationBit(watermarkBit: str, salt: int):
    arrayPos = list(range(len(watermarkBit)))
    random.seed(salt)
    random.shuffle(arrayPos)
    res = ""
    posMap = {}
    for i, val in enumerate(arrayPos):
        if val > 1:
            continue
        posMap[val] = i
    for i in sorted(posMap):
        res = res + watermarkBit[posMap[i]]
    return int(res, base=2)


def getRecoveryBit(watermarkBit: str, salt: int):
    arrayPos = list(range(len(watermarkBit)))
    random.seed(salt)
    random.shuffle(arrayPos)
    res = ""
    posMap = {}
    for i, val in enumerate(arrayPos):
        if val < 2:
            continue
        posMap[val] = i
    for i in sorted(posMap):
        res = res + watermarkBit[posMap[i]]
    return int(res, base=2)


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


def embedWatermarkPerBlock(data, watermarkData):
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


def getWatermarkDataPerBlock(data: np.ndarray):
    res = ""
    for y, _ in enumerate(data):
        for x, _ in enumerate(data[y]):
            watermarkData = data[y, x] % 4
            stringWatermarkData = bin(watermarkData)[2:]
            stringWatermarkData = "00" + stringWatermarkData
            stringWatermarkData = stringWatermarkData[-2:]
            res = res + stringWatermarkData
    return res


def embedWatermark(img):
    subBlock = createSubBlock(img, 2)
    size = (subBlock.shape[0], subBlock.shape[1])

    res = np.zeros(subBlock.shape, dtype=np.uint8)
    for y, _ in enumerate(subBlock):
        for x, _ in enumerate(subBlock[y]):
            tmpmap = arnoldMap(x, y, size[1], size[0], ARNOLD_MAP_N)
            salt = tmpmap[0] + tmpmap[1]
            recoveryBits = calculateRecoveryBit(
                subBlock[tmpmap[1], tmpmap[0]])
            authenticationBits = calculateAuthenticationBit(
                subBlock[y, x], salt)
            watermarkData = calculateWatermarkData(
                authenticationBits, recoveryBits, salt)
            res[y, x] = embedWatermarkPerBlock(subBlock[y, x], watermarkData)
    return mergeSubBlock(res)


def doRestore(data: np.ndarray, recoverData:  np.ndarray, salt: int):
    watermarkData = getWatermarkDataPerBlock(recoverData)
    recoveryBit = getRecoveryBit(watermarkData, salt)
    recoveryBit = recoveryBit << 2
    data = data % 4
    data = data + recoveryBit
    return data


def extractWatermarkAndRestore(img):
    subBlock = createSubBlock(img, 2)
    size = (subBlock.shape[0], subBlock.shape[1])
    watermarkRes = np.zeros(size, dtype=bool)
    imgRes = np.zeros(subBlock.shape, dtype=np.uint8)
    tamperZone = np.zeros(subBlock.shape, dtype=np.uint8)
    tamperCoincidence = 0
    for y, _ in enumerate(subBlock):
        for x, _ in enumerate(subBlock[y]):
            tmpmap = arnoldMap(x, y, size[1], size[0], ARNOLD_MAP_N)
            salt = tmpmap[0] + tmpmap[1]
            authenticationBits = calculateAuthenticationBit(
                subBlock[y, x], salt)
            watermarkData = getWatermarkDataPerBlock(subBlock[y, x])
            extractedAuthenticationBits = getAuthenticationBit(
                watermarkData, salt)
            result = authenticationBits == extractedAuthenticationBits
            imgRes[y, x] = subBlock[y, x]
            # if result == False:
            #     tmpmap = reverseArnoldMap(x, y, size[1], size[0], ARNOLD_MAP_N)
            #     salt = x + y
            #     imgRes[y, x] = doRestore(
            #         subBlock[y, x], subBlock[tmpmap[1], tmpmap[0]], salt)
            watermarkRes[y, x] = result
    for y, _ in enumerate(watermarkRes):
        for x, _ in enumerate(watermarkRes[y]):
            if watermarkRes[y, x] == False:
                tamperZone[y, x] = tamperZone[y, x] + 255
                tmpmap = reverseArnoldMap(x, y, size[1], size[0], ARNOLD_MAP_N)
                if watermarkRes[tmpmap[0], tmpmap[1]] == False:
                    tamperCoincidence = tamperCoincidence + 1
                salt = x + y
                imgRes[y, x] = doRestore(
                    subBlock[y, x], subBlock[tmpmap[1], tmpmap[0]], salt)
    print(tamperCoincidence)
    print(np.unique(watermarkRes, return_counts=True))
    return watermarkRes, mergeSubBlock(imgRes), mergeSubBlock(tamperZone)


if __name__ == "__main__":
    # originalImage = readImage("test1-marked.png")
    # watermarkedImage = embedWatermark(originalImage)
    # Image.fromarray(watermarkedImage).save("test1-marked-v2-embedded.png")
    # watermarkedImage = readImage("test1-marked-v2-embedded.png")
    # print("nilai PSNR: " + str(psnr(originalImage, watermarkedImage)))
    # authRes, imgRes = extractWatermarkAndRestore(watermarkedImage)
    attackedImage = readImage("test1-marked-v2-embedded-attacked.png")
    authRes, attackedImageRestored, tamperZone = extractWatermarkAndRestore(attackedImage)
    Image.fromarray(tamperZone).show()
    # Image.fromarray(attackedImageRestored).show()
    Image.fromarray(attackedImageRestored).save("test1-marked-v2-restored.png")
