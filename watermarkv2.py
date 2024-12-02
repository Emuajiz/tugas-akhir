from PIL import Image, ImageDraw
import numpy as np
import random
import math
import time
from skimage.metrics import structural_similarity as ssim

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
            if result == False:
                tamperZone[y, x] = tamperZone[y, x] + 255
                tmpmap = reverseArnoldMap(x, y, size[1], size[0], ARNOLD_MAP_N)
                salt = x + y
                imgRes[y, x] = doRestore(
                    subBlock[y, x], subBlock[tmpmap[1], tmpmap[0]], salt)
            watermarkRes[y, x] = result
    return watermarkRes, mergeSubBlock(imgRes), mergeSubBlock(tamperZone)


def detectionRate(img, originalImg):
    subBlock = createSubBlock(img, 2)
    originalSubBlock = createSubBlock(originalImg, 2)
    size = (subBlock.shape[0], subBlock.shape[1])
    watermarkRes = np.zeros(size, dtype=bool)
    for y, _ in enumerate(subBlock):
        for x, _ in enumerate(subBlock[y]):
            tmpmap = arnoldMap(x, y, size[1], size[0], ARNOLD_MAP_N)
            salt = tmpmap[0] + tmpmap[1]
            authenticationBits = calculateAuthenticationBit(
                originalSubBlock[y, x], salt)
            watermarkData = getWatermarkDataPerBlock(subBlock[y, x])
            extractedAuthenticationBits = getAuthenticationBit(
                watermarkData, salt)
            result = authenticationBits == extractedAuthenticationBits
            watermarkRes[y, x] = result
    return watermarkRes


def copyPasteAttack(img, size, position, targetPosition):
    copy = np.zeros(size, dtype=np.uint8)
    copy = img[position[0]:position[0]+size[0],
               position[1]:position[1]+size[1]]
    img[targetPosition[0]:targetPosition[0]+size[0],
        targetPosition[1]:targetPosition[1]+size[1]] = copy
    return img


def removeAttack(img, size, position):
    img[position[0]:position[0]+size[0], position[1]:position[1]+size[1]] = 0
    return img


def whiteNoiseAttack(img, size, position):
    img[position[0]:position[0]+size[0], position[1]:position[1]+size[1]] = np.random.randint(0, 256, size)
    return img


def addTextAttack(img, text, position, font_size):
    img = Image.fromarray(img)
    I1 = ImageDraw.Draw(img)
    I1.text(position, text, fill=255, stroke_fill=0,
            stroke_width=1, font_size=font_size)
    return np.array(img).astype(dtype=np.uint8)

def squareImage(img, size, position):
    img[position[0]:position[0]+size[0], position[1]:position[1]+size[1]] = 0
    return img


def processImage(imgName):
    originalImage = readImage("image/original/" + imgName)
    watermarkedImage = embedWatermark(originalImage)
    Image.fromarray(watermarkedImage).save("image/embedded/" + imgName)


def preprocessImage(imgName):
    originalImage = readImage(imgName + ".jpeg")
    maxSide = np.max(originalImage.shape)
    if maxSide % 2 == 1:
        maxSide = maxSide + 1
    img = np.zeros((maxSide, maxSide), dtype=np.uint8)

    # Calculate the starting x and y coordinates
    start_y = (img.shape[0] - originalImage.shape[0]) // 2
    start_x = (img.shape[1] - originalImage.shape[1]) // 2
    # Place the originalImage in the center of img
    img[start_y:start_y + originalImage.shape[0],
        start_x:start_x + originalImage.shape[1]] = originalImage

    Image.fromarray(img).save(imgName + ".png")


if __name__ == "__main__":
    imgNames = ["test1.png", "test2.png",
                "test3.png", "test4.png", "test5.png"]
    # for imgName in imgNames:
    #     print("Processing " + imgName)
    #     startTime = time.time_ns()
    #     processImage(imgName)
    #     print("time elapse: " + str(time.time_ns() - startTime))

    # for imgName in imgNames:
    #     print("Processing " + imgName)
    #     originalImage = readImage("image/original/" + imgName)
    #     watermarkedImage = readImage("image/embedded/" + imgName)
    #     print("nilai PSNR: " + str(psnr(originalImage, watermarkedImage)))

    # watermark extract without attack
    # for imgName in imgNames:
    #     print("Processing " + imgName)
    #     watermarkedImage = readImage("image/embedded/" + imgName)
    #     originalImage = readImage("image/original/" + imgName)
    #     authRes, imgRes, tamperZone = extractWatermarkAndRestore(watermarkedImage)
    #     detectionRateRes = detectionRate(watermarkedImage, originalImage)
    #     print("Hasil pengecekan: " + str(np.all(authRes)))
    #     print("Detection rate: " + str(np.all(detectionRateRes)))
    #     Image.fromarray(tamperZone).save("image/tamper-zone/not-attacked/" + imgName)

    # copy paste attack 5% of image
    # for imgName in imgNames:
    #     print("Processing " + imgName)
    #     watermarkedImage = readImage("image/embedded/" + imgName)
    #     size = (watermarkedImage.shape[0] * 5 // 100, watermarkedImage.shape[1] * 5 // 100)
    #     pos = (watermarkedImage.shape[0] // 2 - size[0], watermarkedImage.shape[1] // 2 - size[1])
    #     targetPos = (watermarkedImage.shape[0] // 2, watermarkedImage.shape[1] // 2)
    #     attackedImage = copyPasteAttack(watermarkedImage, size, pos, targetPos)
    #     Image.fromarray(attackedImage).save("image/attacked/copy-paste-5/" + imgName)

    # copy paste attack 10% of image
    # for imgName in imgNames:
    #     print("Processing " + imgName)
    #     watermarkedImage = readImage("image/embedded/" + imgName)
    #     size = (watermarkedImage.shape[0] * 10 // 100, watermarkedImage.shape[1] * 10 // 100)
    #     pos = (watermarkedImage.shape[0] // 2 - size[0], watermarkedImage.shape[1] // 2 - size[1])
    #     targetPos = (watermarkedImage.shape[0] // 2, watermarkedImage.shape[1] // 2)
    #     attackedImage = copyPasteAttack(watermarkedImage, size, pos, targetPos)
    #     Image.fromarray(attackedImage).save("image/attacked/copy-paste-10/" + imgName)

    # copy paste attack 20% of image
    # for imgName in imgNames:
    #     print("Processing " + imgName)
    #     watermarkedImage = readImage("image/embedded/" + imgName)
    #     size = (watermarkedImage.shape[0] * 20 // 100, watermarkedImage.shape[1] * 20 // 100)
    #     pos = (watermarkedImage.shape[0] // 2 - size[0], watermarkedImage.shape[1] // 2 - size[1])
    #     targetPos = (watermarkedImage.shape[0] // 2, watermarkedImage.shape[1] // 2)
    #     attackedImage = copyPasteAttack(watermarkedImage, size, pos, targetPos)
    #     Image.fromarray(attackedImage).save("image/attacked/copy-paste-20/" + imgName)

    # copy paste attack 50% of image
    # for imgName in imgNames:
    #     print("Processing " + imgName)
    #     watermarkedImage = readImage("image/embedded/" + imgName)
    #     size = (watermarkedImage.shape[0] * 50 // 100, watermarkedImage.shape[1] * 50 // 100)
    #     pos = (watermarkedImage.shape[0] // 2 - size[0], watermarkedImage.shape[1] // 2 - size[1])
    #     targetPos = (watermarkedImage.shape[0] // 2, watermarkedImage.shape[1] // 2)
    #     attackedImage = copyPasteAttack(watermarkedImage, size, pos, targetPos)
    #     Image.fromarray(attackedImage).save("image/attacked/copy-paste-50/" + imgName)

    # remove attack 5% of image
    # for imgName in imgNames:
    #     print("Processing " + imgName)
    #     watermarkedImage = readImage("image/embedded/" + imgName)
    #     size = (watermarkedImage.shape[0] * 5 // 100, watermarkedImage.shape[1] * 5 // 100)
    #     pos = (watermarkedImage.shape[0] // 2 - size[0], watermarkedImage.shape[1] // 2 - size[1])
    #     targetPos = (watermarkedImage.shape[0] // 2, watermarkedImage.shape[1] // 2)
    #     attackedImage = removeAttack(watermarkedImage, size, targetPos)
    #     Image.fromarray(attackedImage).save("image/attacked/remove-5/" + imgName)

    # remove attack 10% of image
    # for imgName in imgNames:
    #     print("Processing " + imgName)
    #     watermarkedImage = readImage("image/embedded/" + imgName)
    #     size = (watermarkedImage.shape[0] * 10 // 100, watermarkedImage.shape[1] * 10 // 100)
    #     pos = (watermarkedImage.shape[0] // 2 - size[0], watermarkedImage.shape[1] // 2 - size[1])
    #     targetPos = (watermarkedImage.shape[0] // 2, watermarkedImage.shape[1] // 2)
    #     attackedImage = removeAttack(watermarkedImage, size, targetPos)
    #     Image.fromarray(attackedImage).save("image/attacked/remove-10/" + imgName)

    # remove attack 20% of image
    # for imgName in imgNames:
    #     print("Processing " + imgName)
    #     watermarkedImage = readImage("image/embedded/" + imgName)
    #     size = (watermarkedImage.shape[0] * 20 // 100, watermarkedImage.shape[1] * 20 // 100)
    #     pos = (watermarkedImage.shape[0] // 2 - size[0], watermarkedImage.shape[1] // 2 - size[1])
    #     targetPos = (watermarkedImage.shape[0] // 2, watermarkedImage.shape[1] // 2)
    #     attackedImage = removeAttack(watermarkedImage, size, targetPos)
    #     Image.fromarray(attackedImage).save("image/attacked/remove-20/" + imgName)

    # remove attack 50% of image
    # for imgName in imgNames:
    #     print("Processing " + imgName)
    #     watermarkedImage = readImage("image/embedded/" + imgName)
    #     size = (watermarkedImage.shape[0] * 50 // 100, watermarkedImage.shape[1] * 50 // 100)
    #     pos = (watermarkedImage.shape[0] // 2 - size[0], watermarkedImage.shape[1] // 2 - size[1])
    #     targetPos = (watermarkedImage.shape[0] // 2, watermarkedImage.shape[1] // 2)
    #     attackedImage = removeAttack(watermarkedImage, size, targetPos)
    #     Image.fromarray(attackedImage).save("image/attacked/remove-50/" + imgName)

    # white noise attack 5% of image
    # for imgName in imgNames:
    #     print("Processing " + imgName)
    #     watermarkedImage = readImage("image/embedded/" + imgName)
    #     size = (watermarkedImage.shape[0] * 5 // 100, watermarkedImage.shape[1] * 5 // 100)
    #     pos = (watermarkedImage.shape[0] // 2 - size[0], watermarkedImage.shape[1] // 2 - size[1])
    #     attackedImage = whiteNoiseAttack(watermarkedImage, size, pos)
    #     Image.fromarray(attackedImage).save("image/attacked/white-noise-5/" + imgName)

    # white noise attack 10% of image
    # for imgName in imgNames:
    #     print("Processing " + imgName)
    #     watermarkedImage = readImage("image/embedded/" + imgName)
    #     size = (watermarkedImage.shape[0] * 10 // 100, watermarkedImage.shape[1] * 10 // 100)
    #     pos = (watermarkedImage.shape[0] // 2 - size[0], watermarkedImage.shape[1] // 2 - size[1])
    #     attackedImage = whiteNoiseAttack(watermarkedImage, size, pos)
    #     Image.fromarray(attackedImage).save("image/attacked/white-noise-10/" + imgName)

    # white noise attack 20% of image
    # for imgName in imgNames:
    #     print("Processing " + imgName)
    #     watermarkedImage = readImage("image/embedded/" + imgName)
    #     size = (watermarkedImage.shape[0] * 20 // 100, watermarkedImage.shape[1] * 20 // 100)
    #     pos = (watermarkedImage.shape[0] // 2 - size[0], watermarkedImage.shape[1] // 2 - size[1])
    #     attackedImage = whiteNoiseAttack(watermarkedImage, size, pos)
    #     Image.fromarray(attackedImage).save("image/attacked/white-noise-20/" + imgName)

    # white noise attack 50% of image
    # for imgName in imgNames:
    #     print("Processing " + imgName)
    #     watermarkedImage = readImage("image/embedded/" + imgName)
    #     size = (watermarkedImage.shape[0] * 50 // 100, watermarkedImage.shape[1] * 50 // 100)
    #     pos = (watermarkedImage.shape[0] // 2 - size[0], watermarkedImage.shape[1] // 2 - size[1])
    #     attackedImage = whiteNoiseAttack(watermarkedImage, size, pos)
    #     Image.fromarray(attackedImage).save("image/attacked/white-noise-50/" + imgName)

    # add text attack
    # for imgName in imgNames:
    #     print("Processing " + imgName)
    #     watermarkedImage = readImage("image/embedded/" + imgName)
    #     font_size = np.min(watermarkedImage.shape) // 10
    #     attackedImage = addTextAttack(watermarkedImage, "Attacked", (
    #         watermarkedImage.shape[1] // 2, watermarkedImage.shape[0] // 2), font_size)
    #     Image.fromarray(attackedImage).save(
    #         "image/attacked/add-text/" + imgName)

    # watermark extract with copy paste attack 5% of image
    # for imgName in imgNames:
    #     print("Processing " + imgName)
    #     attackedImage = readImage("image/attacked/copy-paste-5/" + imgName)
    #     authRes, imgRes, tamperZone = extractWatermarkAndRestore(attackedImage)
    #     print("Hasil pengecekan: " + str(np.all(authRes)))
    #     Image.fromarray(tamperZone).save(
    #         "image/tamper-zone/copy-paste-5/" + imgName)
    #     Image.fromarray(imgRes).save("image/restored/copy-paste-5/" + imgName)

    # watermark extract with copy paste attack 10% of image
    # for imgName in imgNames:
    #     print("Processing " + imgName)
    #     attackedImage = readImage("image/attacked/copy-paste-10/" + imgName)
    #     authRes, imgRes, tamperZone = extractWatermarkAndRestore(attackedImage)
    #     print("Hasil pengecekan: " + str(np.all(authRes)))
    #     Image.fromarray(tamperZone).save(
    #         "image/tamper-zone/copy-paste-10/" + imgName)
    #     Image.fromarray(imgRes).save("image/restored/copy-paste-10/" + imgName)

    # watermark extract with copy paste attack 20% of image
    # for imgName in imgNames:
    #     print("Processing " + imgName)
    #     attackedImage = readImage("image/attacked/copy-paste-20/" + imgName)
    #     authRes, imgRes, tamperZone = extractWatermarkAndRestore(attackedImage)
    #     print("Hasil pengecekan: " + str(np.all(authRes)))
    #     Image.fromarray(tamperZone).save(
    #         "image/tamper-zone/copy-paste-20/" + imgName)
    #     Image.fromarray(imgRes).save("image/restored/copy-paste-20/" + imgName)

    # watermark extract with copy paste attack 50% of image
    # for imgName in imgNames:
    #     print("Processing " + imgName)
    #     attackedImage = readImage("image/attacked/copy-paste-50/" + imgName)
    #     authRes, imgRes, tamperZone = extractWatermarkAndRestore(attackedImage)
    #     print("Hasil pengecekan: " + str(np.all(authRes)))
    #     Image.fromarray(tamperZone).save(
    #         "image/tamper-zone/copy-paste-50/" + imgName)
    #     Image.fromarray(imgRes).save("image/restored/copy-paste-50/" + imgName)

    # watermark extract with remove attack 5% of image
    # for imgName in imgNames:
    #     print("Processing " + imgName)
    #     attackedImage = readImage("image/attacked/remove-5/" + imgName)
    #     authRes, imgRes, tamperZone = extractWatermarkAndRestore(attackedImage)
    #     print("Hasil pengecekan: " + str(np.all(authRes)))
    #     Image.fromarray(tamperZone).save(
    #         "image/tamper-zone/remove-5/" + imgName)
    #     Image.fromarray(imgRes).save("image/restored/remove-5/" + imgName)

    # watermark extract with remove attack 10% of image
    # for imgName in imgNames:
    #     print("Processing " + imgName)
    #     attackedImage = readImage("image/attacked/remove-10/" + imgName)
    #     authRes, imgRes, tamperZone = extractWatermarkAndRestore(attackedImage)
    #     print("Hasil pengecekan: " + str(np.all(authRes)))
    #     Image.fromarray(tamperZone).save(
    #         "image/tamper-zone/remove-10/" + imgName)
    #     Image.fromarray(imgRes).save("image/restored/remove-10/" + imgName)

    # watermark extract with remove attack 20% of image
    # for imgName in imgNames:
    #     print("Processing " + imgName)
    #     attackedImage = readImage("image/attacked/remove-20/" + imgName)
    #     authRes, imgRes, tamperZone = extractWatermarkAndRestore(attackedImage)
    #     print("Hasil pengecekan: " + str(np.all(authRes)))
    #     Image.fromarray(tamperZone).save(
    #         "image/tamper-zone/remove-20/" + imgName)
    #     Image.fromarray(imgRes).save("image/restored/remove-20/" + imgName)

    # watermark extract with remove attack 50% of image
    # for imgName in imgNames:
    #     print("Processing " + imgName)
    #     attackedImage = readImage("image/attacked/remove-50/" + imgName)
    #     authRes, imgRes, tamperZone = extractWatermarkAndRestore(attackedImage)
    #     print("Hasil pengecekan: " + str(np.all(authRes)))
    #     Image.fromarray(tamperZone).save(
    #         "image/tamper-zone/remove-50/" + imgName)
    #     Image.fromarray(imgRes).save("image/restored/remove-50/" + imgName)

    # watermark extract with white noise attack 5% of image
    # for imgName in imgNames:
    #     print("Processing " + imgName)
    #     attackedImage = readImage("image/attacked/white-noise-5/" + imgName)
    #     authRes, imgRes, tamperZone = extractWatermarkAndRestore(attackedImage)
    #     print("Hasil pengecekan: " + str(np.all(authRes)))
    #     Image.fromarray(tamperZone).save(
    #         "image/tamper-zone/white-noise-5/" + imgName)
    #     Image.fromarray(imgRes).save("image/restored/white-noise-5/" + imgName)

    # watermark extract with white noise attack 10% of image
    # for imgName in imgNames:
    #     print("Processing " + imgName)
    #     attackedImage = readImage("image/attacked/white-noise-10/" + imgName)
    #     authRes, imgRes, tamperZone = extractWatermarkAndRestore(attackedImage)
    #     print("Hasil pengecekan: " + str(np.all(authRes)))
    #     Image.fromarray(tamperZone).save(
    #         "image/tamper-zone/white-noise-10/" + imgName)
    #     Image.fromarray(imgRes).save("image/restored/white-noise-10/" + imgName)

    # watermark extract with white noise attack 20% of image
    # for imgName in imgNames:
    #     print("Processing " + imgName)
    #     attackedImage = readImage("image/attacked/white-noise-20/" + imgName)
    #     authRes, imgRes, tamperZone = extractWatermarkAndRestore(attackedImage)
    #     print("Hasil pengecekan: " + str(np.all(authRes)))
    #     Image.fromarray(tamperZone).save(
    #         "image/tamper-zone/white-noise-20/" + imgName)
    #     Image.fromarray(imgRes).save("image/restored/white-noise-20/" + imgName)

    # watermark extract with white noise attack 50% of image
    # for imgName in imgNames:
    #     print("Processing " + imgName)
    #     attackedImage = readImage("image/attacked/white-noise-50/" + imgName)
    #     authRes, imgRes, tamperZone = extractWatermarkAndRestore(attackedImage)
    #     print("Hasil pengecekan: " + str(np.all(authRes)))
    #     Image.fromarray(tamperZone).save(
    #         "image/tamper-zone/white-noise-50/" + imgName)
    #     Image.fromarray(imgRes).save("image/restored/white-noise-50/" + imgName)

    # watermark extract with add text attack
    # for imgName in imgNames:
    #     print("Processing " + imgName)
    #     attackedImage = readImage("image/attacked/add-text/" + imgName)
    #     authRes, imgRes, tamperZone = extractWatermarkAndRestore(attackedImage)
    #     print("Hasil pengecekan: " + str(np.all(authRes)))
    #     Image.fromarray(tamperZone).save(
    #         "image/tamper-zone/add-text/" + imgName)
    #     Image.fromarray(imgRes).save("image/restored/add-text/" + imgName)

    # psnr and ssim calculation for copy paste attack 5% of image
    # for imgName in imgNames:
    #     print("Processing " + imgName)
    #     originalImage = readImage("image/original/" + imgName)
    #     attackedImage = readImage("image/attacked/copy-paste-5/" + imgName)
    #     restoredImage = readImage("image/restored/copy-paste-5/" + imgName)
    #     print("nilai PSNR: " + str(psnr(originalImage, attackedImage)))
    #     similarity = ssim(originalImage, attackedImage, multichannel=True)
    #     print("nilai SSIM: " + str(similarity))
    #     print("nilai PSNR restored: " + str(psnr(originalImage, restoredImage)))
    #     similarity = ssim(originalImage, restoredImage, multichannel=True)
    #     print("nilai SSIM restored: " + str(similarity))

    # psnr and ssim calculation for copy paste attack 10% of image
    # for imgName in imgNames:
    #     print("Processing " + imgName)
    #     originalImage = readImage("image/original/" + imgName)
    #     attackedImage = readImage("image/attacked/copy-paste-10/" + imgName)
    #     restoredImage = readImage("image/restored/copy-paste-10/" + imgName)
    #     print("nilai PSNR: " + str(psnr(originalImage, attackedImage)))
    #     similarity = ssim(originalImage, attackedImage, multichannel=True)
    #     print("nilai SSIM: " + str(similarity))
    #     print("nilai PSNR restored: " + str(psnr(originalImage, restoredImage)))
    #     similarity = ssim(originalImage, restoredImage, multichannel=True)
    #     print("nilai SSIM restored: " + str(similarity))

    # psnr and ssim calculation for copy paste attack 20% of image
    # for imgName in imgNames:
    #     print("Processing " + imgName)
    #     originalImage = readImage("image/original/" + imgName)
    #     attackedImage = readImage("image/attacked/copy-paste-20/" + imgName)
    #     restoredImage = readImage("image/restored/copy-paste-20/" + imgName)
    #     print("nilai PSNR: " + str(psnr(originalImage, attackedImage)))
    #     similarity = ssim(originalImage, attackedImage, multichannel=True)
    #     print("nilai SSIM: " + str(similarity))
    #     print("nilai PSNR restored: " + str(psnr(originalImage, restoredImage)))
    #     similarity = ssim(originalImage, restoredImage, multichannel=True)
    #     print("nilai SSIM restored: " + str(similarity))

    # psnr and ssim calculation for copy paste attack 50% of image
    # for imgName in imgNames:
    #     print("Processing " + imgName)
    #     originalImage = readImage("image/original/" + imgName)
    #     attackedImage = readImage("image/attacked/copy-paste-50/" + imgName)
    #     restoredImage = readImage("image/restored/copy-paste-50/" + imgName)
    #     print("nilai PSNR: " + str(psnr(originalImage, attackedImage)))
    #     similarity = ssim(originalImage, attackedImage, multichannel=True)
    #     print("nilai SSIM: " + str(similarity))
    #     print("nilai PSNR restored: " + str(psnr(originalImage, restoredImage)))
    #     similarity = ssim(originalImage, restoredImage, multichannel=True)
    #     print("nilai SSIM restored: " + str(similarity))

    # psnr and ssim calculation for remove attack 5% of image
    # for imgName in imgNames:
    #     print("Processing " + imgName)
    #     originalImage = readImage("image/original/" + imgName)
    #     attackedImage = readImage("image/attacked/remove-5/" + imgName)
    #     restoredImage = readImage("image/restored/remove-5/" + imgName)
    #     print("nilai PSNR: " + str(psnr(originalImage, attackedImage)))
    #     similarity = ssim(originalImage, attackedImage, multichannel=True)
    #     print("nilai SSIM: " + str(similarity))
    #     print("nilai PSNR restored: " + str(psnr(originalImage, restoredImage)))
    #     similarity = ssim(originalImage, restoredImage, multichannel=True)
    #     print("nilai SSIM restored: " + str(similarity))

    # psnr and ssim calculation for remove attack 10% of image
    # for imgName in imgNames:
    #     print("Processing " + imgName)
    #     originalImage = readImage("image/original/" + imgName)
    #     attackedImage = readImage("image/attacked/remove-10/" + imgName)
    #     restoredImage = readImage("image/restored/remove-10/" + imgName)
    #     print("nilai PSNR: " + str(psnr(originalImage, attackedImage)))
    #     similarity = ssim(originalImage, attackedImage, multichannel=True)
    #     print("nilai SSIM: " + str(similarity))
    #     print("nilai PSNR restored: " + str(psnr(originalImage, restoredImage)))
    #     similarity = ssim(originalImage, restoredImage, multichannel=True)
    #     print("nilai SSIM restored: " + str(similarity))

    # psnr and ssim calculation for remove attack 20% of image
    # for imgName in imgNames:
    #     print("Processing " + imgName)
    #     originalImage = readImage("image/original/" + imgName)
    #     attackedImage = readImage("image/attacked/remove-20/" + imgName)
    #     restoredImage = readImage("image/restored/remove-20/" + imgName)
    #     print("nilai PSNR: " + str(psnr(originalImage, attackedImage)))
    #     similarity = ssim(originalImage, attackedImage, multichannel=True)
    #     print("nilai SSIM: " + str(similarity))
    #     print("nilai PSNR restored: " + str(psnr(originalImage, restoredImage)))
    #     similarity = ssim(originalImage, restoredImage, multichannel=True)
    #     print("nilai SSIM restored: " + str(similarity))

    # psnr and ssim calculation for remove attack 50% of image
    # for imgName in imgNames:
    #     print("Processing " + imgName)
    #     originalImage = readImage("image/original/" + imgName)
    #     attackedImage = readImage("image/attacked/remove-50/" + imgName)
    #     restoredImage = readImage("image/restored/remove-50/" + imgName)
    #     print("nilai PSNR: " + str(psnr(originalImage, attackedImage)))
    #     similarity = ssim(originalImage, attackedImage, multichannel=True)
    #     print("nilai SSIM: " + str(similarity))
    #     print("nilai PSNR restored: " + str(psnr(originalImage, restoredImage)))
    #     similarity = ssim(originalImage, restoredImage, multichannel=True)
    #     print("nilai SSIM restored: " + str(similarity))

    # psnr and ssim calculation for white noise attack 5% of image
    # for imgName in imgNames:
    #     print("Processing " + imgName)
    #     originalImage = readImage("image/original/" + imgName)
    #     attackedImage = readImage("image/attacked/white-noise-5/" + imgName)
    #     restoredImage = readImage("image/restored/white-noise-5/" + imgName)
    #     print("nilai PSNR: " + str(psnr(originalImage, attackedImage)))
    #     similarity = ssim(originalImage, attackedImage, multichannel=True)
    #     print("nilai SSIM: " + str(similarity))
    #     print("nilai PSNR restored: " + str(psnr(originalImage, restoredImage)))
    #     similarity = ssim(originalImage, restoredImage, multichannel=True)
    #     print("nilai SSIM restored: " + str(similarity))

    # psnr and ssim calculation for white noise attack 10% of image
    # for imgName in imgNames:
    #     print("Processing " + imgName)
    #     originalImage = readImage("image/original/" + imgName)
    #     attackedImage = readImage("image/attacked/white-noise-10/" + imgName)
    #     restoredImage = readImage("image/restored/white-noise-10/" + imgName)
    #     print("nilai PSNR: " + str(psnr(originalImage, attackedImage)))
    #     similarity = ssim(originalImage, attackedImage, multichannel=True)
    #     print("nilai SSIM: " + str(similarity))
    #     print("nilai PSNR restored: " + str(psnr(originalImage, restoredImage)))
    #     similarity = ssim(originalImage, restoredImage, multichannel=True)
    #     print("nilai SSIM restored: " + str(similarity))

    # psnr and ssim calculation for white noise attack 20% of image
    # for imgName in imgNames:
    #     print("Processing " + imgName)
    #     originalImage = readImage("image/original/" + imgName)
    #     attackedImage = readImage("image/attacked/white-noise-20/" + imgName)
    #     restoredImage = readImage("image/restored/white-noise-20/" + imgName)
    #     print("nilai PSNR: " + str(psnr(originalImage, attackedImage)))
    #     similarity = ssim(originalImage, attackedImage, multichannel=True)
    #     print("nilai SSIM: " + str(similarity))
    #     print("nilai PSNR restored: " + str(psnr(originalImage, restoredImage)))
    #     similarity = ssim(originalImage, restoredImage, multichannel=True)
    #     print("nilai SSIM restored: " + str(similarity))

    # psnr and ssim calculation for white noise attack 50% of image
    # for imgName in imgNames:
    #     print("Processing " + imgName)
    #     originalImage = readImage("image/original/" + imgName)
    #     attackedImage = readImage("image/attacked/white-noise-50/" + imgName)
    #     restoredImage = readImage("image/restored/white-noise-50/" + imgName)
    #     print("nilai PSNR: " + str(psnr(originalImage, attackedImage)))
    #     similarity = ssim(originalImage, attackedImage, multichannel=True)
    #     print("nilai SSIM: " + str(similarity))
    #     print("nilai PSNR restored: " + str(psnr(originalImage, restoredImage)))
    #     similarity = ssim(originalImage, restoredImage, multichannel=True)
    #     print("nilai SSIM restored: " + str(similarity))

    # PSNR and SSIM calculation for add text attack
    # for imgName in imgNames:
    #     print("Processing " + imgName)
    #     originalImage = readImage("image/original/" + imgName)
    #     attackedImage = readImage("image/attacked/add-text/" + imgName)
    #     restoredImage = readImage("image/restored/add-text/" + imgName)
    #     print("nilai PSNR: " + str(psnr(originalImage, attackedImage)))
    #     similarity = ssim(originalImage, attackedImage, multichannel=True)
    #     print("nilai SSIM: " + str(similarity))
    #     print("nilai PSNR restored: " + str(psnr(originalImage, restoredImage)))
    #     similarity = ssim(originalImage, restoredImage, multichannel=True)
    #     print("nilai SSIM restored: " + str(similarity))

    # attackedImage = readImage("image/attacked/copy-paste-5/test1.png")
    # authRes, imgRes, tamperZone = extractWatermarkAndRestore(attackedImage)
    # Image.fromarray(tamperZone).show()
    # Image.fromarray(imgRes).show()

    # attackedImage = readImage("image/attacked/copy-paste-5/test2.png")
    # authRes, imgRes, tamperZone = extractWatermarkAndRestore(attackedImage)
    # Image.fromarray(tamperZone).show()
    # Image.fromarray(imgRes).show()

    # watermarkedImage = readImage("test1-marked-v2-embedded.png")
    # print("nilai PSNR: " + str(psnr(originalImage, watermarkedImage)))
    # similarity = ssim(originalImage, watermarkedImage, multichannel=True)
    # print(similarity)
    # authRes, imgRes = extractWatermarkAndRestore(watermarkedImage)
    # attackedImage = readImage("test1-marked-v2-embedded-attacked.png")
    # authRes, attackedImageRestored, tamperZone = extractWatermarkAndRestore(attackedImage)
    # Image.fromarray(tamperZone).show()
    # Image.fromarray(attackedImageRestored).show()
    # Image.fromarray(attackedIxmageRestored).save("test1-marked-v2-restored.png")
    # attacked = copyPasteAttack(watermarkedImage, (100, 100), (200, 200), (300, 300))
    # attacked = removeAttack(watermarkedImage, (100, 100), (300, 300))
    # Image.fromarray(attacked).save("test1-marked-v2-embedded-attacked.png")

    # jpg to png
    # preprocessImage(
    #     "image/original/monostotic-melorheostosis-and-glass-shard-2")
    # preprocessImage("image/original/monostotic-melorheostosis-and-glass-shard")
    # preprocessImage(
    #     "image/original/heterotopic-calcification-in-previous-rupture-of-the-radial-collateral-ligament-of-the-elbow-2")
    # preprocessImage(
    #     "image/original/heterotopic-calcification-in-previous-rupture-of-the-radial-collateral-ligament-of-the-elbow")
