import cmath
import hashlib
import math
import random
import string
from PIL import Image
import numpy as np
from skimage.util import random_noise

def stringToBit(text):
    binary = []
    for i in text:
        binary.append('{0:08b}'.format(ord(i)))
    return ''.join(binary)


def charToBit(char):
    assert len(char) == 1
    binary = np.zeros(8, dtype=np.int8)
    tmp = ord(char)
    counter = 0
    while (tmp):
        binary[counter] = tmp % 2
        tmp = tmp // 2
        counter += 1
    return np.flip(binary)


def psnr(A, B):
    mse = np.mean((A - B) ** 2)
    if (mse == 0):
        return 100
    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr


def readImage(filename):
    image = Image.open(filename)
    if image.mode != "L":
        image = image.convert("L")
    return np.array(image)


def splitImage(imageData, outsideWidth=32):
    imageSize = imageData.shape
    # ensure image size is square
    assert imageSize[0] == imageSize[1]
    insidePart = imageData[outsideWidth:imageSize[1] -
                           outsideWidth, outsideWidth:imageSize[0] - outsideWidth]

    leftside = []
    upside = []
    rightside = []
    bottomside = []
    for i in range(0, imageSize[1], outsideWidth):
        # upside
        imageTemp = np.array(imageData[0:outsideWidth, i:i+outsideWidth])
        upside.append(imageTemp)

        # bottomside
        imageTemp = np.array(
            imageData[imageSize[0] - outsideWidth:imageSize[0], imageSize[1] - i - outsideWidth:imageSize[1] - i])
        bottomside.append(imageTemp)

    for i in range(0, imageSize[0], outsideWidth):
        # rightside
        imageTemp = np.array(
            imageData[i:i+outsideWidth, imageSize[1] - outsideWidth:imageSize[1]])
        rightside.append(imageTemp)

        # leftside
        imageTemp = np.array(
            imageData[imageSize[0] - i - outsideWidth:imageSize[0] - i, 0:outsideWidth])
        leftside.append(imageTemp)

    # overlapping for check
    assert psnr(upside[0], leftside[-1]) == 100
    assert psnr(rightside[0], upside[-1]) == 100
    assert psnr(bottomside[0], rightside[-1]) == 100
    assert psnr(leftside[0], bottomside[-1]) == 100

    return insidePart, np.array([upside, rightside, bottomside, leftside])


def mergeImage(insidePart, outsidePart):
    assert psnr(outsidePart[0][0], outsidePart[3][-1]) == 100
    assert psnr(outsidePart[1][0], outsidePart[0][-1]) == 100
    assert psnr(outsidePart[2][0], outsidePart[1][-1]) == 100
    assert psnr(outsidePart[3][0], outsidePart[2][-1]) == 100
    imageSize = (insidePart.shape[0] + 2*outsidePart.shape[2],
                 insidePart.shape[1] + 2*outsidePart.shape[3])
    imageData = np.zeros(shape=imageSize, dtype=np.uint8)
    imageData[outsidePart.shape[2]:insidePart.shape[0] +
              outsidePart.shape[2], outsidePart.shape[3]:insidePart.shape[1] + outsidePart.shape[3]] = insidePart

    for i, val in enumerate(outsidePart[0]):
        # upside
        imageData[0:val.shape[0], i*val.shape[1]:(i+1)*val.shape[1]] = val
        # bottomside
        imageData[-val.shape[0]:, i*val.shape[1]                  :(i+1)*val.shape[1]] = outsidePart[2][-(i+1)]

    for i, val in enumerate(outsidePart[1]):
        # rightside
        imageData[i*val.shape[0]:(i+1)*val.shape[0], -val.shape[1]:] = val
        # leftside
        imageData[i*val.shape[0]:(i+1)*val.shape[0],
                  :val.shape[1]] = outsidePart[3][-(i+1)]
    mergedImage = Image.fromarray(imageData)
    mergedImage.save("merged.png")
    return imageData

# explicit function to normalize array


def normalize(arr, t_min, t_max):
    norm_arr = []
    diff = t_max - t_min
    diff_arr = max(arr) - min(arr)
    for i in arr:
        temp = (((i - min(arr))*diff)/diff_arr) + t_min
        norm_arr.append(temp)
    return norm_arr


def imageHash(imageData):
    h = hashlib.sha256()
    for i in imageData:
        for j in i:
            assert (j // 2) < 128
            h.update(j // 2)
    return h.digest()


def makeFragileWatermarkPayload(imageData):
    fragileWatermarkPayload = imageHash(imageData)
    fragileWatermarkPayloadBitString = []
    for i in fragileWatermarkPayload:
        fragileWatermarkPayloadBitString.append("{0:08b}".format(i))
    return ''.join(fragileWatermarkPayloadBitString)


def embedFragileWatermark(imageData):
    watermarkedImageData = np.zeros(imageData.shape, dtype=np.uint8)
    fragileWatermarkPayload = makeFragileWatermarkPayload(imageData)
    counter = 0
    for y, i in enumerate(imageData):
        for x, j in enumerate(i):
            p = j
            p &= 0b11111110
            p |= int(fragileWatermarkPayload[counter])
            watermarkedImageData[y, x] = p
            counter += 1
            counter = counter % len(fragileWatermarkPayload)
    return watermarkedImageData


def extractFragileWatermark(imageData):
    fragileWatermarkPayload = makeFragileWatermarkPayload(imageData)
    fragileWatermark = np.zeros(imageData.shape, dtype=np.uint8)
    counter = 0
    for y, i in enumerate(imageData):
        for x, j in enumerate(i):
            if (j % 2 == int(fragileWatermarkPayload[counter])):
                fragileWatermark[y, x] = 255
            else:
                fragileWatermark[y, x] = 0
            counter += 1
            counter %= len(fragileWatermarkPayload)
    # Image.fromarray(fragileWatermark).show()
    return fragileWatermark


def calculateWatermarkPosition(vectorLength, imageShape):
    center = (int(imageShape[0] / 2) + 1, int(imageShape[1] / 2) + 1)

    def x(t): return center[0] + int(min(imageShape) / 4 *
                                     math.cos(t * 2 * math.pi / vectorLength))

    def y(t): return center[1] + int(min(imageShape) / 4 *
                                     math.sin(t * 2 * math.pi / vectorLength))

    indices = [(x(t), y(t)) for t in range(vectorLength)]
    return indices


def calculateWatermarkPositionDouble(vectorLength, imageShape):
    center = (int(imageShape[0] / 2) + 1, int(imageShape[1] / 2) + 1)

    def x(t): return center[0] + int(min(imageShape) / 4 *
                                     math.cos(t * 2 * math.pi / vectorLength))

    def y(t): return center[1] + int(min(imageShape) / 4 *
                                     math.sin(t * 2 * math.pi / vectorLength))

    def x2(t): return center[0] + int(min(imageShape) / 3 *
                                      math.cos(t * 2 * math.pi / vectorLength))
    def y2(t): return center[1] + int(min(imageShape) / 3 *
                                      math.sin(t * 2 * math.pi / vectorLength))
    indices = [(x(t), y(t)) for t in range(vectorLength // 2)] + \
        [(x2(t), y2(t)) for t in range(vectorLength // 2)]
    return indices


def embedRobustWatermark(imageData, watermarkData, alpha=1):
    vectorLength = len(watermarkData)
    indices = calculateWatermarkPosition(vectorLength, imageData.shape)
    imageFourier = np.fft.fftshift(np.fft.fft2(imageData))
    mag = np.abs(imageFourier)
    phase = np.angle(imageFourier)
    for i, val in enumerate(indices):
        tmp = mag[val[0], val[1]] * (1 + alpha * int(watermarkData[i]))
        fourierTmp = tmp * cmath.exp(1j * phase[val[0], val[1]])
        imageFourier[val[0], val[1]] = fourierTmp
    watermarkedImageData = np.round(np.abs(np.fft.ifft2(
        np.fft.ifftshift(imageFourier)))).astype(np.uint8)
    return watermarkedImageData


def extractRobustWatermark(imageData, originalImageData, watermarkData, alpha=1):
    vectorLength = len(watermarkData)
    indices = calculateWatermarkPosition(vectorLength, imageData.shape)
    imageFourier = np.fft.fftshift(np.fft.fft2(imageData))
    originalImageFourier = np.fft.fftshift(np.fft.fft2(originalImageData))
    mag = np.abs(imageFourier)
    originalMag = np.abs(originalImageFourier)
    extractedWatermarkData = []
    for i, val in enumerate(indices):
        tmp = (mag[val[0], val[1]]/originalMag[val[0], val[1]] - 1) / alpha
        extractedWatermarkData.append(tmp)
    return extractedWatermarkData


def calculateOutsideWatermarkSize(imageData):
    size = 0
    for i in range(imageData.shape[0]):
        size += (imageData[i].shape[0] - 1)
    return size


def calculateWatermark(password, watermarkLength):
    random.seed(password + str(watermarkLength), version=2)
    res = ''.join(random.choices(string.ascii_letters +
                  string.digits, k=watermarkLength))
    return res


def processEmbedRobustWatermark(imageDataArray, password, embedFactor=10):
    watermarkSize = calculateOutsideWatermarkSize(imageDataArray)
    watermarkData = calculateWatermark(password, watermarkSize)
    # print(watermarkData)
    # print(len(watermarkData))
    counter = 0
    # upside
    upsideDataArray = np.zeros(imageDataArray[0].shape)
    # print(upsideDataArray.shape)
    for i in range(imageDataArray[0].shape[0]):
        # print(imageDataArray[0,i].shape)
        upsideDataArray[i] = embedRobustWatermark(
            imageDataArray[0, i], stringToBit(watermarkData[counter]), embedFactor)
        counter += 1
    counter -= 1
    # print(counter)

    # rightside
    rightsideDataArray = np.zeros(imageDataArray[1].shape)
    # print(rightsideDataArray.shape)
    for i in range(imageDataArray[1].shape[0]):
        # print(imageDataArray[0,i].shape)
        rightsideDataArray[i] = embedRobustWatermark(
            imageDataArray[1, i], stringToBit(watermarkData[counter]), embedFactor)
        counter += 1
    counter -= 1
    # print(counter)

    # bottomside
    bottomsideDataArray = np.zeros(imageDataArray[2].shape)
    # print(bottomsideDataArray.shape)
    for i in range(imageDataArray[2].shape[0]):
        # print(imageDataArray[0,i].shape)
        bottomsideDataArray[i] = embedRobustWatermark(
            imageDataArray[2, i], stringToBit(watermarkData[counter]), embedFactor)
        counter += 1
    counter -= 1
    # print(counter)

    # leftside
    leftsideDataArray = np.zeros(imageDataArray[3].shape)
    # print(leftsideDataArray.shape)
    for i in range(imageDataArray[3].shape[0]):
        # print(imageDataArray[0,i].shape)
        leftsideDataArray[i] = embedRobustWatermark(
            imageDataArray[3, i], stringToBit(watermarkData[counter % watermarkSize]), embedFactor)
        counter += 1
    counter -= 1
    # print(counter)

    assert psnr(upsideDataArray[0], leftsideDataArray[-1]) == 100
    assert psnr(rightsideDataArray[0], upsideDataArray[-1]) == 100
    assert psnr(bottomsideDataArray[0], rightsideDataArray[-1]) == 100
    assert psnr(leftsideDataArray[0], bottomsideDataArray[-1]) == 100

    return np.array([upsideDataArray, rightsideDataArray, bottomsideDataArray, leftsideDataArray])


def checkBitRobustWatermark(extracted, original, threshold=0.5):
    norm_extracted = normalize(extracted, 0, 1)
    bit_extracted = np.where(np.array(norm_extracted) > threshold, 1, 0)
    tmp = charToBit(original)
    for i in range(8):
        if bit_extracted[i] != tmp[i]:
            return False
    return True


def processExtractRobustWatermark(imageDataArray, originalImageDataArray, password, embedFactor=10):
    assert imageDataArray.shape == originalImageDataArray.shape
    watermarkSize = calculateOutsideWatermarkSize(imageDataArray)
    watermarkData = calculateWatermark(password, watermarkSize)

    extracted = extractRobustWatermark(
        imageDataArray[0][0], originalImageDataArray[0][0], stringToBit(watermarkData[0]), embedFactor)

    watermarkCheck = np.ones(watermarkSize, dtype=np.int8)

    counter = 0
    # upside
    for i in range(imageDataArray[0].shape[0]):
        extracted = extractRobustWatermark(
            imageDataArray[0][i], originalImageDataArray[0][i], stringToBit(watermarkData[counter]), embedFactor)
        tmp = checkBitRobustWatermark(
            extracted, watermarkData[counter])
        if tmp is False:
            watermarkCheck[counter] = False
        counter += 1
    counter -= 1

    # rightside
    for i in range(imageDataArray[1].shape[0]):
        extracted = extractRobustWatermark(
            imageDataArray[1][i], originalImageDataArray[1][i], stringToBit(watermarkData[counter]), embedFactor)
        tmp = checkBitRobustWatermark(
            extracted, watermarkData[counter])
        if tmp is False:
            watermarkCheck[counter] = False
        counter += 1
    counter -= 1

    # bottomside
    for i in range(imageDataArray[2].shape[0]):
        extracted = extractRobustWatermark(
            imageDataArray[2][i], originalImageDataArray[2][i], stringToBit(watermarkData[counter]), embedFactor)
        tmp = checkBitRobustWatermark(
            extracted, watermarkData[counter])
        if tmp is False:
            watermarkCheck[counter] = False
        counter += 1
    counter -= 1

    # leftside
    for i in range(imageDataArray[3].shape[0]):
        extracted = extractRobustWatermark(
            imageDataArray[3][i], originalImageDataArray[3][i], stringToBit(watermarkData[counter % watermarkSize]), embedFactor)
        watermarkCheck[counter % watermarkSize] = checkBitRobustWatermark(
            extracted, watermarkData[counter % watermarkSize])
        if tmp is False:
            watermarkCheck[counter % watermarkSize] = False
        counter += 1
    counter -= 1

    return watermarkCheck


imageData = readImage("original.png")
# imageData2 = readImage("original2.png")

# embed watermark
insideImageData, outsideImageData = splitImage(imageData, 32)
watermarkedOutsideImageData = processEmbedRobustWatermark(
    outsideImageData, "thor", 10)
watermarkedInsideImageData = embedFragileWatermark(insideImageData)
# watermark result
mergedImageData = mergeImage(
    watermarkedInsideImageData, watermarkedOutsideImageData)
# # preview watermark
# Image.fromarray(imageData).show()
# Image.fromarray(mergedImageData).show()

# extract watermark
insideWatermark, outsideWatermark = splitImage(mergedImageData, 32)
# fragile
extractedFragile = extractFragileWatermark(insideWatermark)
# print(extractedFragile)
print("is fragile valid: ", end='')
print(len(np.where(extractedFragile == 0)[0]) == 0 and len(
    np.where(extractedFragile == 0)[1]) == 0)

robustCheckResult = processExtractRobustWatermark(outsideWatermark, outsideImageData, "thor", 10)
print("is robust valid: ", end='')
print(len(np.where(robustCheckResult == False)[0]) == 0)

# add noise
Image.fromarray(imageData).show()
# noise_img = random_noise(imageData, mode="gaussian")
# noise_img = random_noise(imageData, mode="localvar")
# noise_img = random_noise(imageData, mode="salt")
# noise_img = random_noise(imageData, mode="pepper")
# noise_img = random_noise(imageData, mode="s&p")
noise_img = random_noise(imageData, mode="speckle")
noise_img = (255*noise_img).astype(np.uint8)
Image.fromarray(noise_img).show()