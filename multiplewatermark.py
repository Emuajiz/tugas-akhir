import cmath
import hashlib
import math
import random
import time
import os
from PIL import Image
import numpy as np
import skimage
from scipy.ndimage import gaussian_filter

BLUR_CONSTANT = 0
INSIDE_ONLY = "INSIDE_ONLY"


def charToBit(char):
    assert len(char) == 1
    binary = np.zeros(8, dtype=np.uint8)
    tmp = ord(char)
    counter = 0
    while (tmp):
        binary[counter] = tmp % 2
        tmp = tmp // 2
        counter += 1
    return np.flip(binary)


def stringToBitArray(string):
    if len(string) == 1:
        return charToBit(string)
    else:
        return np.concatenate((charToBit(string[0]), stringToBitArray(string[1:])))


def psnr(A, B):
    mse = np.mean((A - B) ** 2)
    if (mse == 0):
        return 100
    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr


def readImage(filename):
    image = Image.open(filename)
    return np.array(image)


def validateOutsideImageData(insideImagedata):
    """helper to validate outside image data using overlapping at image corner data"""
    if psnr(insideImagedata[0][0], insideImagedata[3][-1]) != 100:
        return False
    if psnr(insideImagedata[1][0], insideImagedata[0][-1]) != 100:
        return False
    if psnr(insideImagedata[2][0], insideImagedata[1][-1]) != 100:
        return False
    if psnr(insideImagedata[3][0], insideImagedata[2][-1]) != 100:
        return False
    if insideImagedata[0].shape[-2:] != insideImagedata[1].shape[-2:]:
        return False
    if insideImagedata[1].shape[-2:] != insideImagedata[2].shape[-2:]:
        return False
    if insideImagedata[2].shape[-2:] != insideImagedata[3].shape[-2:]:
        return False
    if insideImagedata[0].shape[0] != insideImagedata[2].shape[0]:
        return False
    if insideImagedata[1].shape[0] != insideImagedata[3].shape[0]:
        return False
    return True


def splitImage(imageData, outsideImageSize, mode="ALL"):
    """
    function to split image into inside part and outside part
    example:
    ```py
    splitImage(data, (32, 32))
    ```
    """
    imageSize = imageData.shape
    outsideHeight = outsideImageSize[0]
    outsideWidth = outsideImageSize[1]
    assert imageSize[0] % outsideHeight == 0
    assert imageSize[1] % outsideWidth == 0

    insidePart = imageData[outsideHeight:imageSize[0] -
                           outsideHeight, outsideWidth:imageSize[1] - outsideWidth]

    if (mode == INSIDE_ONLY):
        return [insidePart, []]

    xTotalImagePart = imageSize[1] // outsideWidth
    yTotalImagePart = imageSize[0] // outsideHeight
    xOutsideShape = (xTotalImagePart, outsideHeight, outsideWidth)
    yOutsideShape = (yTotalImagePart, outsideHeight, outsideWidth)
    upside = np.zeros(xOutsideShape, dtype=np.uint8)
    bottomside = np.zeros(xOutsideShape, dtype=np.uint8)
    rightside = np.zeros(yOutsideShape, dtype=np.uint8)
    leftside = np.zeros(yOutsideShape, dtype=np.uint8)
    for i, val in enumerate(range(0, imageSize[1], outsideWidth)):
        # upside
        upside[i] = np.array(imageData[0:outsideHeight, val:val+outsideWidth])

        # bottomside
        bottomside[i] = np.array(imageData[imageSize[0] - outsideHeight:imageSize[0],
                                 imageSize[1] - val - outsideWidth:imageSize[1] - val])

    for i, val in enumerate(range(0, imageSize[0], outsideHeight)):
        # rightside
        rightside[i] = np.array(
            imageData[val:val+outsideHeight, imageSize[1] - outsideWidth:imageSize[1]])

        # leftside
        leftside[i] = np.array(
            imageData[imageSize[0] - val - outsideHeight:imageSize[0] - val, 0:outsideWidth])

    # overlapping for check
    assert validateOutsideImageData(
        [upside, rightside, bottomside, leftside]) == True

    return insidePart, [upside, rightside, bottomside, leftside]


def mergeImage(insidePart, outsidePart):
    assert validateOutsideImageData(outsidePart) == True

    imageSize = (insidePart.shape[0] + 2*outsidePart[0].shape[1],
                 insidePart.shape[1] + 2*outsidePart[0].shape[2])
    imageData = np.zeros(shape=imageSize, dtype=np.uint8)
    imageData[outsidePart[0].shape[1]:insidePart.shape[0] +
              outsidePart[0].shape[1], outsidePart[0].shape[2]:insidePart.shape[1] + outsidePart[0].shape[2]] = insidePart

    for i, val in enumerate(outsidePart[0]):
        # upside
        imageData[0:val.shape[0], i*val.shape[1]:(i+1)*val.shape[1]] = val
        # bottomside
        imageData[-val.shape[0]:, i*val.shape[1]
            :(i+1)*val.shape[1]] = outsidePart[2][-(i+1)]

    for i, val in enumerate(outsidePart[1]):
        # rightside
        imageData[i*val.shape[0]:(i+1)*val.shape[0], -val.shape[1]:] = val
        # leftside
        imageData[i*val.shape[0]:(i+1)*val.shape[0],
                  :val.shape[1]] = outsidePart[3][-(i+1)]
    return imageData


def normalize(arr, t_min, t_max):
    "function to change scaling"
    norm_arr = []
    diff = t_max - t_min
    diff_arr = max(arr) - min(arr)
    for i in arr:
        temp = (((i - min(arr))*diff)/diff_arr) + t_min
        norm_arr.append(temp)
    return norm_arr


def imageHash(imageData, mode="NORMAL"):
    t_start = time.process_time()
    h = hashlib.sha256()
    # remove last bit
    tmp = (imageData / np.array(2, dtype=int))
    # make sure integer
    tmp = tmp.astype(np.uint8)
    h.update(np.ascontiguousarray(tmp))
    t_stop = time.process_time()
    if (mode == "VERBOSE"):
        print("hashing time: ", t_stop-t_start)
    return h.digest()


def makeFragileWatermarkPayload(imageData, mode="NORMAL"):
    t_start = time.process_time()
    fragileWatermarkPayload = imageHash(imageData)
    fragileWatermarkPayloadBitString = []
    for i in fragileWatermarkPayload:
        fragileWatermarkPayloadBitString.append("{0:08b}".format(i))
    t_stop = time.process_time()
    if (mode == "VERBOSE"):
        print("construct fragile payload time: ", t_stop-t_start)
    return ''.join(fragileWatermarkPayloadBitString)


def createEmbedOrder(imageShape, password=False):
    order = [(i, j) for i in range(imageShape[0])
             for j in range(imageShape[1])]
    if password:
        random.seed(password, version=2)
        return random.sample(order, k=len(order))
    return order


def embedFragileWatermark(imageData, password, mode="NORMAL"):
    watermarkedImageData = np.zeros(imageData.shape, dtype=np.uint8)
    fragileWatermarkPayload = makeFragileWatermarkPayload(imageData)
    order = createEmbedOrder(imageData.shape, password)
    t_start = time.process_time()
    for i, val in enumerate(order):
        p = imageData[val[0], val[1]]
        p &= 0b11111110
        p |= int(fragileWatermarkPayload[i % len(fragileWatermarkPayload)])
        watermarkedImageData[val[0], val[1]] = p
    t_stop = time.process_time()
    if (mode == "VERBOSE"):
        print("embedding fragile time: ", t_stop-t_start)
    return watermarkedImageData


def extractFragileWatermark(imageData, password):
    fragileWatermarkPayload = makeFragileWatermarkPayload(imageData)
    fragileWatermark = np.zeros(imageData.shape, dtype=np.uint8)
    order = createEmbedOrder(imageData.shape, password)
    for i, val in enumerate(order):
        if (imageData[val[0], val[1]] % 2 == int(fragileWatermarkPayload[i % len(fragileWatermarkPayload)])):
            fragileWatermark[val[0], val[1]] = 1
        else:
            fragileWatermark[val[0], val[1]] = 0
    return fragileWatermark


def calculateWatermarkPosition(vectorLength, imageShape, radius=-1):
    center = imageShape[0] // 2 + 1, imageShape[1] // 2 + 1

    if radius == -1:
        radius = min(imageShape) // 4
    if radius > (min(imageShape) // 2 - 2):
        raise ValueError

    def x(t): return center[0] + int(radius *
                                     math.cos(t * 2 * math.pi / vectorLength))

    def y(t): return center[1] + int(radius *
                                     math.sin(t * 2 * math.pi / vectorLength))

    indices = [(x(t), y(t)) for t in range(vectorLength)]
    return indices


def calculateForWatermark(magnitude, position):
    tmp = 0
    magnitude = np.pad(magnitude, ((1, 1), (1, 1)))
    tmp2 = (position[0]+1, position[1]+1)
    for i in range(-1, 2, 1):
        for j in range(-1, 2, 1):
            tmp += magnitude[tmp2[0]+i][tmp2[1]+j]
    return tmp / 9


def embedRobustWatermark(imageData, watermarkData, alpha=1, radius=-1):
    # faktor pengali alpha
    vectorLength = len(watermarkData)
    indices = calculateWatermarkPosition(vectorLength, imageData.shape, radius)
    imageFourier = np.fft.fftshift(np.fft.fft2(imageData))
    mag = np.abs(imageFourier)
    phase = np.angle(imageFourier)
    watermarkElement = np.zeros(len(watermarkData), dtype=np.float64)
    for i, val in enumerate(indices):
        watermarkElement[i] = watermarkData[i] * \
            calculateForWatermark(mag, val)
    for i, val in enumerate(indices):
        tmp = mag[val[0], val[1]] + alpha * watermarkElement[i]
        fourierTmp = tmp * cmath.exp(1j * phase[val[0], val[1]])
        imageFourier[val[0], val[1]] = fourierTmp
    watermarkedImageData = np.round(np.abs(np.fft.ifft2(
        np.fft.ifftshift(imageFourier)))).astype(np.uint8)
    return watermarkedImageData


def extractRobustWatermark(imageData, originalImageData, watermarkData, alpha=1, radius=-1):
    vectorLength = len(watermarkData)
    indices = calculateWatermarkPosition(vectorLength, imageData.shape, radius)
    imageFourier = np.fft.fftshift(np.fft.fft2(imageData))
    originalImageFourier = np.fft.fftshift(np.fft.fft2(originalImageData))
    mag = np.abs(imageFourier)
    originalMag = np.abs(originalImageFourier)
    extractedWatermarkData = np.zeros(len(watermarkData), dtype=np.float64)
    watermarkElement = np.zeros(len(watermarkData), dtype=np.float64)
    for i, val in enumerate(indices):
        watermarkElement[i] = calculateForWatermark(mag, val)
    for i, val in enumerate(indices):
        extractedWatermarkData[i] = (mag[val[0], val[1]] - originalMag[val[0], val[1]]) / \
            (alpha * watermarkElement[i])
    return extractedWatermarkData


def calculateOutsideWatermarkSize(imageData, bitPerPart=8):
    size = 0
    for i in range(len(imageData)):
        size += (imageData[i].shape[0] - 1)
    return size * bitPerPart


def calculateWatermark(password, watermarkLength):
    # proses seeding
    random.seed(password + str(watermarkLength), version=2)
    # proses menghasilkan string random
    # Mersenne Twister
    res = np.array(random.choices([0, 1], k=watermarkLength), dtype=np.uint8)
    return res


def processEmbedRobustWatermark(imageDataArray, password, embedFactor=10, bitPerPart=8, radius=-1):
    watermarkSize = calculateOutsideWatermarkSize(imageDataArray, bitPerPart)
    watermarkData = calculateWatermark(password, watermarkSize)
    counter = 0
    # upside
    upsideDataArray = np.zeros(imageDataArray[0].shape)
    for i in range(imageDataArray[0].shape[0]):
        pos = counter*bitPerPart
        upsideDataArray[i] = embedRobustWatermark(
            imageDataArray[0][i], watermarkData[pos:pos+bitPerPart], embedFactor, radius)
        counter += 1
    counter -= 1

    # rightside
    rightsideDataArray = np.zeros(imageDataArray[1].shape)
    for i in range(imageDataArray[1].shape[0]):
        pos = counter*bitPerPart
        rightsideDataArray[i] = embedRobustWatermark(
            imageDataArray[1][i], watermarkData[pos:pos+bitPerPart], embedFactor, radius)
        counter += 1
    counter -= 1

    # bottomside
    bottomsideDataArray = np.zeros(imageDataArray[2].shape)
    for i in range(imageDataArray[2].shape[0]):
        pos = counter*bitPerPart
        bottomsideDataArray[i] = embedRobustWatermark(
            imageDataArray[2][i], watermarkData[pos:pos+bitPerPart], embedFactor, radius)
        counter += 1
    counter -= 1

    # leftside
    leftsideDataArray = np.zeros(imageDataArray[3].shape)
    for i in range(imageDataArray[3].shape[0]):
        pos = (counter*bitPerPart) % watermarkSize
        leftsideDataArray[i] = embedRobustWatermark(
            imageDataArray[3][i], watermarkData[pos:pos+bitPerPart], embedFactor, radius)
        counter += 1
    counter -= 1

    # overlapping for check
    assert validateOutsideImageData(
        [upsideDataArray, rightsideDataArray, bottomsideDataArray, leftsideDataArray]) == True

    return [upsideDataArray, rightsideDataArray, bottomsideDataArray, leftsideDataArray]


def checkBitRobustWatermark(extracted, original, bitPerPart):
    if (len(np.unique(extracted)) > 1):
        normExtracted = normalize(extracted, 0, 1)
        bitExtracted = np.where(np.array(normExtracted) > 0.5, 1, 0)
        similarityArray = np.zeros(bitPerPart, dtype=np.uint8)
        for i in range(bitPerPart):
            if bitExtracted[i] == original[i]:
                similarityArray[i] = 1
        return similarityArray
    if (len(np.unique(original)) == 1):
        return np.repeat(1, bitPerPart)
    else:
        return np.repeat(0.5, bitPerPart)


def processExtractRobustWatermark(imageDataArray, originalImageDataArray, password, embedFactor=10, bitPerPart=8, radius=-1):
    watermarkSize = calculateOutsideWatermarkSize(imageDataArray, bitPerPart)
    watermarkData = calculateWatermark(password, watermarkSize)
    watermarkCheck = np.zeros(watermarkSize, dtype=np.int8)

    counter = 0
    # upside
    for i in range(imageDataArray[0].shape[0]):
        pos = counter*bitPerPart
        extracted = extractRobustWatermark(
            imageDataArray[0][i], originalImageDataArray[0][i], watermarkData[pos:pos+bitPerPart], embedFactor, radius)
        watermarkCheck[pos:pos+bitPerPart] = checkBitRobustWatermark(
            extracted, watermarkData[pos:pos+bitPerPart], bitPerPart)
        counter += 1
    counter -= 1

    # rightside
    for i in range(imageDataArray[1].shape[0]):
        pos = counter*bitPerPart
        extracted = extractRobustWatermark(
            imageDataArray[1][i], originalImageDataArray[1][i], watermarkData[pos:pos+bitPerPart], embedFactor, radius)
        watermarkCheck[pos:pos+bitPerPart] = checkBitRobustWatermark(
            extracted, watermarkData[pos:pos+bitPerPart], bitPerPart)
        counter += 1
    counter -= 1

    # bottomside
    for i in range(imageDataArray[2].shape[0]):
        pos = counter*bitPerPart
        extracted = extractRobustWatermark(
            imageDataArray[2][i], originalImageDataArray[2][i], watermarkData[pos:pos+bitPerPart], embedFactor, radius)
        watermarkCheck[pos:pos+bitPerPart] = checkBitRobustWatermark(
            extracted, watermarkData[pos:pos+bitPerPart], bitPerPart)
        counter += 1
    counter -= 1

    # leftside
    for i in range(imageDataArray[3].shape[0]):
        pos = (counter*bitPerPart) % watermarkSize
        extracted = extractRobustWatermark(
            imageDataArray[3][i], originalImageDataArray[3][i], watermarkData[pos:pos+bitPerPart], embedFactor, radius)
        watermarkCheck[pos:pos+bitPerPart] = checkBitRobustWatermark(
            extracted, watermarkData[pos:pos+bitPerPart], bitPerPart)
        counter += 1
    counter -= 1

    return np.average(watermarkCheck)


def mergeChannel(imageDataR, imageDataG, imageDataB):
    imageDataRGB = np.zeros(imageDataR.shape + (3, ), dtype=np.uint8)
    imageDataRGB[:, :, 0] = imageDataR
    imageDataRGB[:, :, 1] = imageDataG
    imageDataRGB[:, :, 2] = imageDataB
    return imageDataRGB


def rgbToYUV(imageData):
    return np.array(Image.fromarray(imageData, mode="RGB").convert("YCbCr"))


def yuvToRGB(imageData):
    return np.array(Image.fromarray(imageData, mode="YCbCr").convert("RGB"))


def processEmbedMultipleWatermark(imageData, password, outsideImageSize=(32, 32), factor=10, show=False, save=False, out="out.png", bitPerPart=8, radius=-1):
    # embed watermark
    insideImageData, outsideImageData = splitImage(
        imageData, outsideImageSize)
    watermarkedOutsideImageData = processEmbedRobustWatermark(
        outsideImageData, password, factor, bitPerPart, radius)
    watermarkedInsideImageData = embedFragileWatermark(
        insideImageData, password)
    # watermark result
    mergedImageData = mergeImage(
        watermarkedInsideImageData, watermarkedOutsideImageData)

    if show:
        # preview watermark
        Image.fromarray(imageData).show()
        Image.fromarray(mergedImageData).show()

    if save:
        Image.fromarray(mergedImageData).save(out)

    return mergedImageData


def processEmbedFragileWatermarkColor(imageData, password, outsideImageSize=(32, 32), mode="NORMAL"):
    insideImageDataR, _ = splitImage(
        imageData[:, :, 0], outsideImageSize, INSIDE_ONLY)
    insideImageDataG, _ = splitImage(
        imageData[:, :, 1], outsideImageSize, INSIDE_ONLY)
    insideImageDataB, _ = splitImage(
        imageData[:, :, 2], outsideImageSize, INSIDE_ONLY)
    t_start = time.process_time()
    watermarkedInsideImageDataR = embedFragileWatermark(
        insideImageDataR, password)
    watermarkedInsideImageDataG = embedFragileWatermark(
        insideImageDataG, password)
    watermarkedInsideImageDataB = embedFragileWatermark(
        insideImageDataB, password)
    t_stop = time.process_time()

    if (mode == "VERBOSE"):
        print("fragile processing time:",
              t_stop-t_start)

    return watermarkedInsideImageDataR, watermarkedInsideImageDataG, watermarkedInsideImageDataB


def processEmbedMultipleWatermarkColor(imageData, password, outsideImageSize=(32, 32), factor=10, show=False, save=False, dir="watermarked", out="out.png", bitPerPart=8, radius=-1, preCalcFragileWatermark=[], mode="NORMAL"):
    imageDataYUV = rgbToYUV(imageData)
    _, outsideImageDataY = splitImage(
        imageDataYUV[:, :, 0], outsideImageSize)
    _, outsideImageDataU = splitImage(
        imageDataYUV[:, :, 1], outsideImageSize)
    _, outsideImageDataV = splitImage(
        imageDataYUV[:, :, 2], outsideImageSize)

    t1_start = time.process_time()
    if (len(preCalcFragileWatermark) != 3):
        watermarkedInsideImageDataR, watermarkedInsideImageDataG, watermarkedInsideImageDataB = processEmbedFragileWatermarkColor(
            imageData, password, outsideImageSize, mode)
    else:
        watermarkedInsideImageDataR, watermarkedInsideImageDataG, watermarkedInsideImageDataB = preCalcFragileWatermark
    t1_stop = time.process_time()

    t2_start = time.process_time()
    watermarkedOutsideImageDataY = processEmbedRobustWatermark(
        outsideImageDataY, password, factor, bitPerPart, radius)
    t2_stop = time.process_time()

    # variable to construct outside RGB data from outside YUV data
    outsideImageDataR = [
        np.zeros(outsideImageDataY[0].shape, dtype=np.uint8),
        np.zeros(outsideImageDataY[1].shape, dtype=np.uint8),
        np.zeros(outsideImageDataY[2].shape, dtype=np.uint8),
        np.zeros(outsideImageDataY[3].shape, dtype=np.uint8)
    ]
    outsideImageDataG = [
        np.zeros(outsideImageDataY[0].shape, dtype=np.uint8),
        np.zeros(outsideImageDataY[1].shape, dtype=np.uint8),
        np.zeros(outsideImageDataY[2].shape, dtype=np.uint8),
        np.zeros(outsideImageDataY[3].shape, dtype=np.uint8)
    ]
    outsideImageDataB = [
        np.zeros(outsideImageDataY[0].shape, dtype=np.uint8),
        np.zeros(outsideImageDataY[1].shape, dtype=np.uint8),
        np.zeros(outsideImageDataY[2].shape, dtype=np.uint8),
        np.zeros(outsideImageDataY[3].shape, dtype=np.uint8)
    ]

    for i in range(len(watermarkedOutsideImageDataY)):
        for j, _ in enumerate(watermarkedOutsideImageDataY[i]):
            yuvImageData = np.zeros(
                watermarkedOutsideImageDataY[i][j].shape + (3,), dtype=np.uint8)
            yuvImageData[:, :, 0] = watermarkedOutsideImageDataY[i][j]
            yuvImageData[:, :, 1] = outsideImageDataU[i][j]
            yuvImageData[:, :, 2] = outsideImageDataV[i][j]
            rgbImageData = yuvToRGB(yuvImageData)
            outsideImageDataR[i][j] = rgbImageData[:, :, 0]
            outsideImageDataG[i][j] = rgbImageData[:, :, 1]
            outsideImageDataB[i][j] = rgbImageData[:, :, 2]

    imageDataR = mergeImage(watermarkedInsideImageDataR, outsideImageDataR)
    imageDataG = mergeImage(watermarkedInsideImageDataG, outsideImageDataG)
    imageDataB = mergeImage(watermarkedInsideImageDataB, outsideImageDataB)
    watermarkedImageData = mergeChannel(imageDataR, imageDataG, imageDataB)

    if (show == True):
        Image.fromarray(watermarkedImageData).show()

    if (save == True):
        if (os.path.exists(dir) == False):
            os.mkdir(dir)
        if (os.path.isdir(dir)):
            Image.fromarray(watermarkedImageData).save(f"{dir}/{out}")
        else:
            print("cannot save to file")

    if (mode == "VERBOSE"):
        print("fragile processing time:",
              t1_stop-t1_start)
        print("robust processing time:",
              t2_stop-t2_start)
    return watermarkedImageData


def processExtractMultipleWatermark(imageData, originalImageData, password, outsideImageSize=(32, 32), factor=10, bitPerPart=8, radius=-1):
    insideWatermark, outsideWatermark = splitImage(imageData, outsideImageSize)
    blurred = gaussian_filter(originalImageData, sigma=BLUR_CONSTANT)
    _, originalOutsideWatermark = splitImage(
        blurred, outsideImageSize)
    # fragile
    extractedFragile = extractFragileWatermark(insideWatermark, password)
    # robust
    robustCheckResult = processExtractRobustWatermark(
        outsideWatermark, originalOutsideWatermark, password, factor, bitPerPart, radius)

    print(np.average(extractedFragile))
    print(robustCheckResult)


def processExtractMultipleWatermarkColor(imageData, originalImageData, password, outsideImageSize=(32, 32), factor=10, bitPerPart=8, radius=-1):
    imageDataYUV = rgbToYUV(imageData)
    originalImageDataYUV = rgbToYUV(originalImageData)
    insideImageDataR, _ = splitImage(
        imageData[:, :, 0], outsideImageSize, INSIDE_ONLY)
    insideImageDataG, _ = splitImage(
        imageData[:, :, 1], outsideImageSize, INSIDE_ONLY)
    insideImageDataB, _ = splitImage(
        imageData[:, :, 2], outsideImageSize, INSIDE_ONLY)
    _, outsideImageDataY = splitImage(
        imageDataYUV[:, :, 0], outsideImageSize)
    _, originalOutsideImageDataY = splitImage(
        originalImageDataYUV[:, :, 0], outsideImageSize)

    # fragile
    extractedFragileR = extractFragileWatermark(insideImageDataR, password)
    extractedFragileG = extractFragileWatermark(insideImageDataG, password)
    extractedFragileB = extractFragileWatermark(insideImageDataB, password)

    # fragile
    robustCheckResult = processExtractRobustWatermark(
        outsideImageDataY, originalOutsideImageDataY, password, factor, bitPerPart, radius)

    redCheck = np.average(extractedFragileR)
    greenCheck = np.average(extractedFragileG)
    blueCheck = np.average(extractedFragileB)
    return [(redCheck, greenCheck, blueCheck), robustCheckResult]


def splitThenMergeShouldReturnSameImage(filename):
    imageData = readImage(filename)
    insideImageData, outsideImageData = splitImage(imageData, (32, 32))
    merged = mergeImage(insideImageData, outsideImageData)
    return psnr(imageData, merged) == 100

# assert splitThenMergeShouldReturnSameImage("original.png") == True


if __name__ == "__main__":
    outsideShape = (40, 40)
    factor = 20
    bitPerPart = 8
    radius = 10

    imageData = readImage("original1-color.png")
    # watermarked = multipleWatermark(
    #     imageData, "thor", outsideShape, factor, True, False, "watermarked.png", bitPerPart, radius)
    watermarked = processEmbedMultipleWatermarkColor(
        imageData, "thor", outsideShape, factor, True, True, "watermarked", "watermarked.png", bitPerPart, radius)
    # extractMultipleWatermark(watermarked, imageData, "thor", outsideShape, factor, bitPerPart, radius)
    fragileCheck, robustCheck = processExtractMultipleWatermarkColor(
        watermarked, imageData, "thor", outsideShape, factor, bitPerPart, radius)
