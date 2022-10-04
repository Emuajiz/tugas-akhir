import cmath
import hashlib
import math
import random
import string
from PIL import Image
import numpy as np
import skimage


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


def validateInsideImageData(insideImagedata):
    """helper to validate inside image data after splitting using overlapping at image corner data"""
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


def splitImage(imageData, outsideImageSize):
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

    xTotalImagePart = imageSize[1] // outsideWidth
    yTotalImagePart = imageSize[0] // outsideHeight
    upside = np.zeros((xTotalImagePart, outsideHeight,
                      outsideWidth), dtype=np.uint8)
    bottomside = np.zeros(
        (xTotalImagePart, outsideHeight, outsideWidth), dtype=np.uint8)
    rightside = np.zeros((yTotalImagePart, outsideHeight,
                         outsideWidth), dtype=np.uint8)
    leftside = np.zeros((yTotalImagePart, outsideHeight,
                        outsideWidth), dtype=np.uint8)
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
    assert validateInsideImageData(
        [upside, rightside, bottomside, leftside]) == True

    return insidePart, [upside, rightside, bottomside, leftside]


def mergeImage(insidePart, outsidePart):
    assert validateInsideImageData(outsidePart) == True

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
    "function to normalize array"
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


def calculateWatermarkPosition(vectorLength, imageShape, radius=-1):
    center = imageShape[0] // 2 + 1, imageShape[1] // 2 + 1

    if radius == -1:
        radius = min(imageShape) // 4

    def x(t): return center[0] + int(radius *
                                     math.cos(t * 2 * math.pi / vectorLength))

    def y(t): return center[1] + int(radius *
                                     math.sin(t * 2 * math.pi / vectorLength))

    indices = [(x(t), y(t)) for t in range(vectorLength)]
    return indices


def calculateForWatermark(magnitude, position):
    tmp = 0
    for i in range(-1, 2, 1):
        for j in range(-1, 2, 1):
            tmp += magnitude[position[0]+i][position[1]+j]
    return tmp / 9


def embedRobustWatermark(imageData, watermarkData, alpha=1):
    # faktor pengali alpha
    vectorLength = len(watermarkData)
    indices = calculateWatermarkPosition(vectorLength, imageData.shape)
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


def extractRobustWatermark(imageData, originalImageData, watermarkData, alpha=1):
    vectorLength = len(watermarkData)
    indices = calculateWatermarkPosition(vectorLength, imageData.shape)
    imageFourier = np.fft.fftshift(np.fft.fft2(imageData))
    originalImageFourier = np.fft.fftshift(np.fft.fft2(originalImageData))
    mag = np.abs(imageFourier)
    originalMag = np.abs(originalImageFourier)
    extractedWatermarkData = []
    watermarkElement = np.zeros(len(watermarkData), dtype=np.float64)
    for i, val in enumerate(indices):
        watermarkElement[i] = calculateForWatermark(mag, val)
    for i, val in enumerate(indices):
        tmp = (mag[val[0], val[1]] - originalMag[val[0], val[1]]) / \
            (alpha * watermarkElement[i])
        extractedWatermarkData.append(tmp)
    return extractedWatermarkData


def calculateOutsideWatermarkSize(imageData):
    size = 0
    for i in range(len(imageData)):
        size += (imageData[i].shape[0] - 1)
    return size


def calculateWatermark(password, watermarkLength):
    # proses seeding
    random.seed(password + str(watermarkLength), version=2)
    # proses menghasilkan string random
    # Mersenne Twister
    res = ''.join(random.choices(string.ascii_letters +
                                 string.digits, k=watermarkLength))
    return res


def processEmbedRobustWatermark(imageDataArray, password, embedFactor=10):
    watermarkSize = calculateOutsideWatermarkSize(imageDataArray)
    watermarkData = calculateWatermark(password, watermarkSize)
    # exit()
    counter = 0
    # upside
    upsideDataArray = np.zeros(imageDataArray[0].shape)
    # print(upsideDataArray.shape)
    for i in range(imageDataArray[0].shape[0]):
        # print(imageDataArray[0,i].shape)
        upsideDataArray[i] = embedRobustWatermark(
            imageDataArray[0][i], charToBit(watermarkData[counter]), embedFactor)
        counter += 1
    counter -= 1
    # print(counter)

    # rightside
    rightsideDataArray = np.zeros(imageDataArray[1].shape)
    # print(rightsideDataArray.shape)
    for i in range(imageDataArray[1].shape[0]):
        # print(imageDataArray[0,i].shape)
        rightsideDataArray[i] = embedRobustWatermark(
            imageDataArray[1][i], charToBit(watermarkData[counter]), embedFactor)
        counter += 1
    counter -= 1
    # print(counter)

    # bottomside
    bottomsideDataArray = np.zeros(imageDataArray[2].shape)
    # print(bottomsideDataArray.shape)
    for i in range(imageDataArray[2].shape[0]):
        # print(imageDataArray[0,i].shape)
        bottomsideDataArray[i] = embedRobustWatermark(
            imageDataArray[2][i], charToBit(watermarkData[counter]), embedFactor)
        counter += 1
    counter -= 1
    # print(counter)

    # leftside
    leftsideDataArray = np.zeros(imageDataArray[3].shape)
    # print(leftsideDataArray.shape)
    for i in range(imageDataArray[3].shape[0]):
        # print(imageDataArray[0,i].shape)
        leftsideDataArray[i] = embedRobustWatermark(
            imageDataArray[3][i], charToBit(watermarkData[counter % watermarkSize]), embedFactor)
        counter += 1
    counter -= 1
    # print(counter)

    assert psnr(upsideDataArray[0], leftsideDataArray[-1]) == 100
    assert psnr(rightsideDataArray[0], upsideDataArray[-1]) == 100
    assert psnr(bottomsideDataArray[0], rightsideDataArray[-1]) == 100
    assert psnr(leftsideDataArray[0], bottomsideDataArray[-1]) == 100

    return [upsideDataArray, rightsideDataArray, bottomsideDataArray, leftsideDataArray]


def checkBitRobustWatermark(extracted, original, threshold=0.5):
    norm_extracted = normalize(extracted, 0, 1)
    bit_extracted = np.where(np.array(norm_extracted) > threshold, 1, 0)
    tmp = charToBit(original)
    similarity = 0
    for i in range(8):
        if bit_extracted[i] == tmp[i]:
            similarity += 1
    return similarity


def processExtractRobustWatermark(imageDataArray, originalImageDataArray, password, embedFactor=10):
    assert len(imageDataArray) == len(originalImageDataArray)
    watermarkSize = calculateOutsideWatermarkSize(imageDataArray)
    watermarkData = calculateWatermark(password, watermarkSize)
    watermarkCheck = np.zeros(watermarkSize, dtype=np.int8)

    counter = 0
    # upside
    for i in range(imageDataArray[0].shape[0]):
        extracted = extractRobustWatermark(
            imageDataArray[0][i], originalImageDataArray[0][i], charToBit(watermarkData[counter]), embedFactor)
        watermarkCheck[counter] = checkBitRobustWatermark(
            extracted, watermarkData[counter])
        counter += 1
    counter -= 1

    # rightside
    for i in range(imageDataArray[1].shape[0]):
        extracted = extractRobustWatermark(
            imageDataArray[1][i], originalImageDataArray[1][i], charToBit(watermarkData[counter]), embedFactor)
        watermarkCheck[counter] = checkBitRobustWatermark(
            extracted, watermarkData[counter])
        counter += 1
    counter -= 1

    # bottomside
    for i in range(imageDataArray[2].shape[0]):
        extracted = extractRobustWatermark(
            imageDataArray[2][i], originalImageDataArray[2][i], charToBit(watermarkData[counter]), embedFactor)
        watermarkCheck[counter] = checkBitRobustWatermark(
            extracted, watermarkData[counter])
        counter += 1
    counter -= 1

    # leftside
    for i in range(imageDataArray[3].shape[0]):
        extracted = extractRobustWatermark(
            imageDataArray[3][i], originalImageDataArray[3][i], charToBit(watermarkData[counter % watermarkSize]), embedFactor)
        watermarkCheck[counter % watermarkSize] = checkBitRobustWatermark(
            extracted, watermarkData[counter % watermarkSize])
        counter += 1
    counter -= 1

    return np.average(watermarkCheck / 8)


def multipleWatermark(filename, password, outsideImageSize, show=False, save=False, factor=10, out=False):
    imageData = readImage(filename)
    # embed watermark
    insideImageData, outsideImageData = splitImage(imageData, outsideImageSize)
    watermarkedOutsideImageData = processEmbedRobustWatermark(
        outsideImageData, password, factor)
    watermarkedInsideImageData = embedFragileWatermark(insideImageData)
    # watermark result
    mergedImageData = mergeImage(
        watermarkedInsideImageData, watermarkedOutsideImageData)

    if show:
        # preview watermark
        Image.fromarray(imageData).show()
        Image.fromarray(mergedImageData).show()

    if save:
        Image.fromarray(mergedImageData).save(out if out else "out.png")
    
    return mergedImageData

def extractMultipleWatermark():
    # extract watermark
    insideWatermark, outsideWatermark = splitImage(
        mergedImageData, outsideImageSize)
    # fragile
    extractedFragile = extractFragileWatermark(insideWatermark)

    robustCheckResult = processExtractRobustWatermark(
        outsideWatermark, outsideImageData, "thor", factor)


def splitThenMergeShouldReturnSameImage(filename):
    imageData = readImage(filename)
    insideImageData, outsideImageData = splitImage(imageData, (32, 32))
    merged = mergeImage(insideImageData, outsideImageData)
    return psnr(imageData, merged) == 100


assert splitThenMergeShouldReturnSameImage("original.png") == True

multipleWatermark("original.png", "thor", (32, 32), False, False, 10)

# add noise
# Image.fromarray(imageData).show()
# noise_img = random_noise(imageData, mode="gaussian")
# noise_img = random_noise(imageData, mode="localvar")
# noise_img = random_noise(imageData, mode="salt")
# noise_img = random_noise(imageData, mode="pepper")
# noise_img = random_noise(imageData, mode="s&p")
# noise_img = skimage.util.random_noise(imageData, mode="speckle")
# noise_img = (255*noise_img).astype(np.uint8)
# Image.fromarray(noise_img).show()

# # blur
# Image.fromarray(imageData).show()
# sigma = 3.0
# # apply Gaussian blur, creating a new image
# blurred = skimage.filters.gaussian(
#     imageData, sigma=(sigma, sigma), truncate=3.5, channel_axis=2)
# blurred = (255*blurred).astype(np.uint8)
# Image.fromarray(blurred).show()
