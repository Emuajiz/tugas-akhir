import cmath
import hashlib
import math
import random
from PIL import Image
import numpy as np
import skimage


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
    if image.mode != "L":
        image = image.convert("L")
    return np.array(image)


def validateInsideImageData(insideImagedata):
    """helper to validate inside image data using overlapping at image corner data"""
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
        imageData[-val.shape[0]:, i*val.shape[1]:(i+1)*val.shape[1]] = outsidePart[2][-(i+1)]

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
    magnitude = np.pad(magnitude, ((1,1),(1,1)))
    tmp2 = (position[0]+1, position[1]+1)
    for i in range(-1, 2, 1):
        for j in range(-1, 2, 1):
            tmp += magnitude[tmp2[0]+i][tmp2[1]+j]
    return tmp / 9


def embedRobustWatermark(imageData, watermarkData, alpha=10, radius=-1):
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
    extractedWatermarkData = []
    watermarkElement = np.zeros(len(watermarkData), dtype=np.float64)
    for i, val in enumerate(indices):
        watermarkElement[i] = calculateForWatermark(mag, val)
    for i, val in enumerate(indices):
        tmp = (mag[val[0], val[1]] - originalMag[val[0], val[1]]) / \
            (alpha * watermarkElement[i])
        extractedWatermarkData.append(tmp)
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
    assert validateInsideImageData(
        [upsideDataArray, rightsideDataArray, bottomsideDataArray, leftsideDataArray]) == True

    return [upsideDataArray, rightsideDataArray, bottomsideDataArray, leftsideDataArray]


def checkBitRobustWatermark(extracted, original, bitPerPart):
    normExtracted = normalize(extracted, 0, 1)
    bitExtracted = np.where(np.array(normExtracted) > 0.5, 1, 0)
    similarityArray = np.zeros(bitPerPart, dtype=np.uint8)
    for i in range(bitPerPart):
        if bitExtracted[i] == original[i]:
            similarityArray[i] = 1
    return similarityArray


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

    print(watermarkCheck)

    return np.average(watermarkCheck)


def multipleWatermark(imageData, password, outsideImageSize=(32, 32), factor=10, show=False, save=False, out="out.png", bitPerPart=8, radius=-1):
    # embed watermark
    insideImageData, outsideImageData = splitImage(imageData, outsideImageSize)
    watermarkedOutsideImageData = processEmbedRobustWatermark(
        outsideImageData, password, factor, bitPerPart, radius)
    watermarkedInsideImageData = embedFragileWatermark(insideImageData)
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


def extractMultipleWatermark(imageData, originalImageData, password, outsideImageSize=(32, 32), factor=10, bitPerPart=8, radius=-1):
    insideWatermark, outsideWatermark = splitImage(imageData, outsideImageSize)
    _, originalOutsideWatermark = splitImage(
        originalImageData, outsideImageSize)
    # fragile
    extractedFragile = extractFragileWatermark(insideWatermark)
    # robust
    robustCheckResult = processExtractRobustWatermark(
        outsideWatermark, originalOutsideWatermark, password, factor, bitPerPart, radius)

    print(extractedFragile)
    print(robustCheckResult)


def splitThenMergeShouldReturnSameImage(filename):
    imageData = readImage(filename)
    insideImageData, outsideImageData = splitImage(imageData, (32, 32))
    merged = mergeImage(insideImageData, outsideImageData)
    return psnr(imageData, merged) == 100


assert splitThenMergeShouldReturnSameImage("original.png") == True

imageData = readImage("original.png")
watermarked = multipleWatermark(
    imageData, "thor", (32, 32), 10, False, False, "watermarked.png", 8, 8)
extractMultipleWatermark(watermarked, imageData, "thor", (32, 32), 10, 8, 8)

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
