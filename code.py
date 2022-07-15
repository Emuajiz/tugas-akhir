import cmath
import hashlib
import math
from PIL import Image
import numpy as np


def psnr(A, B):
    mse = np.mean((A - B) ** 2)
    if(mse == 0):
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
    # print(insidePart.shape)

    leftside = []
    upside = []
    rightside = []
    bottomside = []
    # Image.fromarray(insidePart).save("inside.png")
    for i in range(0, imageSize[1], outsideWidth):
        # upside
        imageTemp = np.array(imageData[0:outsideWidth, i:i+outsideWidth])
        # print(imageTemp.shape)
        # Image.fromarray(imageTemp).save(
        #     f"upside/outside-(({i},0),({i+outsideWidth},{outsideWidth})).png")
        # Image.fromarray(imageTemp).save(f"upside/{i}.png")
        upside.append(imageTemp)

        # bottomside
        imageTemp = np.array(
            imageData[imageSize[0] - outsideWidth:imageSize[0], imageSize[1] - i - outsideWidth:imageSize[1] - i])
        # print(imageTemp.shape)
        # Image.fromarray(imageTemp).save(
        #     f"bottomside/outside-(({imageSize[1] - i - outsideWidth},{imageSize[0] - outsideWidth}),({imageSize[1] - i},{imageSize[0]})).png")
        # Image.fromarray(imageTemp).save(f"bottomside/{i}.png")
        bottomside.append(imageTemp)

    for i in range(0, imageSize[0], outsideWidth):
        # rightside
        imageTemp = np.array(
            imageData[i:i+outsideWidth, imageSize[1] - outsideWidth:imageSize[1]])
        # print(imageTemp.shape)
        # Image.fromarray(imageTemp).save(
        #     f"rightside/outside-(({imageSize[1]-outsideWidth},{i}),({imageSize[1]},{i+outsideWidth})).png")
        # Image.fromarray(imageTemp).save(f"rightside/{i}.png")
        rightside.append(imageTemp)

        # leftside
        imageTemp = np.array(
            imageData[imageSize[0] - i - outsideWidth:imageSize[0] - i, 0:outsideWidth])
        # print(imageTemp.shape)
        # Image.fromarray(imageTemp).save(
        #     f"leftside/outside-(({0},{imageSize[0] - i - outsideWidth}),({outsideWidth},{imageSize[0] - i})).png")
        # Image.fromarray(imageTemp).save(f"leftside/{i}.png")
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
    # print(imageSize)
    imageData = np.zeros(shape=imageSize, dtype=np.uint8)
    imageData[outsidePart.shape[2]:insidePart.shape[0] +
              outsidePart.shape[2], outsidePart.shape[3]:insidePart.shape[1] + outsidePart.shape[3]] = insidePart

    for i, val in enumerate(outsidePart[0]):
        # upside
        imageData[0:val.shape[0], i*val.shape[1]:(i+1)*val.shape[1]] = val
        imageData[-val.shape[0]:, i*val.shape[1]
            :(i+1)*val.shape[1]] = outsidePart[2][-(i+1)]
    # bottomside
    for i, val in enumerate(outsidePart[1]):
        # rightside
        imageData[i*val.shape[0]:(i+1)*val.shape[0], -val.shape[1]:] = val
        # leftside
        imageData[i*val.shape[0]:(i+1)*val.shape[0],
                  :val.shape[1]] = outsidePart[3][-(i+1)]
    mergedImage = Image.fromarray(imageData)
    # print(mergedImage.mode)
    # mergedImage.show()
    mergedImage.save("merged.png")
    return imageData


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
            if(j % 2 == int(fragileWatermarkPayload[counter])):
                fragileWatermark[y, x] = 255
            else:
                fragileWatermark[y, x] = 0
            counter += 1
            counter %= len(fragileWatermarkPayload)
    Image.fromarray(fragileWatermark).show()
    return fragileWatermark


def embedRobustWatermark(imageData, watermarkData, alpha=0.1):
    watermarkedImageData = np.zeros(imageData.shape, dtype=np.uint8)
    normalizeWatermarkData = watermarkData / 255
    imageFourier = np.fft.fftshift(np.fft.fft2(imageData, norm="ortho"))
    print(np.array_equal(
        np.abs(np.fft.ifft2(np.fft.fft2(imageData)).astype(np.uint8)), imageData))
    print(psnr(np.abs(np.fft.ifft2(np.fft.fft2(imageData)).astype(np.uint8)), imageData))
    mag = np.abs(imageFourier)
    phase = np.angle(imageFourier)
    start = [val//2 - watermarkData.shape[i] //
             2 for i, val in enumerate(mag.shape)]
    for i in range(watermarkData.shape[0]):
        for j in range(watermarkData.shape[1]):
            mag[start[0]+i, start[1]+j] = mag[start[0]+i,
                                              start[1]+j] * (1 + alpha*normalizeWatermarkData[i][j])
    watermarkedImageFourier = [[mag[i][j] * cmath.exp(1j * phase[i][j]) for j in range(
        imageFourier.shape[1])] for i in range(imageFourier.shape[0])]
    # print(watermarkedImageData)
    watermarkedImageData = np.round(
        abs(np.fft.ifft2(watermarkedImageFourier, norm="ortho"))).astype(np.uint8)
    image = Image.fromarray(imageData)
    image.show("a")
    watermarkedImage = Image.fromarray(watermarkedImageData)
    watermarkedImage.show("b")
    # print(psnr(imageData, watermarkedImageData))
    return watermarkedImageData


def extractRobustWatermark(imageData, originalImageData, watermarkData, alpha=0.1):
    extractedWatermarkData = np.zeros(watermarkData.shape, dtype=np.uint8)
    imageFourier = np.fft.fftshift(np.fft.fft2(imageData, norm="ortho"))
    originalImageFourier = np.fft.fftshift(
        np.fft.fft2(originalImageData, norm="ortho"))
    mag = np.abs(imageFourier)
    originalMag = np.abs(originalImageFourier)
    phase = np.angle(imageFourier)
    start = [val//2 - watermarkData.shape[i] //
             2 for i, val in enumerate(mag.shape)]
    for i in range(watermarkData.shape[0]):
        for j in range(watermarkData.shape[1]):
            extractedWatermarkData[i, j] = ((mag[start[0] +
                                                 i][start[1]+j]/originalMag[start[0]+i][start[1]+j] - 1)/alpha)*255
    # print(extractedWatermarkData)
    watermarkedImage = Image.fromarray(extractedWatermarkData)
    watermarkedImage.show()
    # print(psnr(imageData, watermarkedImageData))
    return watermarkData


# print(psnr(imageTestA, imageTestB))
imageData = readImage("original.png")
robustWatermarkData = readImage("fragileWatermark.png")
# print(imageData[0][0] // 2)

# imageHashDigest = imageHash(imageData)
# print(imageHashDigest)

insideImageData, outsideImageData = splitImage(imageData, 32)
# print(imageDataOutside.shape)

watermarkedInsideImageData = embedFragileWatermark(insideImageData)
extractedWatermark = extractFragileWatermark(watermarkedInsideImageData)


# watermarkedOutside = embedRobustWatermark(
#     outsideImageData[0][0], robustWatermarkData, 1)
# for i in range(11):
#     watermarkedOutside = embedRobustWatermark(
#         outsideImageData[0][0], robustWatermarkData, i*0.01)
# extractedWatermark = extractRobustWatermark(
#     watermarkedOutside, outsideImageData[0][0], robustWatermarkData, 1)

mergedImageData = mergeImage(insideImageData, outsideImageData)
