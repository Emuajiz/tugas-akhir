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
    assert psnr(upside[0], leftside[-1])
    assert psnr(rightside[0], upside[-1])
    assert psnr(bottomside[0], rightside[-1])
    assert psnr(leftside[0], bottomside[-1])

    return insidePart, np.array([leftside, upside, rightside, bottomside])


def mergeImage(insidePart, outsidePart):
    imageSize = (insidePart.shape[0] + 2*outsidePart.shape[2],
                 insidePart.shape[1] + 2*outsidePart.shape[3])
    print(imageSize)
    mergedImage = Image.fromarray(insidePart)
    mergedImage.save("merged.png")
    return mergedImage


def imageHash(imageData):
    h = hashlib.new('sha256')
    for i in imageData:
        for j in i:
            h.update(j // 2)
    return h.hexdigest()


# print(psnr(imageTestA, imageTestB))
imageData = readImage("original.png")
# print(imageData[0][0] // 2)

# imageHashDigest = imageHash(imageData)
# print(imageHashDigest)

imageDataInside, imageDataOutside = splitImage(imageData, 32)
# print(imageDataOutside.shape)

mergedImageData = mergeImage(imageDataInside, imageDataOutside)
