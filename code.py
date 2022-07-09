import hashlib
import math
from PIL import Image
import numpy as np


def readImage(filename):
    image = Image.open(filename)
    if image.mode != "L":
        image.convert("L")
    return np.array(image)


def splitImage(imageData, outsideWidth=32):
    imageSize = imageData.shape
    # ensure image size is square
    # assert imageSize[0] == imageSize[1]
    insidePart = imageData[outsideWidth:imageSize[1] -
                           outsideWidth, outsideWidth:imageSize[0] - outsideWidth]
    # print(insidePart.shape)
    outsidePart = []
    Image.fromarray(insidePart).save("inside.png")
    for i in range(0, imageSize[0], outsideWidth):
        # upside
        imageTemp = np.array(imageData[0:outsideWidth, i:i+outsideWidth])
        # print(imageTemp.shape)
        Image.fromarray(imageTemp).save(
            f"upside/outside-(({i},0),({i+outsideWidth},{outsideWidth})).png")
        outsidePart.append(imageTemp)

        # rightside
        imageTemp = np.array(
            imageData[i:i+outsideWidth, imageSize[1] - outsideWidth:imageSize[1]])
        # print(imageTemp.shape)
        Image.fromarray(imageTemp).save(
            f"rightside/outside-(({imageSize[1]-outsideWidth},{i}),({imageSize[1]},{i+outsideWidth})).png")
        outsidePart.append(imageTemp)

        # bottomside
        imageTemp = np.array(
            imageData[imageSize[0] - outsideWidth:imageSize[0], imageSize[1] - i - outsideWidth:imageSize[1] - i])
        # print(imageTemp.shape)
        Image.fromarray(imageTemp).save(
            f"bottomside/outside-(({imageSize[1] - i - outsideWidth},{imageSize[0] - outsideWidth}),({imageSize[1] - i},{imageSize[0]})).png")
        outsidePart.append(imageTemp)

        # leftside
        imageTemp = np.array(
            imageData[imageSize[0] - i - outsideWidth:imageSize[0] - i, 0:outsideWidth])
        # print(imageTemp.shape)
        Image.fromarray(imageTemp).save(
            f"leftside/outside-(({0},{imageSize[0] - i - outsideWidth}),({outsideWidth},{imageSize[0] - i})).png")
        outsidePart.append(imageTemp)

    return insidePart, outsidePart


def imageHash(imageData):
    h = hashlib.new('sha256')
    for i in imageData:
        for j in i:
            h.update(j // 2)
    return h.hexdigest()


def psnr(A, B):
    mse = np.mean((A - B) ** 2)
    if(mse == 0):
        return 100
    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr


# print(psnr(imageTestA, imageTestB))
imageData = readImage("original.png")
# print(imageData[0][0] // 2)

# imageHashDigest = imageHash(imageData)
# print(imageHashDigest)

imageDataInside, imageDataOutside = splitImage(imageData, 32)

imageTestA = readImage("upside/outside-((0,0),(32,32)).png")
imageTestB = readImage("leftside/outside-((0,0),(32,32)).png")
assert psnr(imageTestA, imageTestB) == 100

imageTestA = readImage("upside/outside-((480,0),(512,32)).png")
imageTestB = readImage("rightside/outside-((480,0),(512,32)).png")
assert psnr(imageTestA, imageTestB) == 100

imageTestA = readImage("bottomside/outside-((480,480),(512,512)).png")
imageTestB = readImage("rightside/outside-((480,480),(512,512)).png")
assert psnr(imageTestA, imageTestB) == 100

imageTestA = readImage("bottomside/outside-((0,480),(32,512)).png")
imageTestB = readImage("leftside/outside-((0,480),(32,512)).png")
assert psnr(imageTestA, imageTestB) == 100
