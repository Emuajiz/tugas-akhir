from PIL import Image
import numpy as np


def readImage(filename):
    image = Image.open(filename)
    return np.array(image).astype(dtype=np.uint8)


def createSubBlock(data, size):
    print(data.shape)
    assert data.shape[0] % size == 0
    assert data.shape[1] % size == 0
    x = data.shape[0] // size
    y = data.shape[1] // size
    res = np.zeros((x * y, size, size), dtype=np.uint8)
    counter = 0
    for i in range(x):
        for j in range(y):
            posx = i*size+size
            posy = j*size+size
            res[counter] = data[i*size:posx, j*size:posy]
            counter = counter + 1
    return res


if __name__ == "__main__":
    img = readImage("test1-marked.png")
    subBlock = createSubBlock(img, 2)
    print(subBlock)
