from PIL import Image
import numpy as np

image = Image.open("original2.png")
image.convert(mode="L").crop((0, 1000, 4000, 5000)).resize((1000, 1000)).save("original2-gray.png")
# image.save("pexels-kwnos-iv-13216693.png")