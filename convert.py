from PIL import Image
import numpy as np

image = Image.open("anita-austvika-g0bZhMHJiII-unsplash-square.jpg")
image.convert(mode="L").resize((1000, 1000)).save("original4.png")
# image.save("pexels-kwnos-iv-13216693.png")