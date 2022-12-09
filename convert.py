from PIL import Image

image = Image.open("anita-austvika-g0bZhMHJiII-unsplash-square.jpg")
image.convert(mode="L").resize((1000, 1000)).save("original1.png")
# image.save("pexels-kwnos-iv-13216693.png")