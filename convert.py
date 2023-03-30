from PIL import Image

image = Image.open("peter-thomas-hM0VjEdNOuA-unsplash-square.jpg")
image.convert(mode="RGB").resize((500, 500)).save("test-500x500.png")
# image.save("pexels-kwnos-iv-13216693.png")