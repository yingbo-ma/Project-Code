# load and show an image with Pillow
from PIL import Image
# load the image
image = Image.open("/home/yingbo/FLECKS/FlecksCode/python/AudioSignalAnalysis/image.jpg")
# summarize some details about the image
print(image.format)
print(image.mode)
print(image.size)
# show the image
image.show()