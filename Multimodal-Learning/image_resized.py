import os
from PIL import Image

image_data_path = r"C:\Users\Yingbo\Desktop"
path = os.path.join(image_data_path + r"\original images" + r"\Vign_VeyestoJ_Vopensmile_Vfingerup.jpg")
image = Image.open(path).resize((1920, 1080), Image.ANTIALIAS)
image.save(image_data_path + r"\resized images" + r"\Vign_VeyestoJ_Vopensmile_Vfingerup.jpg")