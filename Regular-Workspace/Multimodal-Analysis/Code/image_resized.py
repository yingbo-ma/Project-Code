import os
from PIL import Image
import numpy as np

GENERATE_SQUARE = 200
IMAGE_CHANNELS = 3

desktop_path = r"C:\Users\Yingbo\Desktop"

clsss_0_original_image_data_path = r"C:\Users\Yingbo\Desktop\New folder\0-other none brick patterns"
clsss_0_resized_image_data_path = r"C:\Users\Yingbo\Desktop\New folder\resized-0"

clsss_0_original_file_list = os.listdir(clsss_0_original_image_data_path)

class_0_data = []

for i in range(len(clsss_0_original_file_list)):
    image_path = os.path.join(clsss_0_original_image_data_path, clsss_0_original_file_list[i])
    image = Image.open(image_path).resize((GENERATE_SQUARE, GENERATE_SQUARE), Image.ANTIALIAS)
    class_0_data.append(np.asarray(image))
    image.save(os.path.join(clsss_0_resized_image_data_path, clsss_0_original_file_list[i]))

print(len(class_0_data))
class_0_data = np.reshape(class_0_data, (-1, GENERATE_SQUARE, GENERATE_SQUARE, IMAGE_CHANNELS))
print(class_0_data.shape)

##########################################################################################################
clsss_1_original_image_data_path = r"C:\Users\Yingbo\Desktop\New folder\1-flemish stretch bond"
clsss_1_resized_image_data_path = r"C:\Users\Yingbo\Desktop\New folder\resized-1"

clsss_1_original_file_list = os.listdir(clsss_1_original_image_data_path)

class_1_data = []

for i in range(len(clsss_1_original_file_list)):
    image_path = os.path.join(clsss_1_original_image_data_path, clsss_1_original_file_list[i])
    image = Image.open(image_path).resize((GENERATE_SQUARE, GENERATE_SQUARE), Image.ANTIALIAS)
    class_1_data.append(np.asarray(image))
    image.save(os.path.join(clsss_1_resized_image_data_path, clsss_1_original_file_list[i]))

print(len(class_1_data))
class_1_data = np.reshape(class_0_data, (-1, GENERATE_SQUARE, GENERATE_SQUARE, IMAGE_CHANNELS))
print(class_1_data.shape)

##########################################################################################################
clsss_2_original_image_data_path = r"C:\Users\Yingbo\Desktop\New folder\2-english bond"
clsss_2_resized_image_data_path = r"C:\Users\Yingbo\Desktop\New folder\resized-2"

clsss_2_original_file_list = os.listdir(clsss_2_original_image_data_path)

class_2_data = []

for i in range(len(clsss_2_original_file_list)):
    image_path = os.path.join(clsss_2_original_image_data_path, clsss_2_original_file_list[i])
    image = Image.open(image_path).resize((GENERATE_SQUARE, GENERATE_SQUARE), Image.ANTIALIAS)
    class_2_data.append(np.asarray(image))
    image.save(os.path.join(clsss_2_resized_image_data_path, clsss_2_original_file_list[i]))

print(len(class_2_data))
class_2_data = np.reshape(class_2_data, (-1, GENERATE_SQUARE, GENERATE_SQUARE, IMAGE_CHANNELS))
print(class_2_data.shape)

##########################################################################################################
clsss_3_original_image_data_path = r"C:\Users\Yingbo\Desktop\New folder\3-strecher bond"
clsss_3_resized_image_data_path = r"C:\Users\Yingbo\Desktop\New folder\resized-3"

clsss_3_original_file_list = os.listdir(clsss_3_original_image_data_path)

class_3_data = []

for i in range(len(clsss_3_original_file_list)):
    image_path = os.path.join(clsss_3_original_image_data_path, clsss_3_original_file_list[i])
    image = Image.open(image_path).resize((GENERATE_SQUARE, GENERATE_SQUARE), Image.ANTIALIAS)
    class_3_data.append(np.asarray(image))
    image.save(os.path.join(clsss_3_resized_image_data_path, clsss_3_original_file_list[i]))

print(len(class_3_data))
class_3_data = np.reshape(class_3_data, (-1, GENERATE_SQUARE, GENERATE_SQUARE, IMAGE_CHANNELS))
print(class_3_data.shape)

##########################################################################################################

Images = np.concatenate((class_0_data, class_1_data, class_2_data, class_3_data), axis=0)
print(Images.shape)

##########################################################################################################
class_0_labels = np.zeros((25, 1))
class_1_labels = np.ones((25, 1))
class_2_labels = 2 * np.ones((25, 1))
class_3_labels = 3 * np.ones((25, 1))

Labels = np.concatenate((class_0_labels, class_1_labels, class_2_labels, class_3_labels), axis=0)
print(Labels)

np.save('Images.npy', Images)
np.save('Labels.npy', Labels)
