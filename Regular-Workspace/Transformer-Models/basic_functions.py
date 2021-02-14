import xlrd
import os
from PIL import Image
import numpy as np

import tensorflow as tf


def read_label_excel(label_file_path):
    data = xlrd.open_workbook(label_file_path)
    table = data.sheet_by_index(0)

    nrows = table.nrows
    ncols = table.ncols

    excel_list = []
    for row in range(0, nrows):
        for col in range(ncols):
            cell_value = int(table.cell(row, col).value)
            excel_list.append(cell_value)
    return excel_list



def read_data(excel_file_path, image_file_path, resized_pixel, num_image_channels):
    # get the length of whole data
    data_size = len(read_label_excel(excel_file_path))

    data = []

    for index in range(data_size):
        path = os.path.join(image_file_path, str(index) + ".jpg")
        image = Image.open(path).resize((resized_pixel, resized_pixel), Image.ANTIALIAS)
        image  = tf.keras.preprocessing.image.img_to_array(image)
        image = tf.cast(image, tf.float32)
        data.append(np.asarray(image))

    data = np.reshape(data, (-1, resized_pixel, resized_pixel, num_image_channels))

    return data



def recursive_data_label_prepare(excel_file_path, image_data_path, num_timesteps, resized_pixel, num_image_channels):
    # read excel list
    origin_target = read_label_excel(excel_file_path)

    # prepare target label
    recursive_target = [[]]
    recursive_index = [[[i + j] for i in range(num_timesteps)] for j in range(len(origin_target) - num_timesteps + 1)]

    for element in recursive_index:
        temp = []
        for index_of_num_timesteps in range(num_timesteps):
            temp.append(origin_target[element[index_of_num_timesteps][0]])
        recursive_target.append(temp)

    recursive_target.remove([])
    recursive_target = np.reshape(recursive_target, (-1, num_timesteps, 1))

    # read image data
    origin_data = read_data(excel_file_path, image_data_path, resized_pixel, num_image_channels)

    # prepare image data
    recursive_data = [[]]

    for element in recursive_index:
        temp = []
        for index_of_num_timesteps in range(num_timesteps):
            temp.append(origin_data[element[index_of_num_timesteps][0]])
        recursive_data.append(temp)

    recursive_data.remove([])
    recursive_data = np.reshape(recursive_data, (-1, num_timesteps, resized_pixel, resized_pixel, num_image_channels))

    return recursive_data, recursive_target
