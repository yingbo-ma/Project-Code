from refined_basic_functions import read_excel, spectrogram_data_prepare
import numpy as np
from sklearn.metrics import classification_report

label_path = r"E:\Research Data\Data_UF\Yingbo_LD2_PKYonge_Class1_Mar142019_B\new_label.xlsx"
image_path = r"E:\Research Data\Data_UF\Yingbo_LD2_PKYonge_Class1_Mar142019_B\Image_Data"

PIXEL = 64
IMAGE_CHANNELS = 3
SPLIT_RATIO = 0.75
CLASS_NUM = 3

label_list = read_excel(label_path)
data_0_train, data_1_train, data_2_train, train_data, test_data, test_label, shuffled_list_0, shuffled_list_1, shuffled_list_2 = spectrogram_data_prepare(label_list, image_path, PIXEL, IMAGE_CHANNELS, SPLIT_RATIO)