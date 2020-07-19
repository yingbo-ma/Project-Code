import time
import os
from natsort import natsorted

si = os.path.abspath(os.path.dirname(__file__)) + "_SpectralImages"

# Step 3: Prepare the test data
print("Loading testing data...")
from PIL import Image
import numpy as np

PIXEL = 64
IMAGE_CHANNELS = 3

si_files = os.listdir(si)
si_files = natsorted(si_files)

X_test = []

for i in si_files:
    image_path = os.path.join(si, i)
    image = Image.open(image_path).resize((PIXEL, PIXEL), Image.ANTIALIAS)
    X_test.append(np.asarray(image))

X_test = np.reshape(X_test, (-1, PIXEL, PIXEL, IMAGE_CHANNELS))

LabelPath = r"C:\Users\Yingbo\Desktop\TestData\LabelFile\1_min_length.xlsx"

import xlrd

def excel_data(file_path):
    data = xlrd.open_workbook(file_path)
    table = data.sheet_by_index(0)

    nrows = table.nrows
    ncols = table.ncols

    excel_list = []
    for row in range(0, nrows):
        for col in range(ncols):
            cell_value = int(table.cell(row, col).value)
            excel_list.append(cell_value)
    return excel_list

list = excel_data(LabelPath)
y_target = []
for i, j in enumerate(list):
        y_target.append(j)

y_target = np.asarray(y_target)

print("Done!")
# Step 4: load the model and start predicting
print("Loading Speech & Silence detecting model...")

from keras.models import load_model
from sklearn.metrics import classification_report

model = load_model('model.h5')
model.summary()

print("Starting analysing...")
y_pred = model.predict(X_test, batch_size=60, verbose=0)

pred_list = y_pred.tolist()

for i in range(len(pred_list)):
    if pred_list[i] > [0.5]:
        pred_list[i] = [1]
    else:
        pred_list[i] = [0]

y_pred = np.asarray(pred_list)

print("Prediction result")
print(classification_report(y_target, y_pred))

time_stop = time.process_time()
print('Total running time: ', time_stop)