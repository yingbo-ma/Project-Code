# Step 1: Parse the input audio input into 1s audio frames
print("Audio parsing...")
import os
from pydub import AudioSegment
from pydub.utils import make_chunks

AudioPath = r"C:\Users\Yingbo\Desktop\TestModel\TestData\AudioFile\1_min_length.wav"

af = os.path.abspath(os.path.dirname(__file__)) + "_AudioFrames"
si = os.path.abspath(os.path.dirname(__file__)) + "_SpectralImages"
os.mkdir(af)
os.mkdir(si)

myaudio = AudioSegment.from_file(AudioPath)
chunk_length_ms = 1000  # pydub calculates in millisec
chunks = make_chunks(myaudio, chunk_length_ms)  # Make chunks of one sec

for i, chunk in enumerate(chunks):  # Export all of the individual chunks as wav files
    chunk_name = "{0}.wav".format(i)
    AudioFramePath = af + "/" + chunk_name
    chunk.export(AudioFramePath, format="wav")

print("Done!")
# Step 2: Get the spectral images of audio frames, using Fourier Transform
print("Generating Spectral Images...")
import re
import matplotlib.pyplot as plt
import librosa.display
from natsort import natsorted

af_files = os.listdir(af)
af_files = natsorted(af_files)

for i in af_files:
    audio_path = os.path.join(af, i)

    (x, sr) = librosa.load(audio_path)
    X = librosa.stft(x)
    Xdb = librosa.amplitude_to_db(abs(X))

    plt.figure(1)
    librosa.display.specshow(Xdb, x_axis='time', y_axis='log')
    plt.axis("off")
    temp = re.findall(r'\d+', i)
    num = int(temp[0])
    # print("Saving STFT image %d.wav .." % (num))
    plt.savefig(si + "/" + "%d.jpg" % (num), bbox_inches='tight', pad_inches=0)

print("Done!")

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

LabelPath = r"C:\Users\Yingbo\Desktop\TestModel\TestData\LabelFile\1_min_length.xlsx"

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

import time
time_stop = time.process_time()
print('Total running time: ', time_stop)