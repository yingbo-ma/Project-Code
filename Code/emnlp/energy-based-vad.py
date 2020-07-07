import math
import numpy as np
import pylab as pl
from scipy.io import wavfile

audio_path = r"C:\Users\Yingbo\Desktop\test3\4\4.wav"

## read wav file ##

(sample_rate, sig) = wavfile.read(audio_path)
wave_data = sig[:, 0]

# calculate zero-crossing-rate ##

def ZeroCR(waveData,frameSize,overLap):
    wlen = len(waveData)
    step = frameSize - overLap
    frameNum = math.ceil(wlen/step)
    zcr = np.zeros((frameNum,1))
    for i in range(frameNum):
        curFrame = waveData[np.arange(i*step,min(i*step+frameSize,wlen))]

        curFrame = curFrame - np.mean(curFrame) # zero-justified
        zcr[i] = sum(curFrame[0:-1]*curFrame[1::]<=0)
    return zcr, wlen, step, frameNum


(sample_rate, sig) = wavfile.read(audio_path)
wave_data = sig[:, 0]

frameSize = 8000
overLap = 0
zcr, a, b, c = ZeroCR(wave_data,frameSize,overLap)

print(zcr)
print(len(zcr))

# plot the wave
time = np.arange(0, len(wave_data)) * (1.0 / sample_rate)
time2 = np.arange(0, len(zcr)) * (len(wave_data)/len(zcr) / sample_rate)
pl.subplot(211)
pl.plot(time, wave_data)
pl.ylabel("Amplitude")
pl.subplot(212)
pl.plot(time2, zcr)
pl.ylabel("ZCR")
pl.xlabel("time (seconds)")
pl.show()

## calculate short-term-energy ##

def calEnergy(wave_data):
    energy = []
    sum = 0
    for i in range(len(wave_data)) :
        sum = sum + (int(wave_data[i]) * int(wave_data[i]))
        if (i + 1) % frameSize == 0 :
            energy.append(sum)
            sum = 0
        elif i == len(wave_data) - 1 :
            energy.append(sum)
    return energy

E = calEnergy(wave_data)


def normalizeData(minRange, maxRange, list):
    dataRange = max(list) - min(list)

    for i in range(len(list)):
        list[i] = (list[i] - min(list)) / dataRange

    rangeNew = maxRange - minRange

    for i in range(len(list)):
        list[i] = (list[i] * rangeNew) + minRange

    return list

ste = normalizeData(0, 300, E)

pl.plot(ste)
pl.show()
print(ste)

final_decision = []

for i in range(len(zcr)):
    if zcr[i] < 1000: # 50
        decision = 1
        final_decision.append(decision)
    else:
        decision = 0
        final_decision.append(decision)

print(final_decision)

for j in range(len(ste)):
    if ste[j] > 10: # 3
        final_decision[j] = 1
    else:
        final_decision[j] = 0

print(final_decision)

################################################################

import xlrd
import numpy as np
from sklearn.metrics import classification_report

label_path = r"C:\Users\Yingbo\Desktop\test3\4\label.xlsx"

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

label_list = excel_data(label_path)
print(classification_report(label_list, final_decision))





