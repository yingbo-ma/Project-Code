import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

(sample_rate, sig) = wavfile.read(r"D:\Data\LD2_PKYonge_Class1_Mar142019_B\AudioClips\LD2_PKYonge_Class1_Mar142019_B\22\2299.wav")
# (sample_rate, sig) = wavfile.read(r"D:\Data\LD2_PKYonge_Class1_Mar142019_B\Audio\LD2_PKYonge_Class1_Mar142019_B.wav")
sample_points = sig.shape[0]
print("sample_rate: %d" % sample_rate)
print("sample_points: %d" % sample_points)
print("time_last = sample_points/sample_rate")

data_0 = sig[:, 0]
time = np.arange(0, sample_points) * (1 / sample_rate)
ix = np.random.randint(0, len(data_0), 500)
audio = data_0[ix]

fig = plt.figure('Figure1').add_subplot(111)
fig.plot(time, data_0/1000)
fig.set_xlabel("Time", fontsize = 10)
fig.set_ylabel("Amplitude", fontsize = 10)
# plt.tick_params(labelsize=14)
# plt.axis("off")

plt.show()
