import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

sample_rate = 64

(sample_rate, sig) = wavfile.read(r"D:\Data\LD2_PKYonge_Class1_Mar142019_B\AudioClips\LD2_PKYonge_Class1_Mar142019_B\2\200.wav")
sample_points = sig.shape[0]
print("sample_rate: %d" % sample_rate)
print("sample_points: %d" % sample_points)
print("time_last = sample_points/sample_rate")

data_0 = sig[:, 0]
time = np.arange(0, sample_points) * (1 / sample_rate)

fig = plt.figure('Figure1', figsize=(4,3)).add_subplot(111)
# fig.plot(time, data_0)
# fig.set_xlabel("time/seconds", fontsize=18)
# fig.set_ylabel("Amplitude", fontsize=18)
# plt.tick_params(labelsize=14)

ix = np.random.randint(0, len(data_0), 64)
y = data_0[ix] / 100
x = [index for index in range(64)]

fig.plot(x, y)
print(x)
print(y)
print(len(x))
print(len(y))

plt.show()
