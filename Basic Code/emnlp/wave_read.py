import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

(sample_rate, sig) = wavfile.read(r"C:\Users\Yingbo\Desktop\cnns\1.wav")
sample_points = sig.shape[0]
print("sample_rate: %d" % sample_rate)
print("sample_points: %d" % sample_points)
print("time_last = sample_points/sample_rate")

data_0 = sig[:, 0]
time = np.arange(0, sample_points) * (1 / sample_rate)

fig = plt.figure('Figure1', figsize=(16,9)).add_subplot(111)
fig.plot(time, data_0, "b")
# plt.gca().axes.get_yaxis().set_visible(False)
fig.set_xlabel("Time (s)", fontsize=14)
fig.set_ylabel("Amplitude", fontsize=14)

fig.xaxis.get_label().set_fontsize(20)
# plt.axis("off")
plt.show()
plt.gca().axes.yaxis.set_ticklabels([])
