#%%
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os

noise_path = "/home/yingbo/FLECKS/AudioData/noise/" 
noise_fileList= os.listdir(noise_path)

for i in range(len(noise_fileList)):
    noise_file_path = "/home/yingbo/FLECKS/AudioData/noise/" + noise_fileList[i]
    (fs, noise) = wavfile.read(noise_file_path)
    noise = noise[:, 0]
    f, t, Zxx = signal.stft(noise, fs, nperseg = 1000)
    plt.figure(1)
    plt.subplot(5, 1, i + 1).pcolormesh(t, f, np.abs(Zxx));
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')

plt.show()
#%%
speech_path = "/home/yingbo/FLECKS/AudioData/speech/"
speech_fileList= os.listdir(speech_path)

for k in range(len(speech_fileList)):
    speech_file_path = "/home/yingbo/FLECKS/AudioData/speech/" + speech_fileList[i]
    (fs, speech) = wavfile.read(speech_file_path)
    speech = speech[:, 0]
    f, t, Zxx = signal.stft(speech, fs, nperseg = 1000)
    plt.figure(2)
    plt.subplot(5, 1, k + 1).pcolormesh(t, f, np.abs(Zxx));
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')

plt.show()