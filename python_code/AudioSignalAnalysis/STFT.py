import matplotlib.pyplot as plt
import numpy as np
import librosa.display

audio_path = "/home/yingbo/FLECKS/Data/AudioClips/LD2_PKYonge_Class1_Mar142019_B/25/2500.wav"

(x, sr) = librosa.load(audio_path)

X = librosa.stft(x)
Xdb = librosa.amplitude_to_db(abs(X))
print(X.shape)

plt.figure(1)
librosa.display.specshow(Xdb, x_axis='time', y_axis='log')
# plt.axis("off")
plt.colorbar(format='%+2.0f dB')
# plt.savefig('image.jpg', bbox_inches='tight', pad_inches=0)
plt.show()