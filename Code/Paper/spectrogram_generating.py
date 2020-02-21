import os
import re
import matplotlib.pyplot as plt
import librosa.display
from natsort import natsorted

folder_path = r"D:\Data\Data_NC_State\TU405-6B\AudioClips\19"
files = os.listdir(folder_path)
files = natsorted(files)

for i in files:
    audio_path = os.path.join(folder_path, i)

    (x, sr) = librosa.load(audio_path)
    X = librosa.stft(x)
    Xdb = librosa.amplitude_to_db(abs(X))

    plt.figure(1)
    librosa.display.specshow(Xdb, x_axis='time', y_axis='log')
    plt.axis("off")

    temp = re.findall(r'\d+', i)
    num = int(temp[0])

    print("Saving STFT image %d.wav .." % (num))
    plt.savefig(r"D:\Data\Data_NC_State\TU405-6B\Image_Data\%d.jpg" % (num), bbox_inches='tight',
                pad_inches=0)