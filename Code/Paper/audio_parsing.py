print("Audio parsing...")
import os
from pydub import AudioSegment
from pydub.utils import make_chunks

AudioPath = r"D:\Data_NC_State\TU409-10B\Audio\TU409-10B.wav"

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
