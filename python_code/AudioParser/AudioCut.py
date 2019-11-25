from pydub import AudioSegment
from pydub.utils import make_chunks

audio_path = "/home/yingbo/Desktop/LD2_PKYonge_Class1_Mar142019_B.wav"

myaudio = AudioSegment.from_file(audio_path , "wav") 
chunk_length_ms = 1000 # pydub calculates in millisec
chunks = make_chunks(myaudio, chunk_length_ms) #Make chunks of one sec

#Export all of the individual chunks as wav files

for i, chunk in enumerate(chunks):
    chunk_name = "{0}.wav".format(i)
    print("exporting", chunk_name)
    chunk.export(chunk_name, format="wav")