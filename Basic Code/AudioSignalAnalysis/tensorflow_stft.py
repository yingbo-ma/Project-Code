import tensorflow as tf

audio_file = r'D:\Data\Data_NC_State\TU405-6B\AudioClips\2'
sampling_rate = 44100
audio_binary = tf.read_file(audio_file)
# tf.contrib.ffmpeg not supported on Windows, refer to issue
# https://github.com/tensorflow/tensorflow/issues/8271
waveform = tf.contrib.ffmpeg.decode_audio(audio_binary,
	file_format='wav', samples_per_second=sampling_rate, channel_count=1)
print(waveform.numpy().shape)

