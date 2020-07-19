import speech_recognition as sr

r = sr.Recognizer()

test = sr.AudioFile("/home/yingbo/Desktop/1.wav")
with test as source:
    r.adjust_for_ambient_noise(source)
    audio1 = r.record(source)

print(r.recognize_google(audio1))
# print(r.recognize_google(audio2))