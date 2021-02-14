import numpy as np
import matplotlib.pyplot as plt

from support_functions import feature_NW, feature_NU

TimeStep_Corpus = np.load("timestep.npy")
Speakers_Corpus = np.load("speaker.npy")
Text_Corpus = np.load("text.npy")

# # extract NW feature
# number_of_words = feature_NW(Text_Corpus)
#
# # extract NU feature
# utter_spok = feature_NU(Speakers_Corpus)
#
# # prepare speaker-turn target lsit
# speaker_turn = []
#
# for index in range(len(Speakers_Corpus)):
#     if index == len((Speakers_Corpus)) - 1:
#         speaker_turn.append(0)
#     elif Speakers_Corpus[index] == Speakers_Corpus[index+1]:
#         speaker_turn.append(0)
#     else:
#         speaker_turn.append(1)
#
# print("feature: number_of_words \n", number_of_words)
# print("feature: utterance of spoken \n", utter_spok)
# print("target: speaker turn \n",speaker_turn)
# #
# print(np.mean(number_of_words))
# print(np.std(number_of_words))
#
# print(np.mean(utter_spok))
# print(np.std(utter_spok))

print(TimeStep_Corpus)