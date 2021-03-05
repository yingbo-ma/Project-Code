from jiwer import wer

ground_truth = ["I like python and java"]
hypothesis = ["I python and add one java"]

error = wer(ground_truth, hypothesis)

print("The word error rate is : ", error)


