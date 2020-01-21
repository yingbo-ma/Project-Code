import numpy as np

List = [1, 2, 3, 4, 5]
ix = np.random.randint(0, len(List), 3)
print(ix)
ix = ix.tolist()
print(ix)
print(List[ix])
