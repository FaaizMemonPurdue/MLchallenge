import numpy as np
import pickle

print('here')
a = np.random.rand(10000, 224, 224, 3)
print("ba")
with open("test.pkl", 'wb') as f:
    pickle.dump((a), f)
print('bas')