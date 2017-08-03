import numpy as np
from matplotlib import pyplot as plt


history = np.load("history_check.npy")
plt.ion()
f = plt.figure()
print(history.shape)
for time, hist in enumerate(history):
    vision = hist[2]
    print(hist)
    plt.plot(np.arange(len(vision)), vision)
    plt.show()
