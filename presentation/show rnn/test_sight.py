import numpy as np
from matplotlib import pyplot as plt

x = np.linspace(-180, 180, 72)
y = np.random.rand(72)
f, axarr = plt.subplots(2,figsize=(8, 8), sharex=True)
ax1, ax2 = axarr
ax1.set_title("Vision: other Agents")
x1, x2, y1, y2 = ax1.axis()
ax1.axis((-190, 190, 0, 1))
ax1.bar(x, y, label="Vision", width=6)
ax2.set_title("Vision: Walls")
x1, x2, y1, y2 = ax2.axis()
ax2.axis((-190, 190, 0, 1))
ax2.bar(x, y, label="Vision")
ax2.set_xlabel("Direction")
plt.show()