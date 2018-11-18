from scipy.special import yn
import matplotlib.pyplot as plt
import numpy as np
import sys

filename = 'predictions_{}.npy'.format(sys.argv[1])

x, p = np.load(filename)

s = 50
y = yn(0, s * x)
y = y + np.flip(y)

plt.plot(x, y, label='V(x)')
plt.plot(x, p, label='P(x)')
plt.axis((0, 1, -1, 1))
plt.xlabel('x')
plt.ylabel('Potential')
plt.legend()
plt.grid()
plt.show()
