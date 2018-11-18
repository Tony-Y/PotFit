from scipy.special import yn
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(12345)

### Data
n = 100
s = 50
sample_x = np.random.random_sample(n)
sample_v = [yn(0, s * x) + yn(0, s * (1-x)) for x in sample_x]
sample_dv = [-yn(1, s * x) + yn(1, s * (1-x)) for x in sample_x]

### Plot
d = 1e-3
x = np.arange(0+d, 1, d)
# print(x[0])
# print(x[-1])
# print(len(x))

y = yn(0, s * x)
y = y + np.flip(y)

#dy = - s * yn(1, s * x)
dy = - yn(1, s * x)
dy = dy - np.flip(dy)

plt.plot(x, y, label='V(x)')
plt.plot(sample_x, sample_v, 'o')
plt.plot(x, dy, label='dV(x) (scaled)')
plt.plot(sample_x, sample_dv, 'o')
plt.axis((0, 1, -1, 1))
plt.xlabel('x')
plt.ylabel('Potential')
plt.legend()
plt.grid()
plt.show()

### Data output
sample_v = np.array(sample_v)
sample_dv = s * np.array(sample_dv)

np.save('data.npy', [sample_x, sample_v, sample_dv])
