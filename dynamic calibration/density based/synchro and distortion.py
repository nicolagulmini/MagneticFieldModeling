import numpy as np
import matplotlib.pyplot as plt

a = np.loadtxt("./sampled_points.csv")
n = a.shape[0]

plt.plot(range(n), a[:, 0]) 
plt.plot(range(n), [10000*np.linalg.norm(a[i, 6:14]) for i in range(n)])