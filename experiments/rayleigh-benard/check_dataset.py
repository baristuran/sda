import h5py
import numpy as np
import matplotlib.pyplot as plt

out = "data/train.h5"
with h5py.File(out, 'r') as f:
    T = f["x"]
    print(T.shape)
    print(np.max(T, axis=(0,1,-1,-2)))
    for i in range(10):
        plt.imshow(T[i,0,0])
        plt.savefig(f"data/sample_{i}.png")