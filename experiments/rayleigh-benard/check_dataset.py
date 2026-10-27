import h5py
import numpy as np
import matplotlib.pyplot as plt
from utils import destandardize

out = "data_longer/train.h5"
with h5py.File(out, 'r') as f:
    T = f["x"][0,0,0]
    T = destandardize(T, datadir="data_longer")
    print(T[0])
    # print(np.max(T, axis=(0,1,-1,-2)))
    # for i in range(50):
    #     plt.imshow(T[10,i,0], cmap="coolwarm")
    #     plt.savefig(f"data/sample_{i}.png")