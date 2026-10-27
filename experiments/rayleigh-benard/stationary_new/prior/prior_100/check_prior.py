import numpy as np
field = np.load("prior_fields.npz")
print(np.max(field["prior"]))