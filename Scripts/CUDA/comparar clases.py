import numpy as np
import matplotlib.pyplot as plt

features = np.fromfile("train.bin", dtype=np.float32).reshape(-1, 4096)
labels = np.fromfile("train_labels.bin", dtype=np.int32)

idx_con = np.where(labels == 0)[0][0]  # con mascarilla
idx_sin = np.where(labels == 1)[0][0]  # sin mascarilla

fig, axs = plt.subplots(1, 2, figsize=(8, 4))
axs[0].imshow(features[idx_con].reshape(64, 64), cmap='gray')
axs[0].set_title(f"Con mascarilla (idx {idx_con})")
axs[1].imshow(features[idx_sin].reshape(64, 64), cmap='gray')
axs[1].set_title(f"Sin mascarilla (idx {idx_sin})")
plt.show()