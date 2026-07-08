import numpy as np

# Exact path using forward slashes
img_path = "C:/bcosta/DS-InSAR/20210801_20210813/20210801_20210813_ifg.data/coh_IW3_VV_01Aug2021_13Aug2021.img"

data_before = np.fromfile(img_path, dtype=np.float32)
print("Shape:", data_before.shape)
print("Min/Max:", round(float(np.nanmin(data_before)), 4), round(float(np.nanmax(data_before)), 4))
print("NaN count before:", int(np.isnan(data_before).sum()))
print("Total pixels:", data_before.size)

np.save("before.npy", data_before)
print("Snapshot saved to before.npy")
