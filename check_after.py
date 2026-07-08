import numpy as np

img_path = "C:/bcosta/DS-InSAR/20190601_20190613/20190601_20190613_ifg.data/coh_IW3_VV_01Jun2019_13Jun2019.img"

data_before = np.load("before.npy")
data_after  = np.fromfile(img_path, dtype=np.float32)

print("=== Shape ===")
print("Before:", data_before.shape)
print("After: ", data_after.shape)
ok = data_before.shape == data_after.shape
print("MATCH:", ok)

print("\n=== NaN count ===")
print("Before:", int(np.isnan(data_before).sum()))
print("After: ", int(np.isnan(data_after).sum()))
print("New NaNs (lake pixels):", int(np.isnan(data_after).sum() - np.isnan(data_before).sum()))

print("\n=== Value preservation (non-lake pixels) ===")
valid = ~np.isnan(data_after)
diff = np.abs(data_after[valid] - data_before[valid])
print("Max diff at valid pixels:", float(diff.max()))
print("Mean diff at valid pixels:", float(diff.mean()))

print("\n=== Output range ===")
print("Min:", round(float(np.nanmin(data_after)), 4))
print("Max:", round(float(np.nanmax(data_after)), 4))
