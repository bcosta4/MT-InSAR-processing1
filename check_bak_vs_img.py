import numpy as np
import os

# Correct path: yyyymmdd_yyyymmdd_ifg.data
bak_path = "C:/bcosta/DS-InSAR/20190601_20190613/20190601_20190613_ifg.data/coh_IW3_VV_01Jun2019_13Jun2019.img.bak"
img_path  = "C:/bcosta/DS-InSAR/20190601_20190613/20190601_20190613_ifg.data/coh_IW3_VV_01Jun2019_13Jun2019.img"

original = np.fromfile(bak_path, dtype=np.float32)
masked   = np.fromfile(img_path,  dtype=np.float32)

print("=== Shape ===")
print("Original (.bak):", original.shape)
print("Masked   (.img):", masked.shape)
print("MATCH:", original.shape == masked.shape)

print("\n=== NaN count ===")
print("NaNs in original:", int(np.isnan(original).sum()))
print("NaNs in masked:  ", int(np.isnan(masked).sum()))
print("New NaNs (lakes):", int(np.isnan(masked).sum() - np.isnan(original).sum()))

print("\n=== Value ranges ===")
print("Original  min/max:", round(float(np.nanmin(original)), 5), "/", round(float(np.nanmax(original)), 5))
print("Masked    min/max:", round(float(np.nanmin(masked)), 5),   "/", round(float(np.nanmax(masked)), 5))

print("\n=== Value preservation at non-lake pixels ===")
valid = ~np.isnan(masked)
diff = np.abs(masked[valid].astype(np.float64) - original[valid].astype(np.float64))
print("Max diff:", float(diff.max()))
print("Mean diff:", float(diff.mean()))

print("\n=== Sanity: coherence should be 0-1 ===")
valid_vals = masked[valid]
print("Values outside [0-1]:", int(((valid_vals < 0) | (valid_vals > 1)).sum()))
