import sys
sys.path.insert(0, ".")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from fit_planes import load_merged_cloud, declutter

cache = Path("_topdown_cache.npy")
if cache.exists():
    xyz = np.load(cache)
else:
    bag = Path(r"C:\Users\loren\Desktop\Dati_vfinal\SLAM\Lidar\rosbag2_2026_07_30-17_50_45")
    xyz = load_merged_cloud(bag, "/cloud_registered", "ROS2_HUMBLE")
    xyz = declutter(xyz, gap=0.30)
    np.save(cache, xyz)

# top-down (XY) density, gridded so it renders fast regardless of point count
xbins = np.linspace(xyz[:,0].min(), xyz[:,0].max(), 400)
ybins = np.linspace(xyz[:,1].min(), xyz[:,1].max(), 300)

from matplotlib.colors import LogNorm
fig, ax = plt.subplots(figsize=(14, 10))
h = ax.hist2d(xyz[:,0], xyz[:,1], bins=[xbins, ybins], cmap="viridis", norm=LogNorm())
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_title("Vista dall'alto (X-Y), nuvola RAW /cloud_registered, decluttered")
ax.set_aspect("equal")
ax.grid(True, alpha=0.3, color="white")
# gridlines every 5m to make ROI picking easy
ax.set_xticks(np.arange(np.floor(xyz[:,0].min()/5)*5, np.ceil(xyz[:,0].max()/5)*5+1, 5))
ax.set_yticks(np.arange(np.floor(xyz[:,1].min()/5)*5, np.ceil(xyz[:,1].max()/5)*5+1, 5))
plt.xticks(rotation=90, fontsize=7)
plt.yticks(fontsize=7)
plt.tight_layout()
plt.savefig("_topdown.png", dpi=130)
print(f"X range: {xyz[:,0].min():.2f} .. {xyz[:,0].max():.2f}")
print(f"Y range: {xyz[:,1].min():.2f} .. {xyz[:,1].max():.2f}")
print(f"Z range: {xyz[:,2].min():.2f} .. {xyz[:,2].max():.2f}")
print("saved _topdown.png")
