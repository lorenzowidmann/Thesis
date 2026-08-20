import numpy as np
import open3d as o3d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

pcd = o3d.io.read_point_cloud(
    r"C:\Users\loren\Desktop\Dati_vfinal\SLAM\Lidar\rosbag2_2026_07_30-17_50_45\loop_closed_map.pcd")
xyz = np.asarray(pcd.points)
print(f"{len(xyz)} punti, X {xyz[:,0].min():.2f}..{xyz[:,0].max():.2f}  "
      f"Y {xyz[:,1].min():.2f}..{xyz[:,1].max():.2f}  Z {xyz[:,2].min():.2f}..{xyz[:,2].max():.2f}")

xbins = np.linspace(xyz[:,0].min(), xyz[:,0].max(), 400)
ybins = np.linspace(xyz[:,1].min(), xyz[:,1].max(), 300)

sumZ, _, _ = np.histogram2d(xyz[:,0], xyz[:,1], bins=[xbins, ybins], weights=xyz[:,2])
cnt, _, _ = np.histogram2d(xyz[:,0], xyz[:,1], bins=[xbins, ybins])
with np.errstate(invalid="ignore"):
    meanZ = sumZ / cnt
meanZ = np.ma.masked_invalid(meanZ)

fig, ax = plt.subplots(figsize=(14, 10))
pcm = ax.pcolormesh(xbins, ybins, meanZ.T, cmap="turbo", shading="auto", vmin=-1, vmax=3)
fig.colorbar(pcm, ax=ax, label="Z medio (m)")
ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
ax.set_title("loop_closed_map.pcd (gravita'+yaw+loop closure), vista dall'alto colorata per Z")
ax.set_aspect("equal")
ax.grid(True, alpha=0.3, color="white")
plt.tight_layout()
plt.savefig("_topdown_corrected.png", dpi=130)
print("saved _topdown_corrected.png")

def stats(name, xr, yr):
    m = (xyz[:,0]>=xr[0]) & (xyz[:,0]<=xr[1]) & (xyz[:,1]>=yr[0]) & (xyz[:,1]<=yr[1])
    z = xyz[m,2]
    if len(z) == 0:
        print(f"{name}: nessun punto"); return
    print(f"{name}: {m.sum()} punti, Z {z.min():.2f}..{z.max():.2f}  "
          f"(mediana {np.median(z):.2f}, std {z.std():.2f})")

lo, hi = xyz.min(0), xyz.max(0)
print(f"bounds: X {lo[0]:.1f}..{hi[0]:.1f}  Y {lo[1]:.1f}..{hi[1]:.1f}")
