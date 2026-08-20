import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

xyz = np.load("_topdown_cache.npy")

# binned mean-Z top-down: same XY grid as before, color = mean height per cell
xbins = np.linspace(xyz[:,0].min(), xyz[:,0].max(), 400)
ybins = np.linspace(xyz[:,1].min(), xyz[:,1].max(), 300)

sumZ, _, _ = np.histogram2d(xyz[:,0], xyz[:,1], bins=[xbins, ybins], weights=xyz[:,2])
cnt, _, _ = np.histogram2d(xyz[:,0], xyz[:,1], bins=[xbins, ybins])
with np.errstate(invalid="ignore"):
    meanZ = sumZ / cnt
meanZ = np.ma.masked_invalid(meanZ)

fig, ax = plt.subplots(figsize=(14, 10))
pcm = ax.pcolormesh(xbins, ybins, meanZ.T, cmap="turbo", shading="auto")
fig.colorbar(pcm, ax=ax, label="Z medio (m)")
ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
ax.set_title("Vista dall'alto colorata per quota Z media")
ax.set_aspect("equal")
ax.set_xticks(np.arange(0, 56, 5)); ax.set_yticks(np.arange(-30, 41, 5))
ax.grid(True, alpha=0.3, color="white")
plt.tight_layout()
plt.savefig("_topdown_z.png", dpi=130)
print("saved _topdown_z.png")

# stats per candidate region
def stats(name, xr, yr):
    m = (xyz[:,0]>=xr[0]) & (xyz[:,0]<=xr[1]) & (xyz[:,1]>=yr[0]) & (xyz[:,1]<=yr[1])
    z = xyz[m,2]
    if len(z) == 0:
        print(f"{name}: nessun punto"); return
    print(f"{name}: {m.sum()} punti, Z {z.min():.2f}..{z.max():.2f}  "
          f"(mediana {np.median(z):.2f}, std {z.std():.2f})")

stats("A (main, Y -3..3, X 4..40)", (4,40), (-3,3))
stats("D (X 39..48, Y -5..36)", (39,48), (-5,36))
stats("E (X 39..53, Y -30..2)", (39,53), (-30,2))
