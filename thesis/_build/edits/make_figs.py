import sys, os, glob, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch
sys.path.insert(0, "/mnt/user-data/uploads/organic_camera")
import eval_metrics as em
OUT = "/home/claude/w/figs"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
C = {"real": "#52514e", "model": "#2a78d6", "onestep": "#1baf7a", "perlin": "#eb6834", "idle": "#1baf7a", "walk": "#2a78d6",
     "rollout_motion": "#1baf7a", "rollout_motion_spec": "#eb6834", "rollout_full_spec": "#2a78d6"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "lines.linewidth": 1.6})
R = "/mnt/user-data/uploads/organic_camera"

def psd(x):
    f, p = welch(x, fs=30, nperseg=256); return f, p

# F1: residual PSD by scenario
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7), dpi=200, sharey=True)
for scen in ["idle", "walk"]:
    ms = [pd.read_csv(f) for f in sorted(glob.glob(f"{R}/dataset/phone/*_{scen}_*/motion.csv"))]
    for k, col in enumerate(["res_yaw", "res_pitch"]):
        ps = [psd(m[col].values) for m in ms]
        f = ps[0][0]; p = np.mean([q for _, q in ps], 0)
        ax[k].semilogy(f, p, color=C[scen], label={"idle": "standing (idle)", "walk": "walking (walk)"}[scen])
for k, t in enumerate(["yaw", "pitch"]):
    ax[k].set_title(t, color=INK); ax[k].set_xlabel("Frequency (Hz)"); ax[k].axvspan(0, 1, color="#f1f0ec", lw=0)
ax[0].set_ylabel("PSD (deg$^2$/Hz)"); ax[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT}/fig4_1.png"); plt.close(fig)

# F2: rollout training curves
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7), dpi=200)
names = {"rollout_motion": "rollout, no spectral term", "rollout_motion_spec": "rollout + spectral, no image",
         "rollout_full_spec": "rollout + spectral, with image"}
for n, lab in names.items():
    h = pd.read_csv(f"{R}/runs/{n}/history.csv")
    ax[0].plot(h.epoch, h.val_mae_deg, color=C[n], label=lab)
    ax[1].plot(h.epoch, h.val_std_ratio, color=C[n], label=lab)
ax[0].axhline(h.val_zero_mae_deg.iloc[0], color=C["real"], ls="--", lw=1.2, label="zero prediction")
ax[0].set_ylabel("validation MAE (deg)"); ax[1].set_ylabel("validation std ratio")
ax[1].axhline(1.0, color=C["real"], ls="--", lw=1.2)
for a in ax: a.set_xlabel("epoch")
ax[0].legend(fontsize=7)
fig.tight_layout(); fig.savefig(f"{OUT}/fig4_2.png"); plt.close(fig)

# load predictions on test clip
one = pd.read_csv(f"{R}/runs/eval/phone/pred_p06_office_walk_01.csv")
rf = pd.read_csv(f"{R}/runs/eval_rollout/rollout_full_spec/pred_p06_office_walk_01.csv")
true = one[["true_yaw", "true_pitch"]].values

# F3: time series, a 4 s walking segment
mot = pd.read_csv(f"{R}/dataset/phone/p06_office_walk_01/motion.csv")
sp = mot.speed.values[one.frame.values]
# choose window with most walking
w = 120; best = max(range(0, len(sp) - w, 15), key=lambda i: sp[i:i + w].sum())
seg = slice(best, best + w); t = (np.arange(w)) / 30
fig, ax = plt.subplots(2, 1, figsize=(7.2, 4.2), dpi=200, sharex=True)
ax[0].plot(t, one.true_yaw.values[seg], color=C["real"], label="real")
ax[0].plot(t, one.teacher_yaw.values[seg], color=C["onestep"], label="one-step model, real history")
ax[0].set_title("(a) one-step prediction with real history (teacher mode)", fontsize=9, color=INK, loc="left")
ax[1].plot(t, one.true_yaw.values[seg], color=C["real"], label="real")
ax[1].plot(t, one.free_yaw.values[seg], color=C["onestep"], label="one-step training, free-running")
ax[1].plot(t, rf.free_yaw.values[seg], color=C["model"], label="rollout + spectral, free-running")
ax[1].set_title("(b) free-running generation (as in the live loop)", fontsize=9, color=INK, loc="left")
for a in ax: a.set_ylabel("yaw residual (deg)"); a.legend(fontsize=7, ncol=3, loc="upper right")
ax[1].set_xlabel("time (s)")
fig.tight_layout(); fig.savefig(f"{OUT}/fig4_3.png"); plt.close(fig)

# F4: PSD comparison on test clip (free-running)
per = em.perlin_matched(true)
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8), dpi=200, sharey=True)
for k, a in enumerate(["yaw", "pitch"]):
    for lab, x, c in [("real", true[:, k], C["real"]), ("one-step training (free)", one[f"free_{a}"].values, C["onestep"]),
                      ("rollout + spectral (free)", rf[f"free_{a}"].values, C["model"]), ("Perlin, equal RMS", per[:, k], C["perlin"])]:
        f, p = psd(x); ax[k].semilogy(f, p, color=c, label=lab)
    ax[k].set_title(a, color=INK); ax[k].set_xlabel("Frequency (Hz)"); ax[k].axvspan(0, 1, color="#f1f0ec", lw=0)
ax[0].set_ylabel("PSD (deg$^2$/Hz)"); ax[1].legend(fontsize=7, loc="lower left")
fig.tight_layout(); fig.savefig(f"{OUT}/fig4_4.png"); plt.close(fig)

# F5: attention maps
from PIL import Image
ims = [Image.open(f) for f in sorted(glob.glob(f"{R}/runs/eval/phone/attention_p06_office_walk_01_teacher_*.png"))]
h = min(i.height for i in ims); ims = [i.resize((int(i.width * h / i.height), h)) for i in ims]
gap = 12; W = sum(i.width for i in ims) + gap * (len(ims) - 1)
c = Image.new("RGB", (W, h), "white"); x = 0
for i in ims: c.paste(i.convert("RGB"), (x, 0)); x += i.width + gap
c.save(f"{OUT}/fig4_5.png")

# F6: latency histogram
d = pd.concat([pd.read_csv(f"{R}/runs/latency/{f}").iloc[30:] for f in ["20261001_030741.csv", "20261001_023844.csv"]])
fig, ax = plt.subplots(figsize=(7.2, 2.6), dpi=200)
ax.hist(d.total_ms, bins=np.arange(0, 36, 0.5), color=C["model"], edgecolor="white", linewidth=0.5)
ax.axvline(33.3, color=C["perlin"], ls="--", lw=1.4); ax.text(32.8, ax.get_ylim()[1] * 0.9, "one 30 Hz step (33.3 ms)", ha="right", fontsize=8, color=INK2)
for q, lab in [(50, "median"), (95, "p95")]:
    v = np.percentile(d.total_ms, q); ax.axvline(v, color=INK2, lw=1); ax.text(v + 0.3, ax.get_ylim()[1] * 0.75, f"{lab} {v:.1f}", fontsize=8, color=INK2)
ax.set_xlabel("total latency (ms)"); ax.set_ylabel("steps"); ax.grid(axis="x", visible=False)
fig.tight_layout(); fig.savefig(f"{OUT}/fig4_6.png"); plt.close(fig)
print("segment start frame", one.frame.values[best])
