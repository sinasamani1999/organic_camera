"""Evaluate trained models on the held-out person (chapter 4 numbers).

    python evaluate.py --ckpt runs/phone/best.pth runs/phone_motion_only/best.pth runs/phone_no_attn/best.pth
    python evaluate.py --ckpt runs/phone/best.pth --split val --attention 8

For every checkpoint: predicts the residual of each frame of each test clip (one step, the same way
the live loop does), compares it with the real residual and with two baselines (zero = no organic
layer, Perlin with the same RMS), and writes
    runs/eval/<name>/metrics.csv          one row per method, metrics of table 3-6
    runs/eval/<name>/per_clip.csv         the same per clip
    runs/eval/<name>/pred_<clip>.csv      frame, true and predicted yaw/pitch (for plots)
    runs/eval/<name>/psd.png              power spectrum: real vs model vs Perlin
    runs/eval/<name>/attention_*.png      attention maps on sample frames (--attention N)
Two evaluation modes for the residual history (section 3-4-5):
    teacher  : the history holds the real past residuals (as in training)
    free     : the history holds the model's own past predictions (as in the live loop)
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
import yaml

import eval_metrics as em
from data_pipeline.phone_dataset import PhoneOrganicDataset, build_phone_splits, load_motion
from models.Organic_Camera_Model import OrganicCameraModel


def load_model(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    m = ck["config"]["model"]
    model = OrganicCameraModel(m["hidden_dim"], m["num_heads"], False, True, m["use_visual"], m["use_attention"])
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    norm = {k: v.to(device) for k, v in ck["norm"].items()}
    return model, norm, ck["config"]


@torch.no_grad()
def predict_clip(model, norm, ds, clip_idx, mode, device, batch=64, attn_frames=()):
    """Residual predictions (deg) for every valid frame of one clip. Returns t indices, pred [N,2], attention dict."""
    frames_dir, feats = ds.clips[clip_idx]
    T = ds.T
    ts = [t for (c, t) in ds.items if c == clip_idx]
    feats = feats.copy()
    preds, attn = [], {}
    if mode == "free":
        # sequential: the residual columns of the history are replaced by the model's own outputs
        feats[:, 2:4] = 0.0
        for t in ts:
            x = ds._frame(frames_dir, t).unsqueeze(0).to(device)
            m = torch.from_numpy(feats[t - T:t]).unsqueeze(0).to(device)
            out, w = model(x, (m - norm["m_mean"]) / norm["m_std"], return_attention=True)
            r = (out[0] * norm["t_std"] + norm["t_mean"]).cpu().numpy()
            feats[t, 2:4] = r
            preds.append(r)
            if t in attn_frames and w is not None:
                attn[t] = w[0].cpu().numpy()
        return np.array(ts), np.array(preds), attn
    for i in range(0, len(ts), batch):
        tb = ts[i:i + batch]
        x = torch.stack([ds._frame(frames_dir, t) for t in tb]).to(device)
        m = torch.from_numpy(np.stack([feats[t - T:t] for t in tb])).to(device)
        out, w = model(x, (m - norm["m_mean"]) / norm["m_std"], return_attention=True)
        preds.append((out * norm["t_std"] + norm["t_mean"]).cpu().numpy())
        if w is not None:
            for j, t in enumerate(tb):
                if t in attn_frames:
                    attn[t] = w[j].cpu().numpy()
    return np.array(ts), np.concatenate(preds), attn


def save_attention(frames_dir, t, w, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    img = np.asarray(Image.open(os.path.join(frames_dir, f"{t:06d}.jpg")).convert("RGB"))
    heat = np.kron(w.reshape(7, 7), np.ones((32, 32)))
    fig, ax = plt.subplots(figsize=(3, 3), dpi=150)
    ax.imshow(img)
    ax.imshow(heat, cmap="magma", alpha=0.45, extent=(0, img.shape[1], img.shape[0], 0))
    ax.set_axis_off()
    fig.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def save_psd(true, methods, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(8, 3), dpi=150)
    for i, a in enumerate(em.AXES):
        for name, x in [("real", true)] + list(methods.items()):
            _, f, p = em.band_powers(x[:, i])
            axes[i].semilogy(f, p, label=name, lw=1.2)
        axes[i].set_title(a)
        axes[i].set_xlabel("Hz")
        axes[i].axvspan(0, 1, color="0.9")
    axes[0].set_ylabel("PSD (deg²/Hz)")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--split", choices=["val", "test"], default="test")
    ap.add_argument("--mode", choices=["teacher", "free", "both"], default="both")
    ap.add_argument("--attention", type=int, default=0, help="number of sample frames per clip to plot")
    ap.add_argument("--out", default="runs/eval")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    summary = []
    for path in args.ckpt:
        model, norm, cfg = load_model(path, device)
        name = os.path.basename(os.path.dirname(path)) or "model"
        out_dir = os.path.join(args.out, name)
        os.makedirs(out_dir, exist_ok=True)
        _, val_ds, test_ds = build_phone_splits(cfg)
        ds = test_ds if args.split == "test" else val_ds
        if len(ds) == 0:
            print(f"{path}: empty {args.split} split")
            continue
        print(f"{name}: {len(ds.clips)} clips, {len(ds)} frames on {device}")

        modes = ["teacher", "free"] if args.mode == "both" else [args.mode]
        all_true, all_pred = [], {m: [] for m in modes}
        per_clip = []
        for ci, (frames_dir, feats) in enumerate(ds.clips):
            clip = os.path.basename(os.path.dirname(frames_dir))
            ts_all = [t for (c, t) in ds.items if c == ci]
            pick = set(np.linspace(ts_all[0], ts_all[-1], args.attention + 2, dtype=int)[1:-1]) if args.attention else set()
            rows = None
            for mode in modes:
                ts, pred, attn = predict_clip(model, norm, ds, ci, mode, device, attn_frames=pick)
                true = feats[ts, 2:4]
                if rows is None:
                    rows = pd.DataFrame({"frame": ts, "true_yaw": true[:, 0], "true_pitch": true[:, 1]})
                    all_true.append(true)
                rows[f"{mode}_yaw"], rows[f"{mode}_pitch"] = pred[:, 0], pred[:, 1]
                all_pred[mode].append(pred)
                per_clip.append({"clip": clip, "method": f"model_{mode}", **em.all_metrics(pred, true)})
                for t, w in attn.items():
                    save_attention(frames_dir, t, w, os.path.join(out_dir, f"attention_{clip}_{mode}_{t:06d}.png"))
            rows.to_csv(os.path.join(out_dir, f"pred_{clip}.csv"), index=False)

        true = np.concatenate(all_true)
        methods = {f"model_{m}": np.concatenate(all_pred[m]) for m in modes}
        methods["zero"] = np.zeros_like(true)
        methods["perlin_matched"] = em.perlin_matched(true)
        res = pd.DataFrame([{"method": k, **em.all_metrics(v, true)} for k, v in methods.items()])
        res.insert(0, "model", name)
        res.to_csv(os.path.join(out_dir, "metrics.csv"), index=False)
        pd.DataFrame(per_clip).to_csv(os.path.join(out_dir, "per_clip.csv"), index=False)
        save_psd(true, {k: v for k, v in methods.items() if k != "zero"}, os.path.join(out_dir, "psd.png"))
        summary.append(res)
        cols = ["method", "mae_deg", "mae_ratio_to_zero", "std_ratio_yaw", "corr_yaw", "lsd_db_yaw"]
        print(res[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    if summary:
        pd.concat(summary).to_csv(os.path.join(args.out, "summary.csv"), index=False)
        print(f"\nwritten to {args.out}/")


if __name__ == "__main__":
    main()
