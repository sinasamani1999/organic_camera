"""Match the amplitude (RMS) of the camera layers for the user study.

Play the same take once per mode (keys 1-4, then P). Each playback writes
<Project>/Saved/OrganicCamera/Logs/<take>_<Mode>_<time>.csv. Then run:

    python tools/rms_match.py "C:/Users/.../unreal/OrganicCamera/Saved/OrganicCamera/Logs" --take take_20261001_101500
    python tools/rms_match.py <logs_dir> --take <take> --perlin-amp 1.5 --shake-scale 1.0

It prints the RMS of the camera offset (view rotation minus control rotation, the same quantity for every
mode, Camera Shake included) and the Perlin amplitude / shake scale that give the same RMS as the AI mode.
Repeat until the ratios are close to 1 (Camera Shake is not exactly linear in its scale, so one more
iteration may be needed).
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

MODES = ["Off", "Perlin", "CameraShake", "AI"]


def rms(df):
    """RMS of the 2-D offset in degrees, after removing the mean (a constant offset is not shake)."""
    y = df["view_yaw"].to_numpy() - df["view_yaw"].mean()
    p = df["view_pitch"].to_numpy() - df["view_pitch"].mean()
    return float(np.sqrt(np.mean(y ** 2 + p ** 2))), float(np.std(y)), float(np.std(p))


def newest(logs, take, mode):
    files = sorted(glob.glob(os.path.join(logs, f"{take}_{mode}_*.csv")), key=os.path.getmtime)
    return files[-1] if files else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs")
    ap.add_argument("--take", required=True, help="take name without .csv")
    ap.add_argument("--perlin-amp", type=float, default=1.5, help="PerlinAmplitudeDeg used for the Perlin log")
    ap.add_argument("--shake-scale", type=float, default=1.0, help="ShakeScale used for the CameraShake log")
    ap.add_argument("--skip", type=float, default=0.5, help="seconds ignored at the start (settling)")
    a = ap.parse_args()

    res = {}
    for m in MODES:
        f = newest(a.logs, a.take, m)
        if not f:
            print(f"{m:12s} no log")
            continue
        df = pd.read_csv(f)
        df = df[df["time"] >= a.skip]
        res[m] = rms(df)
        off = float(np.sqrt(np.mean((df["offset_yaw"] - df["offset_yaw"].mean()) ** 2 + (df["offset_pitch"] - df["offset_pitch"].mean()) ** 2)))
        print(f"{m:12s} RMS {res[m][0]:.3f} deg  (yaw std {res[m][1]:.3f}, pitch std {res[m][2]:.3f})  "
              f"component offset RMS {off:.3f}  n={len(df)}  {os.path.basename(f)}")

    if "AI" not in res:
        print("\nno AI log: the AI mode is the reference amplitude")
        return
    target = res["AI"][0]
    print(f"\ntarget RMS (AI) = {target:.3f} deg")
    if "Perlin" in res and res["Perlin"][0] > 0:
        r = target / res["Perlin"][0]
        print(f"Perlin:      ratio {r:.3f} -> set PerlinAmplitudeDeg = {a.perlin_amp * r:.3f}")
    if "CameraShake" in res and res["CameraShake"][0] > 0:
        r = target / res["CameraShake"][0]
        print(f"CameraShake: ratio {r:.3f} -> set ShakeScale = {a.shake_scale * r:.3f}")


if __name__ == "__main__":
    main()
