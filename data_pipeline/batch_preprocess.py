"""Preprocess every recording in dataset/phone_raw in one go and write a summary table.

    python -m data_pipeline.batch_preprocess                 # new clips only
    python -m data_pipeline.batch_preprocess --force         # redo everything

Each folder dataset/phone_raw/<clip>/ must hold one OpenCamera Sensors recording (mp4 + gyro/accel/timestamps).
Clip names must start with the person id (p01_office_walk_01) because the train/val/test split is by person.
Writes dataset/phone/summary.csv (frames, minutes, residual std, speed shares per clip) and prints totals per person.
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys

import pandas as pd

NAME = re.compile(r"^p\d{2}_[a-z0-9]+_[a-z0-9]+_\d{2}$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="dataset/phone_raw")
    ap.add_argument("--out_root", default="dataset/phone")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--cutoff_hz", type=float, default=1.0)
    a = ap.parse_args()

    clips = sorted(d for d in glob.glob(os.path.join(a.raw, "*")) if os.path.isdir(d))
    failed = []
    for d in clips:
        clip = os.path.basename(d)
        if not NAME.match(clip):
            print(f"  ! {clip}: name does not follow pNN_place_scenario_NN (it will still be processed)")
        if not glob.glob(os.path.join(d, "*.mp4")):
            print(f"  - {clip}: no mp4, skipped")
            continue
        if not a.force and os.path.exists(os.path.join(a.out_root, clip, "info.json")):
            print(f"  = {clip}: already done")
            continue
        print(f"  > {clip}")
        r = subprocess.run([sys.executable, "-m", "data_pipeline.phone_preprocess", d,
                            "--out_root", a.out_root, "--cutoff_hz", str(a.cutoff_hz)], capture_output=True, text=True)
        if r.returncode != 0:
            failed.append(clip)
            print("    FAILED:\n" + "\n".join("    " + l for l in r.stderr.strip().splitlines()[-5:]))

    rows = []
    for info in sorted(glob.glob(os.path.join(a.out_root, "*", "info.json"))):
        i = json.load(open(info))
        parts = i["clip"].split("_")
        rows.append(dict(clip=i["clip"], person=parts[0], scenario=parts[2] if len(parts) > 2 else "",
                         frames=i["frames"], minutes=round(i["motion_rows"] / 30 / 60, 2),
                         res_std_yaw=round(i["res_std_deg"][0], 3), res_std_pitch=round(i["res_std_deg"][1], 3),
                         idle=round(i["speed_share"]["0.0"], 2), walk=round(i["speed_share"]["1.4"], 2),
                         fast=round(i["speed_share"]["3.0"], 2)))
    df = pd.DataFrame(rows)
    if df.empty:
        print("no processed clips")
        return
    df.to_csv(os.path.join(a.out_root, "summary.csv"), index=False)
    pd.set_option("display.width", 160)
    print("\n" + df.to_string(index=False))
    print("\nper person (minutes):\n" + df.groupby("person")["minutes"].sum().round(1).to_string())
    print("\nper scenario (minutes):\n" + df.groupby("scenario")["minutes"].sum().round(1).to_string())
    print(f"\ntotal {df.minutes.sum():.1f} min in {len(df)} clips; summary in {a.out_root}/summary.csv")
    if failed:
        print(f"failed: {failed}")


if __name__ == "__main__":
    main()
