"""Convert an OpenCamera-Sensors recording into training data.

Input  (dataset/phone_raw/<clip>/):  VID_x.mp4, VID_xgyro.csv, VID_xaccel.csv, VID_x_imu_timestamps.csv
Output (dataset/phone/<clip>/):      frames/000000.jpg ... (224x224), motion.csv, info.json

motion.csv, one row per video frame t (rotation between frame t and t+1, degrees):
    frame, yaw, pitch, intent_yaw, intent_pitch, res_yaw, res_pitch, speed

yaw   = rotation about the world vertical (gyro projected on gravity), positive = turn right
pitch = rotation about the camera's horizontal axis, positive = look up
intent   = zero-phase low-pass (default 1 Hz) of the rotation = where the operator meant to look
residual = rotation - intent = hand shake, step bob, settle/overshoot (the model's target)
speed    = locomotion speed proxy in m/s from accelerometer activity (0 idle, 1.4 walk, 3.0 run)

    python -m data_pipeline.phone_preprocess dataset/phone_raw/test01
"""
import argparse
import glob
import json
import os
import subprocess

import shutil

import cv2
import numpy as np
from scipy.signal import butter, filtfilt

FPS = 30.0


def ffmpeg_exe():
    """System ffmpeg if on PATH, otherwise the binary bundled with the imageio-ffmpeg package."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def count_frames(video):
    cap = cv2.VideoCapture(video)
    n = 0
    while cap.grab():
        n += 1
    cap.release()
    return n
FRAME_TIME_OFFSET_NS = 21_000_000  # measured on test01: frame timestamps lead the image content by ~21 ms


def load(prefix):
    ft = np.loadtxt(prefix + "_imu_timestamps.csv", dtype=np.int64)
    g = np.loadtxt(prefix + "gyro.csv", delimiter=",")
    a = np.loadtxt(prefix + "accel.csv", delimiter=",")
    return ft, g[:, :3], g[:, 3].astype(np.int64), a[:, :3], a[:, 3].astype(np.int64)


def per_frame_rotation(ft, gyro, gt, acc, at):
    """Integrate gyro between frame timestamps; returns yaw, pitch (deg/frame) and gravity-up per frame."""
    acc_g = np.stack([np.interp(gt, at, acc[:, k]) for k in range(3)], 1)
    b, a = butter(2, 0.5 / (0.5 * 1e9 / np.diff(gt).mean()))
    up = filtfilt(b, a, acc_g, axis=0)
    up /= np.linalg.norm(up, axis=1, keepdims=True)

    dt = np.diff(gt) / 1e9
    yaw_rate = -(gyro * up).sum(1)          # sign: positive = turn right
    pitch_rate = -gyro[:, 1]                # device y horizontal in landscape; positive = look up
    cum = lambda r: np.concatenate([[0.0], np.cumsum(r[1:] * dt)])
    t = ft + FRAME_TIME_OFFSET_NS
    yaw = np.degrees(np.diff(np.interp(t, gt, cum(yaw_rate))))
    pitch = np.degrees(np.diff(np.interp(t, gt, cum(pitch_rate))))
    return yaw, pitch


def speed_proxy(ft, acc, at):
    """Walking/running intensity from 1 s std of |acc|, mapped to the speeds Unreal will send."""
    mag = np.linalg.norm(acc, axis=1)
    fs = 1e9 / np.diff(at).mean()
    w = int(fs)
    std = np.sqrt(np.maximum(np.convolve(mag ** 2, np.ones(w) / w, "same") - np.convolve(mag, np.ones(w) / w, "same") ** 2, 0))
    s = np.interp(ft[:-1], at, std)
    return np.select([s < 0.6, s < 3.0], [0.0, 1.4], 3.0)


def extract_frames(video, out_dir, n):
    os.makedirs(out_dir, exist_ok=True)
    subprocess.run([ffmpeg_exe(), "-v", "error", "-y", "-i", video, "-frames:v", str(n),
                    "-vf", "scale=224:224", "-q:v", "3", "-start_number", "0",
                    os.path.join(out_dir, "%06d.jpg")], check=True)
    return len(glob.glob(os.path.join(out_dir, "*.jpg")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip_dir")
    ap.add_argument("--out_root", default="dataset/phone")
    ap.add_argument("--cutoff_hz", type=float, default=1.0)
    ap.add_argument("--no_frames", action="store_true")
    args = ap.parse_args()

    video = glob.glob(os.path.join(args.clip_dir, "*.mp4"))[0]
    prefix = video[:-4]
    ft, gyro, gt, acc, at = load(prefix)
    n_video = count_frames(video)
    n = min(n_video, len(ft))
    ft = ft[:n]
    ft = ft[(ft > gt[0]) & (ft < gt[-1])]
    n = len(ft)

    yaw, pitch = per_frame_rotation(ft, gyro, gt, acc, at)
    b, a = butter(2, args.cutoff_hz / (FPS / 2))
    i_yaw, i_pitch = filtfilt(b, a, yaw), filtfilt(b, a, pitch)
    speed = speed_proxy(ft, acc, at)

    clip = os.path.basename(os.path.normpath(args.clip_dir))
    out = os.path.join(args.out_root, clip)
    os.makedirs(out, exist_ok=True)
    rows = np.c_[np.arange(n - 1), yaw, pitch, i_yaw, i_pitch, yaw - i_yaw, pitch - i_pitch, speed]
    np.savetxt(os.path.join(out, "motion.csv"), rows, delimiter=",", fmt=["%d"] + ["%.6f"] * 7,
               header="frame,yaw,pitch,intent_yaw,intent_pitch,res_yaw,res_pitch,speed", comments="")

    n_frames = n if args.no_frames else extract_frames(video, os.path.join(out, "frames"), n)
    info = dict(clip=clip, frames=int(n_frames), motion_rows=int(n - 1), cutoff_hz=args.cutoff_hz,
                frame_time_offset_ms=FRAME_TIME_OFFSET_NS / 1e6,
                res_std_deg=[float(np.std(yaw - i_yaw)), float(np.std(pitch - i_pitch))],
                speed_share={str(v): float(np.mean(speed == v)) for v in (0.0, 1.4, 3.0)})
    json.dump(info, open(os.path.join(out, "info.json"), "w"), indent=2)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
