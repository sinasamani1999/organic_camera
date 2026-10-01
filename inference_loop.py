"""Live inference: Unreal -> Python -> Unreal.

Unreal sends /oc/state [frame, dt, intent_yaw, intent_pitch, speed] every game frame (port 7001).
Python accumulates the intent to 30 Hz (the rate the model was trained at), grabs the game
window, predicts the next organic residual and sends /oc/residual [yaw, pitch] (port 7000).

    python inference_loop.py --ckpt runs/phone/best.pth               # with Unreal (Play, then key 4)
    python inference_loop.py --ckpt runs/phone/best.pth --fake-state  # no Unreal: latency test only

Latency of every step is written to runs/latency/<time>.csv and summarised on Ctrl+C.
"""
import argparse
import csv
import ctypes
import ctypes.wintypes as wt
import os
import queue
import threading
import time
from collections import deque

import numpy as np
import torch
from pythonosc import dispatcher, osc_server, udp_client

from models.Organic_Camera_Model import OrganicCameraModel

MODEL_HZ = 30.0
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


# ---------------------------------------------------------------- screen capture
def find_window_rect(title_part):
    """Client rect (left, top, right, bottom) of the game window.

    Prefers the Play-In-Editor window ("... Preview [NetMode ...]") and never picks the editor itself.
    """
    user32 = ctypes.windll.user32
    found = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
    def enum(hwnd, _):
        if user32.IsWindowVisible(hwnd):
            n = user32.GetWindowTextLengthW(hwnd)
            buf = ctypes.create_unicode_buffer(n + 1)
            user32.GetWindowTextW(hwnd, buf, n + 1)
            if title_part.lower() in buf.value.lower():
                found.append((hwnd, buf.value))
        return True

    user32.EnumWindows(enum, 0)
    found = [f for f in found if "unreal editor" not in f[1].lower()]
    found.sort(key=lambda f: "preview" not in f[1].lower())  # PIE window first
    if not found:
        return None, None
    hwnd, title = found[0]
    rect, pt = wt.RECT(), wt.POINT(0, 0)
    user32.GetClientRect(hwnd, ctypes.byref(rect))
    user32.ClientToScreen(hwnd, ctypes.byref(pt))
    return (pt.x, pt.y, pt.x + rect.right, pt.y + rect.bottom), title


class Grabber:
    """dxcam (fast, DXGI) if available, otherwise mss."""

    def __init__(self, region):
        self.region = region
        try:
            import dxcam
            self.cam = dxcam.create(output_color="RGB")
            self.cam.start(region=region, target_fps=120, video_mode=True)
            self.kind = "dxcam"
        except Exception as e:  # noqa: BLE001
            l, t, r, b = region
            self.mon = {"left": l, "top": t, "width": r - l, "height": b - t}
            self.kind = f"mss ({e.__class__.__name__} from dxcam)"

        # capture runs in its own thread; the model step just takes the newest frame (no waiting)
        self.latest, self.latest_t, self.running = None, 0.0, True
        self.lock = threading.Lock()
        threading.Thread(target=self._loop, daemon=True).start()

    def _grab_blocking(self):
        if self.kind == "dxcam":
            return self.cam.get_latest_frame()
        import mss
        if not hasattr(self._tls, "sct"):
            self._tls.sct = getattr(mss, "MSS", mss.mss)()   # mss objects are per-thread
        img = np.asarray(self._tls.sct.grab(self.mon))[:, :, :3]
        return img[:, :, ::-1]

    _tls = threading.local()

    def _loop(self):
        while self.running:
            img = self._grab_blocking()
            if img is not None:
                with self.lock:
                    self.latest, self.latest_t = img, time.perf_counter()
            if self.kind != "dxcam":
                time.sleep(0.005)  # mss does not wait for new frames; cap at ~120 grabs/s to limit memory churn

    def grab(self):
        """Newest frame and its age in ms (never blocks)."""
        with self.lock:
            return self.latest, (time.perf_counter() - self.latest_t) * 1e3

    def stop(self):
        self.running = False
        if self.kind == "dxcam":
            self.cam.stop()


def preprocess(rgb, device, mean, std):
    """uint8 HxWx3 -> normalised [1,3,224,224] on the GPU (resize on GPU, ~1 ms)."""
    x = torch.from_numpy(np.ascontiguousarray(rgb)).to(device, non_blocking=True)
    x = x.permute(2, 0, 1).unsqueeze(0).float().div_(255.0)
    x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bilinear", antialias=True, align_corners=False)
    return (x - mean) / std


# ---------------------------------------------------------------- main loop
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/phone/best.pth")
    ap.add_argument("--window", default="OrganicCamera", help="part of the game window title")
    ap.add_argument("--fake-state", action="store_true", help="generate states internally (no Unreal)")
    ap.add_argument("--gain", type=float, default=1.0, help="scale of the residual sent to Unreal")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    m = ck["config"]["model"]
    model = OrganicCameraModel(m["hidden_dim"], m["num_heads"], False, True, m["use_visual"], m["use_attention"])
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    norm = {k: v.to(device) for k, v in ck["norm"].items()}
    T = ck["config"]["data"]["seq_len"]
    print(f"model loaded from {args.ckpt} on {device}")

    region, title = find_window_rect(args.window)
    if region is None:
        print(f"no game window with '{args.window}' in its title. In Unreal use Play -> 'New Editor Window (PIE)',"
              " start this script after the game window is open, or pass --window <part of its title>.")
        return
    grabber = Grabber(region)
    print(f"capturing '{title}' {region} with {grabber.kind}")
    time.sleep(0.5)  # let the capture thread deliver a first frame

    # warm-up so the first real step does not include CUDA initialisation
    with torch.no_grad():
        for _ in range(5):
            model(torch.zeros(1, 3, 224, 224, device=device), torch.zeros(1, T, 5, device=device))
    if device.type == "cuda":
        torch.cuda.synchronize()

    states = queue.Queue()
    client = udp_client.SimpleUDPClient("127.0.0.1", 7000)
    server = None
    if args.fake_state:
        def fake():
            f = 0
            while True:
                states.put((time.perf_counter(), f, 1 / 60, 0.2 * np.sin(f / 60), 0.0, 1.4))
                f += 1
                time.sleep(1 / 60)
        threading.Thread(target=fake, daemon=True).start()
    else:
        d = dispatcher.Dispatcher()
        d.map("/oc/state", lambda _a, fr, dt, iy, ip, sp: states.put((time.perf_counter(), fr, dt, iy, ip, sp)))
        server = osc_server.ThreadingOSCUDPServer(("127.0.0.1", 7001), d)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        print("waiting for /oc/state on 7001 (press Play in Unreal, then 4 for AI mode)")

    mean_t = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(STD, device=device).view(1, 3, 1, 1)
    history = deque([np.zeros(5, np.float32)] * T, maxlen=T)
    acc_yaw = acc_pitch = acc_dt = 0.0
    speed = 0.0
    last_res = np.zeros(2, np.float32)

    os.makedirs("runs/latency", exist_ok=True)
    log_path = time.strftime("runs/latency/%Y%m%d_%H%M%S.csv")
    log = open(log_path, "w", newline="")
    w = csv.writer(log)
    w.writerow(["step", "frame", "capture_ms", "frame_age_ms", "preprocess_ms", "model_ms", "total_ms",
                "skipped_rows", "res_yaw", "res_pitch"])
    step = 0
    try:
        while True:
            # take everything that arrived; never fall behind Unreal
            try:
                item = states.get(timeout=0.25)  # a timeout keeps Ctrl+C working on Windows
            except queue.Empty:
                continue
            new_rows = 0
            while True:
                t_recv, frame, dt, iy, ip, sp = item
                acc_yaw, acc_pitch, acc_dt, speed = acc_yaw + iy, acc_pitch + ip, acc_dt + dt, sp
                if acc_dt >= 1.0 / MODEL_HZ:  # one 30 Hz row = [intent_yaw, intent_pitch, res_yaw, res_pitch, speed]
                    history.append(np.array([acc_yaw, acc_pitch, last_res[0], last_res[1], speed], np.float32))
                    acc_yaw = acc_pitch = 0.0
                    acc_dt -= 1.0 / MODEL_HZ
                    new_rows += 1
                try:
                    item = states.get_nowait()
                except queue.Empty:
                    break
            if new_rows == 0:
                continue

            t0 = time.perf_counter()
            img, frame_age = grabber.grab()
            t1 = time.perf_counter()
            if img is None:
                continue
            x = preprocess(img, device, mean_t, std_t)
            motion = torch.from_numpy(np.stack(history)).unsqueeze(0).to(device)
            motion = (motion - norm["m_mean"]) / norm["m_std"]
            if device.type == "cuda":
                torch.cuda.synchronize()
            t2 = time.perf_counter()
            with torch.no_grad():
                out = model(x, motion)[0] * norm["t_std"] + norm["t_mean"]
            if device.type == "cuda":
                torch.cuda.synchronize()
            res = out.float().cpu().numpy()
            t3 = time.perf_counter()
            client.send_message("/oc/residual", [float(res[0] * args.gain), float(res[1] * args.gain)])
            t4 = time.perf_counter()
            last_res = res.astype(np.float32)

            w.writerow([step, frame, f"{(t1 - t0) * 1e3:.2f}", f"{frame_age:.2f}", f"{(t2 - t1) * 1e3:.2f}",
                        f"{(t3 - t2) * 1e3:.2f}", f"{(t4 - t_recv) * 1e3:.2f}", new_rows - 1,
                        f"{res[0]:.4f}", f"{res[1]:.4f}"])
            step += 1
            if step % 30 == 0:
                log.flush()
                print(f"step {step}  model {(t3 - t2) * 1e3:5.1f} ms  total {(t4 - t_recv) * 1e3:5.1f} ms  "
                      f"res yaw {res[0]:+.3f} pitch {res[1]:+.3f}  speed {speed:.1f}", end="\r")
    except KeyboardInterrupt:
        pass
    finally:
        log.close()
        grabber.stop()
        if server:
            server.shutdown()
        summarise(log_path)


def summarise(path):
    import pandas as pd
    df = pd.read_csv(path)
    if df.empty:
        print("\nno steps recorded")
        return
    print(f"\n\n{len(df)} steps logged to {path}")
    print(f"  capture method: see start of run; steps that had to skip rows: {(df.skipped_rows > 0).mean() * 100:.1f}%")
    for col in ["capture_ms", "frame_age_ms", "preprocess_ms", "model_ms", "total_ms"]:
        q = df[col].quantile([0.5, 0.95, 0.99])
        print(f"  {col:14s} p50 {q[0.5]:6.2f}  p95 {q[0.95]:6.2f}  p99 {q[0.99]:6.2f}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 3 and sys.argv[1] == "--summary":  # python inference_loop.py --summary <csv>
        summarise(sys.argv[2])
    else:
        main()
