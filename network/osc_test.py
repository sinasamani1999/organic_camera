"""Week-1 link test between Unreal and Python.

Receives /oc/state from Unreal (port 7001), replies with a synthetic /oc/residual (port 7000).
In the game press 4 (AI mode): the camera should sway gently. Ctrl+C prints the stats.

    python network/osc_test.py
"""
import math
import threading
import time

from pythonosc import dispatcher, osc_server, udp_client

UE_IP, SEND_PORT, RECV_PORT = "127.0.0.1", 7000, 7001
client = udp_client.SimpleUDPClient(UE_IP, SEND_PORT)
stats = {"n": 0, "first": None, "last_frame": None, "speed": 0.0, "intent": (0.0, 0.0)}


def on_state(address, frame, dt, intent_yaw, intent_pitch, speed):
    now = time.perf_counter()
    stats["first"] = stats["first"] or now
    stats["n"] += 1
    stats["last_frame"], stats["speed"], stats["intent"] = frame, speed, (intent_yaw, intent_pitch)
    t = frame / 60.0
    # 1.5 Hz sway, amplitude grows with walking speed
    amp = 0.03 + 0.05 * min(speed, 5.0)
    client.send_message("/oc/residual", [amp * math.sin(2 * math.pi * 1.5 * t),
                                         0.5 * amp * math.sin(2 * math.pi * 3.0 * t)])


d = dispatcher.Dispatcher()
d.map("/oc/state", on_state)
server = osc_server.ThreadingOSCUDPServer(("127.0.0.1", RECV_PORT), d)
threading.Thread(target=server.serve_forever, daemon=True).start()
print(f"listening /oc/state on {RECV_PORT}, sending /oc/residual to {SEND_PORT}. Press Play in Unreal, then 4.")

try:
    while True:
        time.sleep(1.0)
        if stats["first"]:
            rate = stats["n"] / (time.perf_counter() - stats["first"])
            print(f"frame {stats['last_frame']}  {rate:5.1f} msg/s  speed {stats['speed']:.2f} m/s  "
                  f"intent yaw {stats['intent'][0]:+.3f} pitch {stats['intent'][1]:+.3f}", end="\r")
except KeyboardInterrupt:
    server.shutdown()
    print(f"\nreceived {stats['n']} messages")
