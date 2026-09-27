from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer
import csv
import os

# آدرس فایلی که می‌خواهیم ذخیره شود
CSV_FILE_PATH = "C:/CameraDataset/motion_data.csv"

# اگر فایل وجود نداشت، خط اول (هدر ستون‌ها) را می‌نویسد
if not os.path.exists(CSV_FILE_PATH):
    os.makedirs(os.path.dirname(CSV_FILE_PATH), exist_ok=True)
    with open(CSV_FILE_PATH, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Frame_ID", "Delta_Yaw", "Delta_Pitch"])

# تابعی که به محض دریافت پیام از آنریل اجرا می‌شود
def save_to_csv(address, *args):
    frame_id = args[0]
    delta_yaw = args[1]
    delta_pitch = args[2]
    
    # ذخیره در فایل
    with open(CSV_FILE_PATH, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([frame_id, delta_yaw, delta_pitch])
        
    print(f"✅ Frame {frame_id} saved -> Yaw: {delta_yaw:.4f} | Pitch: {delta_pitch:.4f}")

# راه‌اندازی سرور گیرنده روی پورت 9000
dispatcher = Dispatcher()
dispatcher.map("/dataset", save_to_csv)

server = BlockingOSCUDPServer(("127.0.0.1", 9000), dispatcher)
print("🎧 Python is listening on port 9000 to record dataset... (Press Ctrl+C to stop)")
server.serve_forever()