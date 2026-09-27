import socket
import time
import random

# تنظیمات سوکت برای اتصال به آنریل انجین
UDP_IP = "127.0.0.1" # سیستم محلی
UDP_PORT = 12345     # پورتی که در آنریل باز کردیم

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

print("📡 در حال ارسال دیتای فرضی دوربین به آنریل انجین... (برای توقف Ctrl+C بزنید)")

try:
    while True:
        # شبیه‌سازی خروجی مدل هوش مصنوعی (Yaw, Pitch)
        fake_yaw = round(random.uniform(-1.0, 1.0), 2)
        fake_pitch = round(random.uniform(-0.5, 0.5), 2)
        
        # تبدیل به فرمت متنی قابل فهم برای بلوپرینت آنریل
        message = f"{fake_yaw},{fake_pitch}"
        
        # ارسال دیتا از طریق پروتکل UDP
        sock.sendto(bytes(message, "utf-8"), (UDP_IP, UDP_PORT))
        
        print(f"ارسال شد: {message}")
        time.sleep(0.033) # معادل تقریبی ۳۰ فریم در ثانیه
        
except KeyboardInterrupt:
    print("\n🛑 ارسال متوقف شد.")