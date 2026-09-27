from pythonosc.udp_client import SimpleUDPClient
import time
import random


# تنظیمات اتصال به سرور داخلی آنریل انجین
# IP محلی سیستم و پورتی که در بلوپرینت آنریل باز کردی را وارد کن
UDP_IP = "127.0.0.1"
UDP_PORT = 8000

# راه‌اندازی کلاینت OSC
client = SimpleUDPClient(UDP_IP, UDP_PORT)

print(f"📡 فرستنده OSC فعال شد. ارسال دیتا به پورت {UDP_PORT}...")
print("برای متوقف کردن خروجی، کلیدهای Ctrl+C را فشار دهید.\n")

try:
    while True:
        # در پروژه واقعی، این دو عدد خروجی مدل هوش مصنوعی تو (Organic_Camera_Model) خواهند بود
        # شبیه‌سازی حرکت ارگانیک یا تکان‌های نرم دوربین:
        fake_yaw = random.uniform(-1.0, 1.0)
        fake_pitch = random.uniform(-0.5, 0.5)
        
        # فرستادن پکت داده به آدرس مشخصی که در بلوپرینت آنریل فیلتر می‌کنی
        # ما هر دو عدد Yaw و Pitch را در قالب یک پکت و یک‌جا ارسال می‌کنیم
        client.send_message("/camera", [fake_yaw, fake_pitch])
        
        print(f"[OSC Sent] Address: /camera | Yaw: {fake_yaw:.2f}, Pitch: {fake_pitch:.2f}")
        
        # هماهنگ با نرخ فریم بازی (تقریباً ۳۰ فریم در ثانیه)
        time.sleep(0.033)

except KeyboardInterrupt:
    print("\n ارسال داده‌ها توسط کاربر متوقف شد.")