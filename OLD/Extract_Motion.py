import cv2
import numpy as np
import csv

def extract_motion_to_csv(video_path, output_csv_path):
    # ۱. باز کردن فایل ویدیویی
    cap = cv2.VideoCapture(video_path)
    
    # خواندن فریم اول
    ret, first_frame = cap.read()
    if not ret:
        print("خطا در بارگذاری ویدیو!")
        return

    # تبدیل فریم اول به حالت خاکستری
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    
    # ۲. آماده‌سازی فایل CSV برای نوشتن داده‌ها
    with open(output_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        # نوشتن سرستون‌های فایل CSV
        writer.writerow(["Frame_ID", "Delta_X", "Delta_Y"])
        
        frame_id = 1
        
        # ۳. حلقه پردازش فریم به فریم ویدیو
        while True:
            ret, frame = cap.read()
            if not ret:
                break # پایان ویدیو
            
            # تبدیل فریم فعلی به حالت خاکستری
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # ۴. محاسبه جریان نوری متراکم (Dense Optical Flow) بین فریم قبل و فعلی
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None, 
                pyr_scale=0.5, levels=3, winsize=15, 
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            
            # flow یک ماتریس است که برای هر پیکسل [dx, dy] را دارد.
            # میانگین حرکت کل پیکسل‌های این فریم را می‌گیریم تا حرکت کلی دوربین بدست آید.
            avg_dx = np.mean(flow[..., 0])
            avg_dy = np.mean(flow[..., 1])
            
            # ۵. ذخیره اطلاعات در فایل CSV
            writer.writerow([frame_id, avg_dx, avg_dy])
            
            # به‌روزرسانی فریم قبلی برای گام بعد
            prev_gray = gray
            frame_id += 1

    cap.release()
    print(f"🟢 استخراج با موفقیت پایان یافت! دیتای حرکتی در فایل ذخیره شد: {output_csv_path}")

# ====== اجرای آزمایشی ======
if __name__ == "__main__":
    # در پروژه واقعی مسیر ویدیو خودت را میدهی
    # extract_motion_to_csv("my_gameplay.mp4", "motion_data.csv")
    print("اسکریپت آماده است. با دادن مسیر ویدیو، فایل CSV تولید می‌شود.")