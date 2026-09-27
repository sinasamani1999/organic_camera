import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# وارد کردن ماژول‌هایی که خودمان در گام‌های قبلی ساختیم
from Real_Camera_Dataset import OrganicCameraDataset
from Organic_Camera_Model import OrganicCameraModel

def train_model():
    # ۱. تنظیم سخت‌افزار پردازشی
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 فرآیند آموزش روی این سخت‌افزار اجرا می‌شود: {device}")

    # ۲. مقداردهی اولیه به پایپ‌لاین داده‌های واقعی روی هارد
    # مسیر پوشه‌های ویدیو و تکست خودت را اینجا جایگزین می‌کنی
    dataset = OrganicCameraDataset(video_dir="dataset/train_videos", motion_dir="dataset/train_motions")
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True, drop_last=True)

    # ۳. فراخوانی مدل یکپارچه نهایی و انتقال آن به کارت گرافیک
    model = OrganicCameraModel().to(device)

    # ۴. تعریف تابع خطا (رگرسیون) و بهینه‌ساز شبکه
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # ۵. حلقه اصلی آموزش (شروع با ۵ دور آزمایشی)
    epochs = 5
    print("🎬 شروع فرآیند آموزش شبکه...")
    
    for epoch in range(epochs):
        model.train() # قرار دادن مدل در فاز تمرین
        running_loss = 0.0
        
        for batch_idx, (frames, motions, targets) in enumerate(train_loader):
            # انتقال بسته‌های داده به کارت گرافیک
            frames = frames.to(device)
            motions = motions.to(device)
            targets = targets.to(device)
            
            # صفر کردن گرادیان‌های دوره قبل
            optimizer.zero_grad()
            
            # الف) گام رو به جلو (Forward Pass) -> گرفتن پیش‌بینی مدل
            predictions = model(frames, motions)
            
            # ب) محاسبه میزان خطای این دوره
            loss = criterion(predictions, targets)
            
            # ج) گام به عقب (Backward Pass) -> محاسبه مشتقات خطا
            loss.backward()
            
            # د) به‌روزرسانی وزن‌های شبکه بر اساس محاسبات اتنشن و GRU
            optimizer.step()
            
            running_loss += loss.item()
            
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}] -------> میانگین خطای مدل (Loss): {epoch_loss:.4f}")

    # ۶. ذخیره کردن وزن‌های هوشمند مدل پس از پایان تمرین
    torch.save(model.state_dict(), "organic_camera_weights.pth")
    print("💾 تمرین تمام شد! فایل وزن‌های مدل با موفقیت ذخیره شد: organic_camera_weights.pth")

if __name__ == "__main__":
    # اجرای حلقه تمرین (برای اجرا باید پوشه دیتابیس حاوی حداقل یک ویدیو و csv باشد)
    # train_model()
    print("کد ساختار مربی (train.py) کامپایل شد و آماده اتصال به داده‌هاست.")