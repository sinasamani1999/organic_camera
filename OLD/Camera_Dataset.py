import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class CameraOrganicDataset(Dataset):
    def __init__(self, num_samples=100, seq_length=10):
        """
        در این فاز برای تست پایپ‌لاین، دیتای شبیه‌سازی شده اما با ساختار کاملاً واقعی می‌سازیم.
        در آینده این بخش فایل‌های روی هارد تو را اسکن خواهد کرد.
        """
        self.num_samples = num_samples
        self.seq_length = seq_length
        
        # شبیه‌سازی دیتای فریم‌های بازی: num_samples فریم با کیفیت 224x224
        self.visual_data = torch.randn(num_samples, 3, 224, 224)
        
        # شبیه‌سازی دیتای حرکتی ماوس: برای هر فریم، یک توالی ۱۰ فریمی از [Delta_X, Delta_Y] داریم
        self.motion_data = torch.randn(num_samples, seq_length, 2)
        
        # شبیه‌سازی برچسب نهایی (مقداری که مدل باید پیش‌بینی کند - جابجایی ارگانیک فریم بعدی)
        self.target_data = torch.randn(num_samples, 2)

    def __len__(self):
        # تعداد کل فریم‌ها/نمونه‌های موجود برای آموزش
        return self.num_samples

    def __getitem__(self, idx):
        """
        وقتی لودر دیتابیس صدا زده می‌شود، این تابع فریم مربوطه، تاریخچه حرکت دست و هدف را برمی‌گرداند
        """
        frame = self.visual_data[idx]
        motion_seq = self.motion_data[idx]
        target = self.target_data[idx]
        
        return frame, motion_seq, target

# ====== کد تست جهت اعتبارسنجی پایپ‌لاین ورود داده‌ها ======
if __name__ == "__main__":
    # ۱. ساخت شیء دیتابیس با ۱۰۰ نمونه فریم فرضی
    dataset = CameraOrganicDataset(num_samples=100, seq_length=10)
    
    # ۲. تعریف DataLoader برای دسته‌بندی دیتابیس به بسته‌های ۴ تایی (Batch Size = 4)
    # shuffle=True یعنی در هر دور آموزش، ترتیب داده‌ها به هم می‌ریزد تا مدل حفظ نکند
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # ۳. تست خروجی گرفتن از لودر داده‌ها
    first_batch = next(iter(train_loader))
    frames, motions, targets = first_batch
    
    print("--- تست لایه لودر داده‌ها (Data Pipeline) با موفقیت انجام شد ---")
    print(f"تعداد کل نمونه‌های موجود در دیتابیس: {len(dataset)}")
    print(f"ابعاد بسته‌ی فریم‌های بازی لود شده: {frames.shape}") # باید [4, 3, 224, 224] باشد
    print(f"ابعاد بسته‌ی توالی حرکتی ماوس: {motions.shape}")       # باید [4, 10, 2] باشد
    print(f"ابعاد بسته‌ی اهداف نهایی (Targets): {targets.shape}")   # باید [4, 2] باشد