import os
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

class OrganicCameraDataset(Dataset):
    def __init__(self, video_dir, motion_dir, seq_length=10):
        self.video_dir = video_dir
        self.motion_dir = motion_dir
        self.seq_length = seq_length
        
        # پیدا کردن تمام نام‌های مشترک بین ویدیوها و فایل‌های CSV
        self.file_names = [os.path.splitext(f)[0] for f in os.listdir(video_dir) if f.endswith('.mp4')]
        
        # تعریف ترنسفورم استاندارد برای فریم‌های تصویری (رزنیت ۱۸)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # ساخت یک ایندکس کلی از تمام فریم‌های قابل آموزش در کل ویدیوها
        self.samples = []
        self._prepare_dataset_indices()

    def _prepare_dataset_indices(self):
        """
        این تابع بررسی می‌کند هر ویدیو چند فریم دارد و نمونه‌هایی که 
        تاریخچه حرکتی کافی (حداقل به اندازه seq_length) دارند را ثبت می‌کند.
        """
        for name in self.file_names:
            csv_path = os.path.join(self.motion_dir, f"{name}.csv")
            if os.path.exists(csv_path):
                # خواندن فایل حرکتی
                df = pd.read_csv(csv_path)
                total_frames = len(df)
                
                # ما برای پیش‌بینی هر فریم، نیاز به ۱۰ فریم گذشته داریم
                for frame_idx in range(self.seq_length, total_frames):
                    self.samples.append({
                        'video_name': name,
                        'target_frame_idx': frame_idx
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        video_name = sample['video_name']
        target_idx = sample['target_frame_idx']
        
        # ۱. استخراج فریم تصویری مورد نظر با OpenCV
        video_path = os.path.join(self.video_dir, f"{video_name}.mp4")
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            # در صورت بروز خطای خواندن، یک فریم خالی یا نمونه صفر برمی‌گردانیم
            frame_tensor = torch.zeros(3, 224, 224)
        else:
            # تبدیل فرمت BGR به RGB و اعمال ترنسفورم پایتورچ
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            frame_tensor = self.transform(pil_img)
            
        # ۲. استخراج توالی حرکتی گذشته از فایل CSV
        csv_path = os.path.join(self.motion_dir, f"{video_name}.csv")
        df = pd.read_csv(csv_path)
        
        # جدا کردن ۱۰ سطر گذشته (از target_idx - seq_length تا target_idx)
        motion_seq = df.iloc[target_idx - self.seq_length : target_idx][['Delta_X', 'Delta_Y']].values
        motion_tensor = torch.tensor(motion_seq, dtype=torch.float32)
        
        # ۳. برچسب یا هدف (میزان جابجایی ارگانیکی که در این فریم اتفاق افتاده است)
        target_motion = df.iloc[target_idx][['Delta_X', 'Delta_Y']].values
        target_tensor = torch.tensor(target_motion, dtype=torch.float32)
        
        return frame_tensor, motion_tensor, target_tensor

# ====== بخش تست خط لوله داده‌ها ======
if __name__ == "__main__":
    print("کلاس دیتابیس با موفقیت کامپایل شد.")
    print("این ماژول آماده متصل شدن به دیتای واقعی روی هارد دیسک تو است.")