import torch
import torch.nn as nn
import torchvision.models as models
import os

class VisualBackbone(nn.Module):
    def __init__(self, hidden_dim=128):
        """
        بارگذاری کاملاً آفلاین وزن‌های پیش‌تمرین‌دیده ResNet-18 برای استخراج ویژگی‌های بصری بازی
        """
        super(VisualBackbone, self).__init__()
        
        # ۱. ساخت یک مدل رزنیت خام و بدون وزن (وزن‌ها را اینترنت لود نمی‌کند)
        resnet = models.resnet18(weights=None)
        
        # ۲. نام دقیق فایلی که دانلود کردی و کنار این کد گذاشتی
        weights_path = "resnet18-f37072fd.pth"
        
        if os.path.exists(weights_path):
            # بارگذاری آفلاین وزن‌ها از روی هارد سیستم به حافظه (CPU/GPU)
            state_dict = torch.load(weights_path, map_location=torch.device('cpu'))
            resnet.load_state_dict(state_dict)
            print("🟢 وزن‌های رسمی ResNet-18 به صورت ۱۰۰٪ آفلاین و موفقیت‌آمیز بارگذاری شدند.")
        else:
            raise FileNotFoundError(
                f"❌ خطا: فایل وزن‌ها پیدا نشد! مطمئن شو فایل '{weights_path}' دقیقاً کنار این کد پایتون قرار دارد."
            )
        
        # ۳. حذف لایه‌های نهایی طبقه‌بندی رزنیت (لایه‌های AvgPool و FC نهایی)
        # ما فقط بخش کانولوشنی استخراج ویژگی را نگه می‌داریم
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        
        # ۴. لایه خودکار کاهش ابعاد فضایی به 1x1 (Adaptive Average Pooling)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # ۵. لایه خطی انطباق ابعاد (Linear Embedding)
        # بردار ۵۱۲ تایی خروجی رزنیت را به ابعاد Hidden Dimension مدنظر ما (مثلا ۱۲۸) نگاشت می‌کند
        self.fc_alignment = nn.Linear(512, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        """
        x: فریم‌های ورودی بازی با ابعاد [Batch_Size, 3, 224, 224]
        """
        # عبور تصویر از لایه‌های کانولوشنی رزنیت -> خروجی: [Batch_Size, 512, 7, 7]
        x = self.features(x)
        
        # فشرده‌سازی ابعاد فضایی -> خروجی: [Batch_Size, 512, 1, 1]
        x = self.global_pool(x)
        
        # تخت کردن ماتریس به یک بردار تک‌بعدی -> خروجی: [Batch_Size, 512]
        x = torch.flatten(x, 1)
        
        # انطباق ابعاد با لایه خطی جهت هماهنگی با شبکه GRU -> خروجی: [Batch_Size, hidden_dim]
        x = self.fc_alignment(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x

# ====== کد تست جهت اعتبارسنجی ابعاد خروجی لایه بصری ======
if __name__ == "__main__":
    # شبیه‌سازی ورود ۴ فریم از محیط آنریل انجین با ابعاد استاندارد ۲۲۴ در ۲۲۴
    dummy_frames = torch.randn(4, 3, 224, 224)
    
    try:
        # نمونه‌سازی از مدل با ابعاد خروجی ۱۲۸ بعد
        visual_model = VisualBackbone(hidden_dim=128)
        
        # پردازش فریم‌ها توسط مدل
        output_features = visual_model(dummy_frames)
        
        print("\n--- successful")
        print(f"tensor size: {dummy_frames.shape}")
        print(f"exported tensor size(ready for attention layer): {output_features.shape}")
        # خروجی باید [4, 128] باشد؛ یعنی لایه به ازای هر فریم، یک بردار ویژگی غنی ۱۲۸ عددی ساخته است.
        
    except Exception as e:
        print(f"fail: {e}")