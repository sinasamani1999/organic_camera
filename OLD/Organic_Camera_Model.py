import torch
import torch.nn as nn

# وارد کردن کلاس‌هایی که در گام‌های قبل به طور مجزا ساختیم
from CNN_Base import VisualBackbone
from Temporal_Base import TemporalBackbone
from Attention_Base import CrossAttention

class OrganicCameraModel(nn.Module):
    def __init__(self, hidden_dim=128):
        super(OrganicCameraModel, self).__init__()
        
        # ۱. فراخوانی لایه استخراج ویژگی‌های بصری (فاز ۲.۱)
        self.visual_backbone = VisualBackbone(hidden_dim=hidden_dim)
        
        # ۲. فراخوانی لایه پردازش زمانی حرکت دست (فاز ۲.۲)
        self.temporal_backbone = TemporalBackbone(input_dim=2, hidden_size=64, hidden_dim=hidden_dim)
        
        # ۳. فراخوانی لایه واسط توجه دوگانه (فاز ۲.۳)
        self.cross_attention = CrossAttention(feature_dim=hidden_dim)
        
        # ۴. لایه خطی خروجی (Regression Layer)
        # این لایه بردار ۱۲۸ بعدی تلفیق‌شده را به ۲ عدد نهایی تبدیل می‌کند: [Predicted_Cam_X, Predicted_Cam_Y]
        self.fc_out = nn.Linear(hidden_dim, 2)

    def forward(self, video_frame, motion_sequence):
        """
        video_frame: [Batch_Size, 3, 224, 224] -> فریم تصویر بازی
        motion_sequence: [Batch_Size, Sequence_Length, 2] -> توالی جابجایی دست در فریم‌های گذشته
        """
        # الف) استخراج ویژگی‌های بصری صحنه بازی
        vis_feats = self.visual_backbone(video_frame) # خروجی: [Batch_Size, 128]
        
        # ب) استخراج ویژگی‌های داینامیک و شتاب حرکت دست
        motion_feats = self.temporal_backbone(motion_sequence) # خروجی: [Batch_Size, 128]
        
        # ج) ادغام هوشمند دید و حرکت با مکانیزم کراس‌اتنشن
        fused_feats = self.cross_attention(vis_feats, motion_feats) # خروجی: [Batch_Size, 128]
        
        # د) محاسبه خروجی نهایی برای زاویه دید دوربین
        camera_prediction = self.fc_out(fused_feats) # خروجی: [Batch_Size, 2]
        
        return camera_prediction

# ====== کد تست کل شبکه یکپارچه هوش مصنوعی ======
if __name__ == "__main__":
    # شبیه‌سازی داده‌های ورودی هم‌زمان (۴ نمونه کاملاً موازی)
    batch_size = 4
    seq_len = 10 # ۱۰ فریم گذشته
    
    dummy_frames = torch.randn(batch_size, 3, 224, 224) # فریم‌های بازی
    dummy_motions = torch.randn(batch_size, seq_len, 2)  # دیتای ماوس بازیکن
    
    # نمونه‌سازی از کل مدل ترکیبی پروپوزال
    full_model = OrganicCameraModel(hidden_dim=128)
    
    # پردازش هم‌زمان و گرفتن خروجی کنترلر دوربین
    predicted_camera_move = full_model(dummy_frames, dummy_motions)
    
    print("========================================================")
    print("🟢 تبریک! کل معماری شبکه ترکیبی CNN-GRU + Cross-Attention با موفقیت ساخته شد.")
    print("========================================================")
    print(f"ابعاد فریم‌های ورودی بازی: {dummy_frames.shape}")
    print(f"ابعاد دیتای سری زمانی ماوس: {dummy_motions.shape}")
    print(f"ابعاد تنسور خروجی نهایی (پیش‌بینی حرکت ارگانیک دوربین): {predicted_camera_move.shape}")
    print("--- نمونه خروجی عددی مدل برای یک فریم (CamX , CamY) ---")
    print(predicted_camera_move[0].detach().numpy())