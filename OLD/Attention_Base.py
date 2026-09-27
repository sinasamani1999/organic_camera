import torch
import torch.nn as nn
import math

class CrossAttention(nn.Module):
    def __init__(self, feature_dim=128):
        """
        feature_dim: ابعاد بردارهای ورودی (CNN و GRU) که در گام‌های قبل ۱۲۸ تنظیم کردیم
        """
        super(CrossAttention, self).__init__()
        
        # تعریف لایه‌های خطی برای ساخت مکانیزم Query، Key و Value
        self.q_linear = nn.Linear(feature_dim, feature_dim)
        self.k_linear = nn.Linear(feature_dim, feature_dim)
        self.v_linear = nn.Linear(feature_dim, feature_dim)
        
        # لایه نرمال‌سازی سافت‌مکس برای تبدیل امتیازها به درصد و احتمال (بین ۰ و ۱)
        self.softmax = nn.Softmax(dim=-1)
        
        # لایه خروجی برای ترکیب نهایی
        self.out_linear = nn.Linear(feature_dim, feature_dim)

    def forward(self, visual_features, motion_features):
        """
        visual_features (CNN): [Batch_Size, feature_dim] -> به عنوان Key و Value
        motion_features (GRU): [Batch_Size, feature_dim] -> به عنوان Query
        """
        
        # ۱. تولید بردارهای Q، K و V
        # برای ضرب ماتریسی کراس‌اتنشن، نیاز داریم ابعاد را کمی باز کنیم (اضافه کردن بُعد توالی فرضی)
        Q = self.q_linear(motion_features).unsqueeze(1)  # ابعاد: [Batch_Size, 1, feature_dim]
        K = self.k_linear(visual_features).unsqueeze(1)  # ابعاد: [Batch_Size, 1, feature_dim]
        V = self.v_linear(visual_features).unsqueeze(1)  # ابعاد: [Batch_Size, 1, feature_dim]
        
        # ۲. محاسبه امتیاز توجه (ضرب ماتریسی Query در ترانهاده Key)
        # تقسیم بر رادیکال ابعاد (Scaled Dot-Product) برای پایداری گرادیان در فاز تمرین
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.size(-1))
        
        # ۳. اعمال Softmax برای به دست آوردن وزن‌های توجه (Attention Weights)
        attention_weights = self.softmax(scores)
        
        # ۴. ضرب وزن‌های توجه در بردار Value برای استخراج محتوای همگام‌شده
        context = torch.matmul(attention_weights, V)  # ابعاد: [Batch_Size, 1, feature_dim]
        
        # ۵. حذف بعد فرضی و عبور از لایه خطی پایانی
        context = context.squeeze(1)  # ابعاد: [Batch_Size, feature_dim]
        output = self.out_linear(context)
        
        return output

# ====== کد تست جهت اعتبارسنجی ابعاد خروجی لایه کراس‌اتنشن ======
if __name__ == "__main__":
    # شبیه‌سازی خروجی‌های گام ۲.۱ و ۲.۲ (۴ نمونه با ویژگی‌های ۱۲۸ بعدی)
    batch_size = 4
    dim = 128
    
    dummy_visual_out = torch.randn(batch_size, dim)  # خروجی فرضی CNN
    dummy_motion_out = torch.randn(batch_size, dim)  # خروجی فرضی GRU
    
    # نمونه‌سازی از کلاس توجه دوگانه
    attention_layer = CrossAttention(feature_dim=128)
    
    # ادغام هوشمند دو لایه بصری و زمانی
    fused_features = attention_layer(dummy_visual_out, dummy_motion_out)
    
    print("--- تست لایه واسط توجه دوگانه (Cross-Attention) با موفقیت انجام شد ---")
    print(f"ابعاد بردار ویژگی بصری ورودی: {dummy_visual_out.shape}")
    print(f"ابعاد بردار ویژگی زمانی ورودی: {dummy_motion_out.shape}")
    print(f"ابعاد بردار ادغام‌شده خروجی (Fused Features): {fused_features.shape}")
    # خروجی باید [4, 128] باشد که یعنی دو حس بینایی و حرکت به طور هوشمند با هم ترکیب شده‌اند.
