import torch
import torch.nn as nn

class TemporalBackbone(nn.Module):
    def __init__(self, input_dim=2, hidden_size=64, hidden_dim=128):
        """
        input_dim: ابعاد ورودی در هر فریم که شامل [Delta_X, Delta_Y] حرکت دست است (مقدار ۲)
        hidden_size: تعداد نورون‌های داخلی سلول GRU برای حفظ حافظه حرکتی
        hidden_dim: ابعاد بردار خروجی نهایی که باید با بخش بصری هماهنگ و ۱۲۸ شود
        """
        super(TemporalBackbone, self).__init__()
        
        # ۱. تعریف شبکه بازگشتی GRU
        # batch_first=True یعنی ابعاد ورودی به صورت [Batch_Size, Sequence_Length, Input_Dim] خواهد بود
        self.gru = nn.GRU(
            input_size=input_dim, 
            hidden_size=hidden_size, 
            num_layers=1, 
            batch_first=True
        )
        
        # ۲. لایه خطی انطباق ابعاد (Linear Alignment)
        # تبدیل خروجی حافظه GRU (مثلا ۶۴) به ابعاد ۱۲۸ بعد جهت هماهنگی با لایه توجه
        self.fc_alignment = nn.Linear(hidden_size, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        """
        x: توالی حرکتی دست در فریم‌های گذشته با ابعاد [Batch_Size, Sequence_Length, 2]
        Sequence_Length تعداد فریم‌های گذشته است که مدل به عنوان حافظه به آن‌ها نگاه می‌کند (مثلا ۱۰ فریم گذشته)
        """
        # عبور داده‌ها از سلول بازگشتی GRU 
        # gru_out شامل خروجی تمام گام‌های زمانی است
        gru_out, _ = self.gru(x)
        
        # ما فقط خروجی آخرین فریم (آخرین گام زمانی توالی) را نیاز داریم تا تصمیم نهایی را بگیریم
        last_step_out = gru_out[:, -1, :] # خروجی تبدیل می‌شود به [Batch_Size, hidden_size]
        
        # انطباق ابعاد بردار حرکتی با لایه خطی -> خروجی: [Batch_Size, hidden_dim]
        out = self.fc_alignment(last_step_out)
        out = self.relu(out)
        out = self.dropout(out)
        return out

# ====== کد تست جهت اعتبارسنجی ابعاد خروجی لایه زمانی ======
if __name__ == "__main__":
    # شبیه‌سازی داده‌های حرکتی:
    # فرض می‌کنیم ۴ نمونه (Batch) داریم که در هر کدام، توالی حرکت دست بازیکن در ۱۰ فریم گذشته (Sequence Length)
    # و در هر فریم مقدار جابجایی ماوس [Mouse_X, Mouse_Y] ثبت شده است.
    batch_size = 4
    sequence_length = 10
    input_features = 2 # Delta X , Delta Y
    
    dummy_motion = torch.randn(batch_size, sequence_length, input_features)
    
    # نمونه‌سازی از مدل زمانی با ابعاد خروجی ۱۲۸
    temporal_model = TemporalBackbone(input_dim=2, hidden_size=64, hidden_dim=128)
    
    # پردازش توالی حرکتی توسط مدل
    output_motion_features = temporal_model(dummy_motion)
    
    print("--- تست لایه زمانی GRU با موفقیت انجام شد ---")
    print(f"ابعاد تنسور ورودی حرکت دست (۱۰ فریم گذشته): {dummy_motion.shape}")
    print(f"ابعاد بردار ویژگی خروجی زمانی (آماده برای لایه توجه): {output_motion_features.shape}")
    # خروجی باید [4, 128] باشد که دقیقاً هم‌اندازه با خروجی لایه بصری رزنیت است.