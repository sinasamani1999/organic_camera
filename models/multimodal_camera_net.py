import torch
import torch.nn as nn
import torchvision.models as models

class MultimodalCameraNet(nn.Module):
    def __init__(self, visual_feature_dim=2048, motion_dim=2, hidden_dim=256, num_heads=4):
        super(MultimodalCameraNet, self).__init__()
        
        # ۱. شاخه پردازش بینایی (ResNet)
        # استفاده از لایه‌های پیش‌آموزش‌دیده تا قبل از لایه Pooling نهایی
        resnet = models.resnet50(pretrained=True)
        self.visual_backbone = nn.Sequential(*list(resnet.children())[:-2])
        
        # ۲. شاخه پردازش زمانی حرکتی (GRU)
        self.motion_gru = nn.GRU(
            input_size=motion_dim, 
            hidden_size=hidden_dim, 
            num_layers=2, 
            batch_first=True
        )
        
        # ۳. لایه نگاشت خطی برای هم‌بعدسازی (Linear Projection)
        # خروجی رزنیت [Batch, 2048, 7, 7] است که به [Batch, 49, 2048] تبدیل می‌شود
        # ما لایه تصویری را به بعد hidden_dim می‌آوریم تا با GRU هم‌تراز شود
        self.visual_projection = nn.Linear(visual_feature_dim, hidden_dim)
        
        # ۴. لایه توجه متقاطع (Cross-Attention)
        # Query از GRU می‌آید و Key/Value از ResNet
        self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        
        # ۵. لایه رگرسیون نهایی برای پیش‌بینی Yaw و Pitch
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2) # خروجی: [Yaw, Pitch]
        )

    def forward(self, visual_frames, motion_sequences):
        # visual_frames shape: [Batch, Sequence_Len, 3, 224, 224]
        # motion_sequences shape: [Batch, Sequence_Len, 2] (Yaw/Pitch های قبلی)
        
        batch_size, seq_len, c, h, w = visual_frames.size()
        
        # استخراج ویژگی‌های تصویری فریم فعلی (آخرین فریم توالی)
        current_frame = visual_frames[:, -1, :, :, :] # [Batch, 3, 224, 224]
        visual_feat = self.visual_backbone(current_frame) # [Batch, 2048, 7, 7]
        
        # تغییر شکل برای ورود به اتنشن
        visual_feat = visual_feat.view(batch_size, 2048, -1).permute(0, 2, 1) # [Batch, 49, 2048]
        visual_projected = self.visual_projection(visual_feat) # [Batch, 49, hidden_dim] -> کلید و مقدار (K, V)
        
        # پردازش سری زمانی حرکتی با GRU
        gru_out, _ = self.motion_gru(motion_sequences) # [Batch, Sequence_Len, hidden_dim]
        current_motion_feat = gru_out[:, -1, :].unsqueeze(1) # [Batch, 1, hidden_dim] -> پرس‌وجو (Q)
        
        # اعمال توجه متقاطع (Cross-Attention)
        attn_output, _ = self.cross_attention(
            query=current_motion_feat, 
            key=visual_projected, 
            value=visual_projected
        ) # [Batch, 1, hidden_dim]
        
        # حذف بعد اضافه و ورود به رگرسور
        attn_output = attn_output.squeeze(1) # [Batch, hidden_dim]
        predicted_angles = self.regressor(attn_output) # [Batch, 2] -> [Predicted_Yaw, Predicted_Pitch]
        
        return predicted_angles