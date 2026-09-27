import asyncio
import json
import torch
import torch.nn as nn
import websockets

# ۱. تعریف یک شبکه GRU بسیار ساده برای تست
class SimpleGRUNet(nn.Module):
    def __init__(self):
        super(SimpleGRUNet, self).__init__()
        # ورودی: X و Y ماوس (اندازه ۲) | خروجی مخفی: ۱۶ | تعداد لایه: ۱
        self.gru = nn.GRU(input_size=2, hidden_size=16, num_layers=1, batch_first=True)
        self.fc = nn.Linear(16, 2) # خروجی: جابجایی اصلاح‌شده X و Y دوربین
        
    def forward(self, x, h):
        # x shape: (batch, seq_len, input_size)
        out, h = self.gru(x, h)
        out = self.fc(out[:, -1, :]) # برگرداندن آخرین گام زمانی
        return out, h

# مقداردهی اولیه مدل و حالت مخفی (Hidden State)
model = SimpleGRUNet()
model.eval() # قرار دادن مدل در حالت ارزیابی
hidden = torch.zeros(1, 1, 16) # (num_layers, batch, hidden_size)

print("Model initialized successfully.")

# ۲. مدیریت ارتباط سوکت با آنریل انجین
async def handle_unreal_client(websocket):
    global hidden
    print("Unreal Engine connected!")
    try:
        async for message in websocket:
            # دریافت داده‌های ماوس از آنریل
            data = json.loads(message)
            mouseX = float(data.get("mouseX", 0.0))
            mouseY = float(data.get("mouseY", 0.0))
            
            # تبدیل به تنسور پایتورچ (Batch=1, SeqLen=1, Features=2)
            input_tensor = torch.tensor([[[mouseX, mouseY]]], dtype=torch.float32)
            
            # اجرای مدل بدون محاسبه گرادیان (برای سرعت بالاتر)
            with torch.no_grad():
                prediction, hidden = model(input_tensor, hidden)
            
            # استخراج خروجی
            pred_x = prediction[0][0].item()
            pred_y = prediction[0][1].item()
            
            # ارسال پاسخ به آنریل انجین
            response = {"camX": pred_x, "camY": pred_y}
            await websocket.send(json.dumps(response))
            
    except websockets.exceptions.ConnectionClosed:
        print("Unreal Engine disconnected.")
        # بازنشانی حالت مخفی برای اتصال بعدی
        hidden = torch.zeros(1, 1, 16)

# راه‌اندازی سرور روی پورت 8080
async def main():
    server = await websockets.serve(handle_unreal_client, "localhost", 8080)
    print("Python Socket Server running on ws://localhost:8080")
    await server.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())