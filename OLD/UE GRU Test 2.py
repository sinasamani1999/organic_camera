from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import torch
import torch.nn as nn

# ۱. تعریف شبکه GRU ساده
class SimpleGRUNet(nn.Module):
    def __init__(self):
        super(SimpleGRUNet, self).__init__()
        self.gru = nn.GRU(input_size=2, hidden_size=16, num_layers=1, batch_first=True)
        self.fc = nn.Linear(16, 2)
        
    def forward(self, x, h):
        out, h = self.gru(x, h)
        out = self.fc(out[:, -1, :])
        return out, h

model = SimpleGRUNet()
model.eval()
hidden = torch.zeros(1, 1, 16)

print("Model initialized successfully for HTTP.")

# ۲. تعریف رفتار سرور در پاسخ به درخواست‌های آنریل
class CameraAIServer(BaseHTTPRequestHandler):
    def do_POST(self):
        global hidden
        
        # خواندن دیتای ارسالی از آنریل
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        
        # پارس کردن JSON
        data = json.loads(post_data.decode('utf-8'))
        mouseX = float(data.get("mouseX", 0.0))
        mouseY = float(data.get("mouseY", 0.0))
        
        # ورودی به پایتورچ
        input_tensor = torch.tensor([[[mouseX, mouseY]]], dtype=torch.float32)
        
        with torch.no_grad():
            prediction, hidden = model(input_tensor, hidden)
        
        pred_x = prediction[0][0].item()
        pred_y = prediction[0][1].item()
        
        # ساخت پاسخ JSON
        response_data = json.dumps({"camX": pred_x, "camY": pred_y}).encode('utf-8')
        
        # ارسال پاسخ HTTP به آنریل
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(response_data)

# راه‌اندازی سرور روی پورت 8080
# راه‌اندازی سرور روی پورت جدید 8090
def run():
    server_address = ('127.0.0.1', 8090) # پورت به 8090 تغییر یافت
    httpd = HTTPServer(server_address, CameraAIServer)
    print("Python HTTP Server running on http://127.0.0.1:8090 ...")
    httpd.serve_forever()

if __name__ == '__main__':
    run()