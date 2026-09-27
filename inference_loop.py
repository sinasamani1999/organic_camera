import time
import torch
from pythonosc import udp_client

# Import the integrated network structure
from models.Organic_Camera_Model import OrganicCameraModel

def run_live_inference():
    # 1. Device configuration setup (CUDA GPU or CPU fallback)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Live Inference Engine initializing on device: {device}")

    # 2. Network connection setup for Open Sound Control (OSC) via UDP Localhost
    target_ip = "127.0.0.1"
    target_port = 8000
    osc_address = "/camera"
    
    try:
        client = udp_client.SimpleUDPClient(target_ip, target_port)
        print(f"📡 OSC Network Client successfully bound to {target_ip}:{target_port} on path '{osc_address}'")
    except Exception as e:
        print(f"❌ Failed to initialize OSC network client: {e}")
        return

    # 3. Instantiate the model core and transition to evaluation mode
    # hidden_dim=128 matches our thesis structural architectural specification
    model = OrganicCameraModel(hidden_dim=128).to(device)
    model.eval() 
    print("🧠 Integrated Neural Network model instantiated with initialization states.")
    print("⚠️ Note: Currently utilizing initialization weights for end-to-end network piping verification.")

    # 4. Target inference performance parameters (30 Hz refresh rate ceiling)
    target_fps = 30
    frame_duration = 1.0 / target_fps
    print(f"⏱️ Operational tick rate locked at {target_fps} FPS (Interval: {frame_duration:.4f}s)")
    print("🎬 Starting live inference simulation stream loop. Press Ctrl+C to terminate...")

    try:
        while True:
            start_time = time.time()

            # 5. Synthesize Mock Data mimicking the live engine capture stream
            # Fabricated single frame image tensor matching ResNet shape: [Batch=1, Channels=3, Height=224, Width=224]
            mock_frame_tensor = torch.randn(1, 3, 224, 224).to(device)
            
            # Fabricated historical motion sequence vector matching GRU shape: [Batch=1, Sequence=10, Features=2]
            mock_motion_tensor = torch.randn(1, 10, 2).to(device)

            # 6. Execute forward pass context calculation without keeping gradient tracking graphs
            with torch.no_grad():
                # Extract predicted Delta Yaw and Delta Pitch from cross-attention fusion network
                predictions = model(mock_frame_tensor, mock_motion_tensor)
                
            # Convert GPU multi-dimensional tensor back into local CPU linear floating-point primitives
            predicted_angles = predictions.squeeze(0).cpu().tolist()
            predicted_yaw = float(predicted_angles[0])
            predicted_pitch = float(predicted_angles[1])

            # 7. Package and fire spatial variables into the local loopback network interface
            client.send_message(osc_address, [predicted_yaw, predicted_pitch])
            
            # 8. Frame rate pacing control calculation
            elapsed_time = time.time() - start_time
            sleep_time = frame_duration - elapsed_time
            
            if sleep_time > 0:
                time.sleep(sleep_time)
                
            # Log periodic network stream telemetry updates
            current_hz = 1.0 / (time.time() - start_time)
            print(f"🛰️ [Streaming] Outbound Send -> Address: {osc_address} | Yaw: {predicted_yaw:+0.4f} | Pitch: {predicted_pitch:+0.4f} | Net Sync Frequency: {current_hz:.1f} Hz", end="\r")

    except KeyboardInterrupt:
        print("\n🛑 Live inference simulation stream loop terminated by user request safely.")

if __name__ == "__main__":
    # Prior to running, verify that 'python-osc' is installed inside your active python virtual environment
    # Installation command: pip install python-osc
    run_live_inference()