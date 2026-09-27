import cv2
import numpy as np
import csv

def extract_motion_to_csv(video_path, output_csv_path):
    # 1. Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Read the first frame
    ret, first_frame = cap.read()
    if not ret:
        print("Error loading video!")
        return

    # Convert the first frame to grayscale
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    
    # 2. Prepare the CSV file for writing data
    with open(output_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write CSV headers
        writer.writerow(["Frame_ID", "Delta_X", "Delta_Y"])
        
        frame_id = 1
        
        # 3. Frame-by-frame video processing loop
        while True:
            ret, frame = cap.read()
            if not ret:
                break # End of video
            
            # Convert the current frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 4. Calculate Dense Optical Flow between the previous and current frame
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None, 
                pyr_scale=0.5, levels=3, winsize=15, 
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            
            # 'flow' is a matrix containing [dx, dy] for each individual pixel.
            # Take the mean movement across all pixels to calculate global camera motion.
            avg_dx = np.mean(flow[..., 0])
            avg_dy = np.mean(flow[..., 1])
            
            # 5. Save information to the CSV file
            writer.writerow([frame_id, avg_dx, avg_dy])
            
            # Update the previous frame for the next iteration
            prev_gray = gray
            frame_id += 1

    cap.release()
    print(f" Motion extraction completed successfully! Motion data saved to: {output_csv_path}")

# ====== Experimental Execution ======
if __name__ == "__main__":
    # Provide your actual video path in the real project
    # extract_motion_to_csv("my_gameplay.mp4", "motion_data.csv")
    print("Script is ready. Pass a video path to generate the CSV file.")