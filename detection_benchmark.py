import cv2
import numpy as np
import time
from cv_core import BasketballTracker, dist

def run_detection_benchmark(video_path, hoop_left, hoop_right):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open {video_path}")
        return

    tracker = BasketballTracker(hoop_left, hoop_right)
    
    total_frames = 0
    detected_frames = 0
    jump_distances = []
    
    print(f"Benchmarking detection on: {video_path}")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        total_frames += 1
        prev_pos = tracker.center
        
        # We only call find_ball to isolate detection performance/accuracy
        ball = tracker.find_ball(frame)
        
        if ball is not None:
            detected_frames += 1
            curr_pos = (ball[0], ball[1])
            
            if prev_pos is not None:
                d = np.sqrt(dist(curr_pos[0], curr_pos[1], prev_pos[0], prev_pos[1]))
                jump_distances.append(d)
                
            # Update tracker state so ROI tracking works in next frame
            tracker.prev_circle = (ball[0], ball[1], ball[2])
            tracker.center = curr_pos
        else:
            # Optionally reset prev_circle to force full-frame search next time
            # tracker.prev_circle = None 
            pass

    cap.release()
    
    detection_rate = (detected_frames / total_frames) * 100 if total_frames > 0 else 0
    avg_jump = np.mean(jump_distances) if jump_distances else 0
    max_jump = np.max(jump_distances) if jump_distances else 0
    
    print("\n--- DETECTION BENCHMARK RESULTS ---")
    print(f"Total Frames:      {total_frames}")
    print(f"Detected Frames:   {detected_frames}")
    print(f"Detection Rate:    {detection_rate:.2f}%")
    print(f"Average Movement:  {avg_jump:.2f} px/frame")
    print(f"Max Jump:          {max_jump:.2f} px")
    print("-----------------------------------\n")
    
    if detection_rate < 80:
        print("ADVICE: Detection rate is low. Consider adjusting param2 in HoughCircles or Gaussian Blur size.")
    if max_jump > 200:
        print("ADVICE: High Max Jump detected. You likely have 'false positives' (detecting background as ball).")

if __name__ == "__main__":
    # Example usage with one of the user's videos
    # Values taken from ground_truth.json for "video.mp4"
    VIDEO = "video.mp4"
    HOOP_L = (118, 541)
    HOOP_R = (295, 583)
    
    run_detection_benchmark(VIDEO, HOOP_L, HOOP_R)
