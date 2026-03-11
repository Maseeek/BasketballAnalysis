import cv2
import time
import os
import sys

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cv_core import BasketballTracker

def run_performance_test(video_path, scaled=True, adaptive_skip=True):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0, 0

    # Get first frame for scale calculation
    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        return 0, 0
    
    orig_w = first_frame.shape[1]
    scale = 640.0 / orig_w if scaled else 1.0
    
    # Simple coordinates
    hoop_l = (int(100 * scale), int(100 * scale))
    hoop_r = (int(200 * scale), int(100 * scale))
    tracker = BasketballTracker(hoop_l, hoop_r)
    
    # Setup loop
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_count = 0
    base_skip = 6
    current_skip_target = base_skip
    frames_skipped = 0
    
    start_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame_count += 1
        
        if adaptive_skip:
            if frames_skipped < current_skip_target - 1:
                frames_skipped += 1
                continue
        else:
            # Traditional 6-frame skip
            if frame_count % 6 != 0: continue
            
        if scaled:
            frame = cv2.resize(frame, (640, int(frame.shape[0] * scale)))
            
        tracker.process_frame(frame, debug=False)
        
        if adaptive_skip:
            if tracker.center is not None:
                current_skip_target = base_skip
            else:
                current_skip_target = min(30, current_skip_target + 2)
            frames_skipped = 0
            
    end_time = time.time()
    duration = end_time - start_time
    cap.release()
    
    return frame_count, duration

def main():
    # Use a sample video
    video_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Resources", "Videos")
    video_files = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.mov', '.avi'))]
    
    if not video_files:
        print("No videos found in Resources/Videos/")
        return
        
    video_path = os.path.join(video_dir, video_files[0])
    print(f"Benchmarking: {video_files[0]}\n")
    
    print("Running Unoptimized (Full Res, Simple Skip)...")
    count1, time1 = run_performance_test(video_path, scaled=False, adaptive_skip=False)
    fps1 = count1 / time1 if time1 > 0 else 0
    
    print("Running Optimized (640px Res, Adaptive Skip)...")
    count2, time2 = run_performance_test(video_path, scaled=True, adaptive_skip=True)
    fps2 = count2 / time2 if time2 > 0 else 0
    
    print("\n--- RESULTS ---")
    print(f"Unoptimized: {time1:.2f}s ({fps1:.1f} total-fps)")
    print(f"Optimized:   {time2:.2f}s ({fps2:.1f} total-fps)")
    print(f"Speedup:     {time1/time2:.2f}x")
    print("----------------")

if __name__ == "__main__":
    main()
