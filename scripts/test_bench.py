import cv2
import json
import argparse
import os
import glob
import sys

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cv_core import BasketballTracker

GT_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ground_truth.json")
VIDEO_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Resources", "Videos")

def get_click_coordinates(frame, window_name):
    click_x, click_y = -1, -1
    clicked = False

    def mouse_click(event, x, y, flags, param):
        nonlocal click_x, click_y, clicked
        if event == cv2.EVENT_LBUTTONDOWN:
            click_x, click_y = x, y
            clicked = True

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_click)
    cv2.imshow(window_name, frame)
    
    # Bring window to front (OS dependent, but helpful)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

    print(f"Please click: {window_name}")
    while not clicked:
        cv2.waitKey(1)
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            print("Window closed before selection.")
            return None
            
    cv2.destroyWindow(window_name)
    return [click_x, click_y]

def run_setup(video_dir):
    data = {}
    if os.path.exists(GT_FILE):
        with open(GT_FILE, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {}
            
    # Find videos
    extensions = ['*.mp4', '*.avi', '*.mov']
    video_files = []
    for ext in extensions:
        video_files.extend(glob.glob(os.path.join(video_dir, ext)))
        
    print(f"Found {len(video_files)} videos in {video_dir}")
    
    for video_path in video_files:
        video_key = os.path.abspath(video_path)
        
        if video_key in data:
            # Check if it's the old format (missing 'sequence')
            if 'sequence' in data[video_key]:
                print(f"Skipping {os.path.basename(video_path)} (already setup).")
                continue
            else:
                print(f"Updating format for {os.path.basename(video_path)}...")
            
        print(f"\n--- Setting up {os.path.basename(video_path)} ---")
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print("Error reading video. Skipping.")
            continue
            
        print("Click LEFT side of hoop...")
        hoop_left = get_click_coordinates(frame, "Set LEFT Side of Hoop")
        if hoop_left is None: continue

        print("Click RIGHT side of hoop...")
        hoop_right = get_click_coordinates(frame, "Set RIGHT Side of Hoop")
        if hoop_right is None: continue
        
        print("\nEnter Shot Sequence (1=Make, 0=Miss).")
        print("Example: '101' means Make, then Miss, then Make.")
        valid_input = False
        sequence = []
        
        while not valid_input:
            seq_str = input("Sequence: ").strip()
            if all(c in '01' for c in seq_str):
                sequence = [int(c) for c in seq_str]
                valid_input = True
            else:
                print("Invalid. Please enter only 1s and 0s (e.g., 1001).")
            
        data[video_key] = {
            "hoop_left": hoop_left,
            "hoop_right": hoop_right,
            "sequence": sequence
        }
        
        # Save incrementally
        with open(GT_FILE, 'w') as f:
            json.dump(data, f, indent=4)
            print("Saved.")

def run_test():
    if not os.path.exists(GT_FILE):
        print("No ground_truth.json found. Run with --setup first.")
        return

    with open(GT_FILE, 'r') as f:
        data = json.load(f)
        
    total_videos = 0
    perfect_videos = 0
    
    # Header
    print(f"\n{'VIDEO':<40} | {'ACTUAL':<15} | {'PREDICTED':<15} | {'STATUS'}")
    print("-" * 85)
    
    for video_path, gt in data.items():
        if not os.path.exists(video_path):
            print(f"Video not found: {os.path.basename(video_path)}")
            continue
            
        # Support fallback to old format if user didn't re-run setup
        if 'sequence' not in gt:
            print(f"{os.path.basename(video_path):<40} | OLD FORMAT     | -               | SKIP")
            continue

        cap = cv2.VideoCapture(video_path)
        ret, first_frame = cap.read()
        if not ret:
            cap.release()
            continue

        scale = 640.0 / first_frame.shape[1]
        scaled_hoop_left = (int(gt['hoop_left'][0] * scale), int(gt['hoop_left'][1] * scale))
        scaled_hoop_right = (int(gt['hoop_right'][0] * scale), int(gt['hoop_right'][1] * scale))

        tracker = BasketballTracker(scaled_hoop_left, scaled_hoop_right)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        base_skip = int(1/0.15)  # Following server's default accuracy
        current_skip_target = base_skip
        frames_skipped = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            if frames_skipped < current_skip_target - 1:
                frames_skipped += 1
                continue

            resized_frame = cv2.resize(frame, (640, int(frame.shape[0] * scale)))
            tracker.process_frame(resized_frame, debug=False)

            if tracker.center is not None:
                current_skip_target = base_skip
            else:
                current_skip_target = min(30, current_skip_target + 2)
            frames_skipped = 0
            
        cap.release()
        
        # Compare Sequences
        actual_seq = gt['sequence']
        pred_seq = tracker.shots
        
        actual_str = "".join(map(str, actual_seq))
        pred_str = "".join(map(str, pred_seq))
        
        # Check strict equality
        if actual_seq == pred_seq:
            status = "PASS"
            match = True
        else:
            status = "FAIL"
            match = False
        
        print(f"{os.path.basename(video_path):<40} | {actual_str:<15} | {pred_str:<15} | {status}")
        
        total_videos += 1
        if match: perfect_videos += 1
        
    print("-" * 85)
    if total_videos > 0:
        accuracy_pct = (perfect_videos / total_videos) * 100
        print(f"Accuracy: {perfect_videos}/{total_videos} ({accuracy_pct:.1f}%)")
        
        # Regression Threshold: fail the test if accuracy is too low
        if accuracy_pct < 80.0:
            print("ERROR: Accuracy fell below 80% regression threshold!")
            sys.exit(1)
    else:
        print("No valid videos tested.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Basketball CV Test Bench")
    parser.add_argument("--setup", action="store_true", help="Run setup mode to populate ground truth")
    parser.add_argument("--test", action="store_true", help="Run test mode to evaluate accuracy")
    parser.add_argument("--dir", default=VIDEO_DIR, help="Directory containing videos")
    
    args = parser.parse_args()
    
    if args.setup:
        run_setup(args.dir)
    elif args.test:
        run_test()
    else:
        parser.print_help()
