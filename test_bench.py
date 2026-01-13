import cv2
import json
import argparse
import os
import glob
from cv_core import BasketballTracker

GT_FILE = "ground_truth.json"
VIDEO_DIR = r"Resources/Videos"

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
            data = json.load(f)
            
    # Find videos
    extensions = ['*.mp4', '*.avi', '*.mov']
    video_files = []
    for ext in extensions:
        video_files.extend(glob.glob(os.path.join(video_dir, ext)))
        
    print(f"Found {len(video_files)} videos in {video_dir}")
    
    for video_path in video_files:
        video_key = os.path.abspath(video_path)
        
        if video_key in data:
            print(f"Skipping {video_path} (already in DB).")
            continue
            
        print(f"\n--- Setting up {video_path} ---")
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
        
        try:
            makes = int(input("Enter actual MAKES: "))
            misses = int(input("Enter actual MISSES: "))
        except ValueError:
            print("Invalid input. Skipping.")
            continue
            
        data[video_key] = {
            "hoop_left": hoop_left,
            "hoop_right": hoop_right,
            "makes": makes,
            "misses": misses,
            "total_shots": makes + misses
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
    total_shots_actual = 0
    total_shots_predicted = 0
    correct_shots = 0 # This is a rough metric since we don't match specific shots yet
    
    print(f"\n{'VIDEO':<50} | {'ACTUAL':<10} | {'PREDICTED':<10} | {'STATUS'}")
    print("-" * 90)
    
    for video_path, gt in data.items():
        if not os.path.exists(video_path):
            print(f"Video not found: {video_path}")
            continue
            
        tracker = BasketballTracker(gt['hoop_left'], gt['hoop_right'])
        cap = cv2.VideoCapture(video_path)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            tracker.process_frame(frame, debug=False)
            
        cap.release()
        
        # Scoring
        actual_str = f"{gt['makes']}/{gt['misses']}"
        pred_misses = tracker.fga - tracker.fgm
        pred_str = f"{tracker.fgm}/{pred_misses}"
        
        match = (gt['makes'] == tracker.fgm) and (gt['misses'] == pred_misses)
        status = "PASS" if match else "FAIL"
        
        print(f"{os.path.basename(video_path):<50} | {actual_str:<10} | {pred_str:<10} | {status}")
        
        total_videos += 1
        if match: perfect_videos += 1
        
    print("-" * 90)
    print(f"Overall Accuracy (Perfect Videos): {perfect_videos}/{total_videos} ({perfect_videos/total_videos*100:.1f}%)" if total_videos > 0 else "No videos tested.")

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
