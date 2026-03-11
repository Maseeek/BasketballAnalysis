import requests
import os
import sys
import time
import subprocess
import cv2

# Add project root to path so we can import cv_core
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cv_core import BasketballTracker

BASE_URL = "http://127.0.0.1:5001"
SERVER_SCRIPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "server.py")

# Using NEAVid2.mp4 as our standard benchmark video
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEO_PATH = os.path.join(PROJECT_ROOT, "Resources", "Videos", "NEAVid2.mp4")
HOOP_LEFT = [1541, 375]
HOOP_RIGHT = [1646, 348]
MAX_FRAMES = 5000 # Keep same as server.py for fair comparison

def run_core_benchmark(video_path, hoop_left, hoop_right, show_angle=True):
    print("\n--- 1. CORE PROCESSING BENCHMARK (Direct cv_core.py) ---")
    if not os.path.exists(video_path):
        print(f"Error: Could not find {video_path}")
        return

    tracker = BasketballTracker(hoop_left, hoop_right)
    cap = cv2.VideoCapture(video_path)
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_processed = 0
    accuracy = 0.5 if show_angle else 0.15
    skip_interval = int(1/accuracy)

    print(f"Processing {os.path.basename(video_path)} (Total frames: {total_frames}, Skip: {skip_interval})")
    
    start_time = time.time()
    
    while cap.isOpened() and frames_processed < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
            
        frames_processed += 1
        
        # Emulate server.py skipping logic
        if frames_processed % skip_interval != 0:
            continue
            
        tracker.process_frame(frame, debug=False)

    cap.release()
    end_time = time.time()
    
    total_time = end_time - start_time
    fps = frames_processed / total_time if total_time > 0 else 0
    
    print(f"Result: {frames_processed} frames processed in {total_time:.2f} seconds.")
    print(f"Speed:  {fps:.2f} FPS (Frames Per Second)")
    return total_time


def is_server_running():
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=1)
        return r.status_code == 200
    except requests.exceptions.RequestException:
        return False

def start_server():
    print(f"\nStarting {os.path.basename(SERVER_SCRIPT)} in background...")
    process = subprocess.Popen([sys.executable, SERVER_SCRIPT], cwd=PROJECT_ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    for _ in range(20): # Wait up to 10 seconds
        if is_server_running():
            print("Server started successfully.")
            return process
        time.sleep(0.5)
        
    print("Failed to start server.")
    process.terminate()
    return None

def run_server_benchmark(video_path, hoop_left, hoop_right, show_angle=True):
    print("\n--- 2. SERVER ENDPOINT BENCHMARK (End-to-End API) ---")
    
    server_process = None
    if not is_server_running():
        server_process = start_server()
        if not server_process:
            return
    else:
        print("\nUsing already running server instance.")

    if not os.path.exists(video_path):
        print(f"Error: Could not find {video_path}")
        if server_process: server_process.terminate()
        return

    print(f"Sending POST request to {BASE_URL}/upload-and-analyze ...")
    
    try:
        with open(video_path, 'rb') as f:
            files = {'video': (os.path.basename(video_path), f, 'video/mp4')}
            data = {
                'hoopLeft': str(hoop_left),
                'hoopRight': str(hoop_right),
                'showAngle': str(show_angle).lower()
            }
            
            start_time = time.time()
            resp = requests.post(f"{BASE_URL}/upload-and-analyze", files=files, data=data) 
            end_time = time.time()
            
            total_time = end_time - start_time
            
            if resp.status_code == 200 and resp.json().get('success'):
                print(f"Result: Server responded successfully in {total_time:.2f} seconds.")
                print(f"Latency Breakdown: Includes HTTP overhead, file saving, and JSON serialization.")
            else:
                print(f"Server Error (Status {resp.status_code}): {resp.text}")
                
            return total_time
            
    finally:
        if server_process:
            print("Stopping background server...")
            server_process.terminate()
            server_process.wait()

if __name__ == "__main__":
    print(f"=== BASKETBALL ANALYSIS PERFORMANCE BENCHMARK ===")
    
    core_time = run_core_benchmark(VIDEO_PATH, HOOP_LEFT, HOOP_RIGHT)
    server_time = run_server_benchmark(VIDEO_PATH, HOOP_LEFT, HOOP_RIGHT)
    
    if core_time and server_time:
        overhead = server_time - core_time
        print("\n=== SUMMARY ===")
        print(f"Core CV Processing Time: {core_time:.2f} sec")
        print(f"Total API Response Time: {server_time:.2f} sec")
        print(f"Server Routing Overhead: {overhead:.2f} sec ({(overhead/server_time)*100:.1f}%)")
