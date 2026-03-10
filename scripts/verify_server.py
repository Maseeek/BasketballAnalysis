import requests
import cv2
import numpy as np
import time
import sys
import subprocess
import os
import signal

BASE_URL = "http://127.0.0.1:5001"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVER_SCRIPT = os.path.join(PROJECT_ROOT, "server.py")

def is_server_running():
    try:
        r = requests.get(f"{BASE_URL}/health")
        return r.status_code == 200
    except:
        return False

def start_server():
    print(f"Starting {SERVER_SCRIPT}...")
    # Start server as a subprocess
    process = subprocess.Popen([sys.executable, SERVER_SCRIPT], cwd=PROJECT_ROOT)
    
    # Wait for it to come up
    for i in range(20):
        if is_server_running():
            print("Server started successfully.")
            return process
        time.sleep(0.5)
        
    print("Failed to start server.")
    process.terminate()
    return None

def test_server():
    server_process = None
    if not is_server_running():
        server_process = start_server()
        if not server_process:
            sys.exit(1)
    else:
        print("Server already running, using existing instance.")

    try:
        # 1. Init
        print("Testing /init...")
        init_data = {
            "hoop_left": [100, 100],
            "hoop_right": [200, 100]
        }
        resp = requests.post(f"{BASE_URL}/init", json=init_data)
        if resp.status_code == 200:
            print("PASS: /init")
        else:
            print(f"FAIL: /init {resp.text}")
            sys.exit(1)

        # 2. Process
        print("Testing /process...")
        # Create a dummy image
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.circle(img, (150, 150), 20, (33, 121, 250), -1) # Draw a "ball"
        _, img_encoded = cv2.imencode('.jpg', img)
        
        files = {'image': ('test.jpg', img_encoded.tobytes(), 'image/jpeg')}
        
        resp = requests.post(f"{BASE_URL}/process", files=files)
        if resp.status_code == 200:
            data = resp.json()
            print(f"PASS: /process. Response: {data}")
            if 'fgm' in data and 'fga' in data:
                 print("PASS: Response structure valid")
            else:
                 print("FAIL: Invalid response structure")
        else:
            print(f"FAIL: /process {resp.text}")

        # 3. Reset
        print("Testing /reset...")
        resp = requests.post(f"{BASE_URL}/reset")
        if resp.status_code == 200:
            print("PASS: /reset")
        else:
            print(f"FAIL: /reset {resp.text}")

    finally:
        if server_process:
            print("Stopping server...")
            server_process.terminate()
            server_process.wait()

if __name__ == "__main__":
    test_server()
