import requests
import os
import time
import sys
import subprocess
import signal

BASE_URL = "http://127.0.0.1:5001"
SERVER_SCRIPT = "server.py"

# Test data for NEAVid2.mp4
VIDEO_PATH = "NEAVid2.mp4"
HOOP_LEFT = [1541, 375]
HOOP_RIGHT = [1646, 348]

def is_server_running():
    try:
        r = requests.get(f"{BASE_URL}/health")
        return r.status_code == 200
    except Exception as e:
        print(f"Connection failed: {e}") 
        return False

def start_server():
    print(f"Starting {SERVER_SCRIPT} on port 5000...")
    process = subprocess.Popen([sys.executable, SERVER_SCRIPT], cwd=os.getcwd())
    
    for i in range(40): # Wait up to 20 seconds
        if is_server_running():
            print("Server started successfully.")
            return process
        time.sleep(0.5)
        
    print("Failed to start server.")
    process.terminate()
    return None

def test_upload_and_analyze():
    server_process = None
    if not is_server_running():
        server_process = start_server()
        if not server_process:
            sys.exit(1)
    else:
        print("Server already running, using existing instance.")

    try:
        print(f"Testing /upload-and-analyze with {VIDEO_PATH}...")
        
        if not os.path.exists(VIDEO_PATH):
            print(f"FAIL: Video file {VIDEO_PATH} not found in current directory.")
            sys.exit(1)

        with open(VIDEO_PATH, 'rb') as f:
            files = {'video': (VIDEO_PATH, f, 'video/mp4')}
            data = {
                'hoopLeft': str(HOOP_LEFT),
                'hoopRight': str(HOOP_RIGHT),
                'showAngle': 'true'
            }
            
            start_time = time.time()
            resp = requests.post(f"{BASE_URL}/upload-and-analyze", files=files, data=data) 
            end_time = time.time()
            
            print(f"Request took {end_time - start_time:.2f} seconds")

        if resp.status_code == 200:
            result = resp.json()
            if result.get('success'):
                print("PASS: /upload-and-analyze")
                data = result['data']
                print(f"Results: Makes={data['makes']}, Misses={data['misses']}")
                print(f"Full Data: {data}")
            else:
                print(f"FAIL: /upload-and-analyze returned success=False. Error: {result.get('error')}")
        else:
            print(f"FAIL: /upload-and-analyze status {resp.status_code}")
            print(f"Response: {resp.text}")

    finally:
        if server_process:
            print("Stopping server...")
            server_process.terminate()
            server_process.wait()

if __name__ == "__main__":
    test_upload_and_analyze()
