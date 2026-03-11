import requests
import os
import sys
import time

# Target the actual port your app runs on
BASE_URL = "http://127.0.0.1:5001"
VIDEO_SAMPLE = "Resources/Videos/video.mp4" # Using a known small sample

def test_api_upload():
    if not os.path.exists(VIDEO_SAMPLE):
        print(f"Error: {VIDEO_SAMPLE} not found. Please ensure the path is correct.")
        return

    print(f"Testing API upload with {VIDEO_SAMPLE}...")
    
    # Setup the payload
    # Note: server.py expects 'hoopLeft', 'hoopRight' as JSON strings in form-data
    data = {
        "hoopLeft": "[118, 541]",
        "hoopRight": "[295, 583]",
        "showAngle": "true"
    }

    try:
        with open(VIDEO_SAMPLE, 'rb') as f:
            files = {'video': f}
            start_time = time.time()
            response = requests.post(f"{BASE_URL}/upload-and-analyze", data=data, files=files)
            end_time = time.time()

        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                print("PASS: Video uploaded and analyzed successfully.")
                print(f"Time taken: {end_time - start_time:.2f}s")
                print(f"Result: {result['data']['makes']}/{result['data']['total_shots']} shots made.")
            else:
                print(f"FAIL: Server returned success=False: {result.get('error')}")
        else:
            print(f"FAIL: HTTP Error {response.status_code}: {response.text}")

    except requests.exceptions.ConnectionError:
        print("FAIL: Could not connect to the server. Make sure it is running on port 5001.")
    except Exception as e:
        print(f"FAIL: An unexpected error occurred: {e}")

if __name__ == "__main__":
    test_api_upload()
