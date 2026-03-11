import cv2
import numpy as np
import sys
import json
import os

from cv_core import BasketballTracker

def test_params(p2, blur_k):
    with open('ground_truth.json') as f:
        gt_data = json.load(f)

    video_path = next((k for k in gt_data.keys() if 'flipped' in k), None)
    if not video_path or not os.path.exists(video_path):
        return

    gt = gt_data[video_path]
    cap = cv2.VideoCapture(video_path)
    ret, first_frame = cap.read()
    if not ret: return

    scale = 640.0 / first_frame.shape[1]
    scaled_hoop_left = (int(gt['hoop_left'][0] * scale), int(gt['hoop_left'][1] * scale))
    scaled_hoop_right = (int(gt['hoop_right'][0] * scale), int(gt['hoop_right'][1] * scale))
    
    # We will monkey patch the tracker HoughCircles call
    original_find_ball = BasketballTracker.find_ball
    
    def patched_find_ball(self, frame):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        chosen = None
        
        if self.prev_circle is not None:
            prev_x, prev_y, _ = self.prev_circle
            margin = int(self.max_radius * 5)
            h, w = gray_frame.shape
            
            x1 = max(0, int(prev_x - margin))
            y1 = max(0, int(prev_y - margin))
            x2 = min(w, int(prev_x + margin))
            y2 = min(h, int(prev_y + margin))
            
            if x2 > x1 and y2 > y1:
                roi = gray_frame[y1:y2, x1:x2]
                blurred_roi = cv2.GaussianBlur(roi, (blur_k, blur_k), 0)
                
                circles = cv2.HoughCircles(
                    blurred_roi, cv2.HOUGH_GRADIENT, 1.2, max(20, int(self.ball_radius_est * 3)),
                    param1=100, param2=p2, 
                    minRadius=self.min_radius, maxRadius=self.max_radius
                )
                
                if circles is not None:
                    circles = np.uint16(np.around(circles))
                    best_dist = float('inf')
                    for i in circles[0, :]:
                        gx = int(i[0] + x1)
                        gy = int(i[1] + y1)
                        gr = i[2]
                        
                        curr_dist = ((gx-prev_x)**2 + (gy-prev_y)**2)
                        if curr_dist < best_dist:
                            best_dist = curr_dist
                            chosen = np.array([gx, gy, gr])

        if chosen is None:
            blurred_frame = cv2.GaussianBlur(gray_frame, (blur_k, blur_k), 0)
            circles = cv2.HoughCircles(
                blurred_frame, cv2.HOUGH_GRADIENT, 1.2, max(20, int(self.ball_radius_est * 3)),
                param1=100, param2=p2, 
                minRadius=self.min_radius, maxRadius=self.max_radius
            )
            
            if circles is not None:
                circles = np.uint16(np.around(circles))
                if self.prev_circle is not None:
                    prev_x, prev_y, _ = self.prev_circle
                    best_dist = float('inf')
                    for i in circles[0, :]:
                        curr_dist = ((i[0]-prev_x)**2 + (i[1]-prev_y)**2)
                        if curr_dist <= best_dist:
                            best_dist = curr_dist
                            chosen = i
                else:
                    chosen = circles[0, 0]
        return chosen
        
    BasketballTracker.find_ball = patched_find_ball
    
    tracker = BasketballTracker(scaled_hoop_left, scaled_hoop_right)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    base_skip = int(1/0.15)
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
    BasketballTracker.find_ball = original_find_ball
    # actual sequence vs pred seq
    print(f"p2={p2}, blur_k={blur_k} | PREDICTED: {tracker.shots} ACTUAL: {gt['sequence']}")

for p2 in [18, 20, 22, 24, 26, 28, 30]:
    for bk in [5, 7, 9]:
        test_params(p2, bk)
