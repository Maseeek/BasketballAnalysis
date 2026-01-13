import cv2
import tkinter as tk
from tkinter import filedialog
from cv_core import BasketballTracker, VANILLA

def get_video_path():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title="Select Video File", filetypes=[("Video Files", "*.mp4;*.avi;*.mov")])
    return file_path

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

    print(f"Please click: {window_name}")
    while not clicked:
        cv2.waitKey(1)
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break
            
    cv2.destroyWindow(window_name)
    print(f"Selected: {click_x}, {click_y}")
    return (click_x, click_y)

def main():
    video_path = get_video_path()
    if not video_path:
        print("No video selected.")
        return

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        print("Failed to read video.")
        return

    # User Setup
    hoop_left = get_click_coordinates(frame, "Left Side of Hoop")
    hoop_right = get_click_coordinates(frame, "Right Side of Hoop")
    
    # Initialize Tracker
    tracker = BasketballTracker(hoop_left, hoop_right)
    
    # Reset video to start
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process Frame
        stats = tracker.process_frame(frame, debug=True)
        debug_frame = stats['frame']

        # UI Overlay
        if stats['fga'] != 0:
             cv2.putText(debug_frame, f"FGM: {stats['fgm']}, FGA: {stats['fga']}, FG%: {stats['fg_percent']:.2f}", 
                         (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, VANILLA, 2, cv2.LINE_AA)

        cv2.imshow('Basketball Tracker Debug', debug_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
    print(f"Final Stats - FGM: {stats['fgm']}, FGA: {stats['fga']}, FG%: {stats['fg_percent']:.2f}%")

if __name__ == "__main__":
    main()
