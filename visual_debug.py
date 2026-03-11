import cv2
import tkinter as tk
from tkinter import filedialog
import numpy as np
from cv_core import BasketballTracker

# --- 1. CONFIGURATION & UI STYLING ---

COLORS = {
    'bg_dark': (20, 20, 25),       # Nearly black background for panels
    'panel_bg': (30, 32, 40),      # Dark sidebar background
    'text_main': (240, 240, 240),  # Bright white/grey for main text
    'text_dim': (150, 160, 170),   # Dimmed text for labels
    
    # Accents - "Cyber/Neon" vibe
    'accent_cyan': (255, 220, 100), 
    'accent_gold': (0, 215, 255),  # Gold/Orange
    'success': (100, 255, 100),    # Bright Green
    'warning': (50, 150, 255),     # Orange
    'danger': (80, 80, 255),       # Red
}

FONT_MAIN = cv2.FONT_HERSHEY_DUPLEX
FONT_SMALL = cv2.FONT_HERSHEY_SIMPLEX

def draw_glass_panel(img, x, y, w, h, color, alpha=0.6, border=True):
    """Draws a modern semi-transparent panel with optional border."""
    if y+h > img.shape[0] or x+w > img.shape[1]: 
        return
        
    sub_img = img[y:y+h, x:x+w]
    rect = np.full(sub_img.shape, color, dtype=np.uint8)
    
    res = cv2.addWeighted(sub_img, 1 - alpha, rect, alpha, 0)
    img[y:y+h, x:x+w] = res
    
    if border:
        cv2.rectangle(img, (x, y), (x+w, y+h), (60, 60, 70), 1)

def draw_text_centered(img, text, x, y, font, scale, color, thickness=1):
    text_size = cv2.getTextSize(text, font, scale, thickness)[0]
    text_x = int(x - text_size[0] // 2)
    text_y = int(y + text_size[1] // 2)
    cv2.putText(img, text, (text_x, text_y), font, scale, color, thickness, cv2.LINE_AA)

def draw_labeled_stat(img, label, value, x, y, width, is_good=None):
    cv2.putText(img, label, (x + 10, y), FONT_SMALL, 0.5, COLORS['text_dim'], 1, cv2.LINE_AA)
    
    val_color = COLORS['text_main']
    if is_good is True: val_color = COLORS['success']
    if is_good is False: val_color = COLORS['danger']
    
    val_str = str(value)
    text_size = cv2.getTextSize(val_str, FONT_MAIN, 0.6, 1)[0]
    cv2.putText(img, val_str, (x + width - 10 - text_size[0], y), FONT_MAIN, 0.6, val_color, 1, cv2.LINE_AA)

# --- 2. LOGIC ---

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
    # Bring window to front
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

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

    hoop_left = get_click_coordinates(frame, "Left Side of Hoop")
    hoop_right = get_click_coordinates(frame, "Right Side of Hoop")
    
    # Handle window closing instead of clicking
    if hoop_left == (-1, -1) or hoop_right == (-1, -1):
        print("Setup cancelled.")
        return

    tracker = BasketballTracker(hoop_left, hoop_right)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    SIDEBAR_WIDTH = 320

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        stats = tracker.process_frame(frame, debug=True)
        debug_frame = stats['frame']
        
        # --- UI RENDERING ---
        h, w, c = debug_frame.shape
        canvas = np.zeros((h, w + SIDEBAR_WIDTH, c), dtype=np.uint8)
        canvas[:] = COLORS['bg_dark']
        
        canvas[0:h, 0:w] = debug_frame
        
        sb_x = w
        center_x = sb_x + SIDEBAR_WIDTH // 2
        
        # 1. Header
        cv2.rectangle(canvas, (sb_x, 0), (sb_x + SIDEBAR_WIDTH, 80), COLORS['panel_bg'], -1)
        cv2.line(canvas, (sb_x, 80), (sb_x+SIDEBAR_WIDTH, 80), (50,50,60), 1)
        draw_text_centered(canvas, "BASKETBALL", center_x, 30, FONT_MAIN, 0.8, COLORS['accent_gold'], 2)
        draw_text_centered(canvas, "ANALYSIS", center_x, 60, FONT_MAIN, 0.8, COLORS['text_main'], 1)

        # 2. Key Stats
        draw_glass_panel(canvas, sb_x + 15, 100, SIDEBAR_WIDTH - 30, 140, COLORS['panel_bg'])
        
        fgm = stats['fgm']
        fga = stats['fga']
        fg_percent = stats['fg_percent']
        
        pct_color = COLORS['text_main']
        if fga > 0:
            if fg_percent >= 50: pct_color = COLORS['success']
            elif fg_percent >= 30: pct_color = COLORS['warning']
            else: pct_color = COLORS['danger']
            
        draw_text_centered(canvas, f"{fg_percent:.1f}%", center_x, 150, FONT_MAIN, 2.0, pct_color, 3)
        draw_text_centered(canvas, "FIELD GOAL %", center_x, 190, FONT_SMALL, 0.5, COLORS['text_dim'])
        
        # 3. Shot Counter Row
        draw_glass_panel(canvas, sb_x + 15, 260, SIDEBAR_WIDTH - 30, 60, COLORS['panel_bg'])
        draw_labeled_stat(canvas, "MADE", fgm, sb_x + 30, 300, 110, True)
        draw_labeled_stat(canvas, "ATTEMPTED", fga, sb_x + 160, 300, 130)

        # 4. Live Shot Status
        draw_glass_panel(canvas, sb_x + 15, 340, SIDEBAR_WIDTH - 30, 120, COLORS['panel_bg'])
        
        status_txt = "WAITING"
        status_col = COLORS['text_dim']
        
        if stats.get('cooldown', 0) > 0:
            status_txt = "SHOT LOGGED"
            status_col = COLORS['success']
        elif stats.get('shot_in_progress', False):
            status_txt = "TRACKING"
            status_col = COLORS['warning']
            
        draw_text_centered(canvas, status_txt, center_x, 380, FONT_MAIN, 0.9, status_col, 2)
        
        if stats.get('shot_in_progress', False) and stats.get('current_release_angle', 0) > 0:
            draw_text_centered(canvas, f"Angle: {stats['current_release_angle']:.1f}", center_x, 420, FONT_SMALL, 0.7, COLORS['text_main'])

        # 5. Recent History
        hist_y_start = 480
        draw_text_centered(canvas, "RECENT SHOTS", center_x, hist_y_start, FONT_SMALL, 0.6, COLORS['text_dim'])
        
        recent_shots = stats.get('shots', [])[-5:]
        recent_shots.reverse()
        
        for i, res in enumerate(recent_shots):
            y_pos = hist_y_start + 40 + (i * 35)
            if y_pos > h - 40: break
            
            c_color = COLORS['success'] if res == 1 else COLORS['danger']
            label = "MAKE" if res == 1 else "MISS"
            
            cv2.circle(canvas, (sb_x + 50, y_pos - 5), 6, c_color, -1)
            cv2.putText(canvas, label, (sb_x + 70, y_pos), FONT_SMALL, 0.7, COLORS['text_main'], 1)

        cv2.imshow('Basketball Analysis HUD', canvas)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
    print(f"Final Stats - FGM: {stats['fgm']}, FGA: {stats['fga']}, FG%: {stats['fg_percent']:.2f}%")

if __name__ == "__main__":
    main()
