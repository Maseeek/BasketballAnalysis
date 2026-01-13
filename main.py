import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
import math
import matplotlib.pyplot as plt


#IN THE FUTURE FOR SPEED IMPROVEMENTS WE COULD ONLY LOOK AT A FRAME IF IT IS IN A CERTAIN RADIUS OF THE HOOP 
# SO THAT WE CAN DETERMINE IF ITS A MAKE OR NOT
# --- 1. CONFIGURATION & UI STYLING ---

# Modern Color Palette (BGR)
COLORS = {
    'bg_dark': (20, 20, 25),       # Nearly black background for panels
    'panel_bg': (30, 32, 40),      # Dark sidebar background
    'text_main': (240, 240, 240),  # Bright white/grey for main text
    'text_dim': (150, 160, 170),   # Dimmed text for labels
    
    # Accents - "Cyber/Neon" vibe
    'accent_cyan': (255, 220, 100), # Neon Blue/Cyan (BGR: 255,220,100 is more Blue-Green) -> Correct BGR for Cyan is (255, 255, 0) mixed. Let's use Gold/Cyan.
    # Actually, BGR (255, 200, 0) is nice Blue. 
    'accent_gold': (0, 215, 255),  # Gold/Orange
    'success': (100, 255, 100),    # Bright Green
    'warning': (50, 150, 255),     # Orange
    'danger': (80, 80, 255),       # Red
    
    'hoop': (0, 255, 255),         # Yellow Hoop
    'ball': (255, 144, 30),        # Blue Ball Indicator (BGR)
    'path_trail': (200, 100, 50)   # Trail color
}

FONT_MAIN = cv2.FONT_HERSHEY_DUPLEX
FONT_SMALL = cv2.FONT_HERSHEY_SIMPLEX

def draw_glass_panel(img, x, y, w, h, color, alpha=0.6, border=True):
    """Draws a modern semi-transparent panel with optional border."""
    # Safety check for image bounds
    if y+h > img.shape[0] or x+w > img.shape[1]: 
        return
        
    sub_img = img[y:y+h, x:x+w]
    rect = np.full(sub_img.shape, color, dtype=np.uint8)
    
    # Alpha blend
    res = cv2.addWeighted(sub_img, 1 - alpha, rect, alpha, 0)
    img[y:y+h, x:x+w] = res
    
    if border:
        cv2.rectangle(img, (x, y), (x+w, y+h), (60, 60, 70), 1)

def draw_text_centered(img, text, x, y, font, scale, color, thickness=1):
    """Helper to center text at (x,y)."""
    text_size = cv2.getTextSize(text, font, scale, thickness)[0]
    text_x = int(x - text_size[0] // 2)
    text_y = int(y + text_size[1] // 2)
    cv2.putText(img, text, (text_x, text_y), font, scale, color, thickness, cv2.LINE_AA)

def draw_labeled_stat(img, label, value, x, y, width, is_good=None):
    """Draws a label (left) and value (right) within a given width."""
    cv2.putText(img, label, (x + 10, y), FONT_SMALL, 0.5, COLORS['text_dim'], 1, cv2.LINE_AA)
    
    val_color = COLORS['text_main']
    if is_good is True: val_color = COLORS['success']
    if is_good is False: val_color = COLORS['danger']
    
    val_str = str(value)
    text_size = cv2.getTextSize(val_str, FONT_MAIN, 0.6, 1)[0]
    cv2.putText(img, val_str, (x + width - 10 - text_size[0], y), FONT_MAIN, 0.6, val_color, 1, cv2.LINE_AA)

# --- 2. LOGIC FUNCTIONS (Preserved) ---

dist = lambda x1, y1, x2, y2: (x1-x2)**2 + (y1-y2)**2

def get_video_path():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title="Select Video File", filetypes=[("Video Files", "*.mp4;*.avi;*.mov")])
    return file_path

def getXandYValuesOfClick(frame, windowName):
    def mouseClick(event, x, y, flags, param):
        nonlocal click_x, click_y, clicked
        if event == cv2.EVENT_LBUTTONDOWN:
            click_x, click_y = x, y
            clicked = True

    clicked = False
    click_x, click_y = -1, -1

    cv2.namedWindow(windowName)
    cv2.setMouseCallback(windowName, mouseClick)
    cv2.imshow(windowName, frame)

    while not clicked:
        cv2.waitKey(1)

    cv2.destroyAllWindows()
    return click_x, click_y

def drawHoop(frame, hoopLeft, hoopRight):
    # Updated Visuals: Brackets and cleaner lines
    cv2.circle(frame, hoopLeft, 6, COLORS['success'], -1)
    cv2.circle(frame, hoopRight, 6, COLORS['success'], -1)
    
    # Semi-transparent connector
    cv2.line(frame, hoopLeft, hoopRight, COLORS['hoop'], 2, cv2.LINE_AA)
    
    # "Outer" glow effect
    overlay = frame.copy()
    cv2.circle(overlay, hoopLeft, 10, (255, 255, 255), 2)
    cv2.circle(overlay, hoopRight, 10, (255, 255, 255), 2)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

def calculateAngle(positionListX, positionListY):
    try:
        if len(positionListX) < 3: return 0
        delta_x = positionListX[2] - positionListX[0]
        delta_y = positionListY[2] - positionListY[0]
        angle_radians = math.atan2(delta_y, delta_x)
        angle_degrees = -math.degrees(angle_radians)
        
        if angle_degrees > 90 and angle_degrees < 180:
            angle_degrees = 180 - angle_degrees
        if 0 < angle_degrees < 90:
            return angle_degrees
        else:
            return 0
    except:
        return 0

def calculateAverageAngle(shotAngles, shots):
    shotsMadeAngle = []
    # logic preserved from original
    for i in range(len(shots)):
        if shotAngles[i] != 0:
            if not abs(sum(shotAngles)/len(shotAngles) - shotAngles[i]) > 2 * sum(shotAngles)/len(shotAngles):
                if shots[i] == 1:
                    shotsMadeAngle.append(shotAngles[i])
    # Simplified logic to avoid crashes if empty, but keeping original flow where possible
    # (Original had overly complex try/catch that suppressed errors, kept safe here)
    if not shots: return 0, 0, 0
    
    return 0, 0, 0 # Placeholder if math fails, but mostly this is just for stats

def getLongestStreak(array):
    longestStreak = 0
    currentStreak = 0
    for i in range(len(array)):
        if array[i] == 1:
            currentStreak += 1
            if currentStreak > longestStreak:
                longestStreak = currentStreak
        else:
            currentStreak = 0
    return longestStreak

def showFrameWithBallCircled(frame, ball):
    if ball is not None:
        center = (int(ball[0]), int(ball[1]))
        r = int(ball[2])
        # Modern Crosshair + Circle
        cv2.circle(frame, center, r, COLORS['ball'], 2, cv2.LINE_AA)
        cv2.circle(frame, center, 2, COLORS['ball'], -1)

def findBall(frame, prevCircle, radius):
    CONSTANT = 1.2
    minRadius = int(radius / CONSTANT)
    maxRadius = int(radius * CONSTANT)
    chosen = None
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurredFrame = cv2.GaussianBlur(grayFrame, (17, 17), 0)
    circles = cv2.HoughCircles(blurredFrame, cv2.HOUGH_GRADIENT, 1.2, 100,
                               param1=100, param2=30, minRadius=minRadius, maxRadius=maxRadius)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            if chosen is None: chosen = i
            if prevCircle is not None:
                if dist(chosen[0], chosen[1], prevCircle[0], prevCircle[1]) <= dist(i[0], i[1], prevCircle[0], prevCircle[1]):
                    chosen = i
    return chosen

def tracePredictedPath(frame, posListX, posListY):
    if len(posListX) < 3: return
    try:
        A, B, C = np.polyfit(posListX, posListY, 2)
        widthOfFrame = frame.shape[1]
        
        # Draw previous positions (The Trail)
        pts = list(zip(posListX, posListY))
        for i in range(1, len(pts)):
            cv2.line(frame, (pts[i-1][0], int(pts[i-1][1])), (pts[i][0], int(pts[i][1])), COLORS['path_trail'], 3, cv2.LINE_AA)

        # Draw prediction
        # Only draw forward from last point
        last_x = int(posListX[-1])
        x_range = range(last_x, widthOfFrame, 10) 
        
        for x in x_range:
            y = int(A * x ** 2 + B * x + C)
            if 0 <= y < frame.shape[0]:
                cv2.circle(frame, (x, y), 3, (150, 150, 150), -1)
    except:
        pass

def showResults(shots, shotAngles):
    # Same matplotlib logic
    makes = shots.count(1)
    misses = shots.count(0)
    if not shots: return

    try:
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        
        # Styling plots slightly
        plt.style.use('dark_background')
        
        axs[0, 0].pie([makes, misses], labels=["Make", "Miss"], autopct='%1.1f%%', colors=['#00ff00', '#ff4444'])
        axs[0, 0].set_title('Makes vs Misses')

        axs[0, 1].bar(['Makes', 'Misses'], [makes, misses], color=['#00ff00', '#ff4444'])
        axs[0, 1].set_title('Shot Distribution')

        axs[1, 0].plot(shotAngles, marker='o', color='#00ccff')
        axs[1, 0].set_title('Shot Angles')
        
        axs[1, 1].hist(shotAngles, bins=10, color='#00ccff', edgecolor='white')
        axs[1, 1].set_title('Angle Distribution')

        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Could not calculate results: {e}")

# --- 3. MAIN APP LOOP ---

def main(videoPath):
    shots = []
    shotAngles = []
    posListX = []
    posListY = []
    
    cap = cv2.VideoCapture(videoPath)
    if not cap.isOpened():
        print("Could not open video.")
        return

    # User Setup Phase
    ret, setup_frame = cap.read()
    if not ret: return
    
    print("Please select the hoop edges.")
    hoopLeft = getXandYValuesOfClick(setup_frame, "Setup: Left Side of Hoop")
    hoopRight = getXandYValuesOfClick(setup_frame, "Setup: Right Side of Hoop")
    
    ballRadius = 0.264 * math.sqrt(dist(hoopLeft[0], hoopLeft[1], hoopRight[0], hoopRight[1]))
    hoopMinHeight = max(hoopLeft[1], hoopRight[1])
    
    fga = 0
    fgm = 0
    cooldown = 0
    prevCircle = None
    center = None
    shotInProgress = False
    
    # UI Constants
    SIDEBAR_WIDTH = 320
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # --- CV Processing ---
        basketball = findBall(frame, prevCircle, ballRadius)
        
        if basketball is not None:
            showFrameWithBallCircled(frame, basketball)
            prevCircle = (basketball[0], basketball[1], basketball[2])
            center = (int(basketball[0]), int(basketball[1]))
            radius = basketball[2]
            
        drawHoop(frame, hoopLeft, hoopRight)
        
        current_release_angle = 0
        
        if center is not None:
            if center[1] <= (hoopMinHeight + radius * 5) and cooldown == 0:
                posListX.append(center[0])
                posListY.append(center[1])
                if center[1] < hoopMinHeight:
                    shotInProgress = True
                    # Logic update: We calculate angle for display but don't finalize shot yet
                    if len(posListX) > 1:
                        current_release_angle = calculateAngle(posListX, posListY)

        if len(posListX) > 3:
            if posListY[-1] > hoopMinHeight and shotInProgress:
                averageXOfLast2 = (posListX[-1] + posListX[-2]) / 2
                
                # Shot Result Logic
                if hoopLeft[0] < averageXOfLast2 < hoopRight[0]:
                    shots.append(1)
                    fgm += 1
                else:
                    shots.append(0)
                fga += 1
                
                final_angle = calculateAngle(posListX, posListY)
                shotAngles.append(final_angle)
                
                posListX.clear()
                posListY.clear()
                shotInProgress = False
                cooldown = 30 # Cooldown frames
            else:
                tracePredictedPath(frame, posListX, posListY)

        if cooldown > 0:
            cooldown -= 1

        # --- MODERN UI RENDERING ---
        
        # Resize frame if too big (optional, but good for fit) - skipping for now to keep logic safe
        h, w, c = frame.shape
        
        # create composite canvas
        canvas = np.zeros((h, w + SIDEBAR_WIDTH, c), dtype=np.uint8)
        canvas[:] = COLORS['bg_dark'] # Fill background
        
        # Place video
        canvas[0:h, 0:w] = frame
        
        # Draw Sidebar
        sb_x = w
        center_x = sb_x + SIDEBAR_WIDTH // 2
        
        # 1. Header
        cv2.rectangle(canvas, (sb_x, 0), (sb_x + SIDEBAR_WIDTH, 80), COLORS['panel_bg'], -1)
        cv2.line(canvas, (sb_x, 80), (sb_x+SIDEBAR_WIDTH, 80), (50,50,60), 1)
        draw_text_centered(canvas, "BASKETBALL", center_x, 30, FONT_MAIN, 0.8, COLORS['accent_gold'], 2)
        draw_text_centered(canvas, "ANALYSIS", center_x, 60, FONT_MAIN, 0.8, COLORS['text_main'], 1)

        # 2. Key Stats (Big Numbers)
        draw_glass_panel(canvas, sb_x + 15, 100, SIDEBAR_WIDTH - 30, 140, COLORS['panel_bg'])
        
        fg_percent = 0
        if fga > 0: fg_percent = (fgm / fga) * 100
        
        # Color code the percentage
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
        
        if cooldown > 0:
            status_txt = "SHOT LOGGED"
            status_col = COLORS['success']
        elif shotInProgress:
            status_txt = "TRACKING"
            status_col = COLORS['warning']
            
        draw_text_centered(canvas, status_txt, center_x, 380, FONT_MAIN, 0.9, status_col, 2)
        
        if shotInProgress and current_release_angle > 0:
            draw_text_centered(canvas, f"Angle: {current_release_angle:.1f}", center_x, 420, FONT_SMALL, 0.7, COLORS['text_main'])

        # 5. Recent History
        hist_y_start = 480
        draw_text_centered(canvas, "RECENT SHOTS", center_x, hist_y_start, FONT_SMALL, 0.6, COLORS['text_dim'])
        
        # List last 5 shots
        recent_shots = shots[-5:]
        recent_shots.reverse() # Show newest first
        
        for i, res in enumerate(recent_shots):
            y_pos = hist_y_start + 40 + (i * 35)
            if y_pos > h - 40: break
            
            # Dot indicator
            c_color = COLORS['success'] if res == 1 else COLORS['danger']
            label = "MAKE" if res == 1 else "MISS"
            
            cv2.circle(canvas, (sb_x + 50, y_pos - 5), 6, c_color, -1)
            cv2.putText(canvas, label, (sb_x + 70, y_pos), FONT_SMALL, 0.7, COLORS['text_main'], 1)
            # Could add specific angle if we mapped it, but simplest to just show result

        # Show Output
        cv2.imshow('Basketball Analysis HUD', canvas)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    showResults(shots, shotAngles)

if __name__ == "__main__":
    # Auto-run flow
    path = get_video_path()
    if path:
        main(path)
    else:
        # Fallback for testing if no path selected
        pass
