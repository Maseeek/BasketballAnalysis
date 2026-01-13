import cv2
import numpy as np
import math

# Constants
PUMPKIN = (33, 121, 250)
CELADON = (187, 229, 169)
VANILLA = (177, 246, 252)
FELDGRAU = (59, 75, 63)
GREEN = (63, 99, 68)

dist = lambda x1, y1, x2, y2: (x1-x2)**2 + (y1-y2)**2

class BasketballTracker:
    def __init__(self, hoop_left, hoop_right):
        self.hoop_left = hoop_left
        self.hoop_right = hoop_right
        
        # Calculate hoop metrics
        self.ball_radius = 0.264 * math.sqrt(dist(hoop_left[0], hoop_left[1], hoop_right[0], hoop_right[1]))
        self.hoop_max_height = min(hoop_left[1], hoop_right[1])
        # self.hoop_average_height = (hoop_left[1] + hoop_right[1]) / 2 # Unused in original
        self.hoop_min_height = max(hoop_left[1], hoop_right[1])
        
        # State variables
        self.shots = [] # 1 for make, 0 for miss
        self.shot_angles = []
        self.pos_list_x = []
        self.pos_list_y = []
        self.fga = 0
        self.fgm = 0
        self.cooldown = 0
        self.prev_circle = None
        self.center = None
        self.shot_in_progress = False
        self.radius = 0

    def find_ball(self, frame):
        """Locates the ball in the current frame using HoughCircles."""
        CONSTANT = 1.2
        min_radius = int(self.ball_radius / CONSTANT)
        max_radius = int(self.ball_radius * CONSTANT)
        chosen = None
        
        # Preprocessing
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred_frame = cv2.GaussianBlur(gray_frame, (17, 17), 0)
        
        # Detection
        circles = cv2.HoughCircles(blurred_frame, cv2.HOUGH_GRADIENT, 1.2, 100,
                                   param1=100, param2=30, minRadius=min_radius, maxRadius=max_radius)
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                if chosen is None: chosen = i
                if self.prev_circle is not None:
                    # Choose circle closest to previous position
                    if dist(chosen[0], chosen[1], self.prev_circle[0], self.prev_circle[1]) <= \
                       dist(i[0], i[1], self.prev_circle[0], self.prev_circle[1]):
                        chosen = i
        return chosen

    def process_frame(self, frame, debug=False):
        """
        Analyzes a single frame, updates state, and returns modified frame (if debug=True).
        Returns a dictionary of current stats.
        """
        basketball = self.find_ball(frame)
        
        if basketball is not None:
            if debug:
                self.show_frame_with_ball_circled(frame, basketball)
            self.prev_circle = (basketball[0], basketball[1], basketball[2])
            self.center = (int(basketball[0]), int(basketball[1]))
            self.radius = basketball[2]
        
        if debug:
            self.draw_hoop(frame)

        # Shot Logic
        if self.center is not None:
            # If ball is near hoop height
            if self.center[1] <= (self.hoop_min_height + self.radius * 5) and self.cooldown == 0:
                self.pos_list_x.append(self.center[0])
                self.pos_list_y.append(self.center[1])
                
                if self.center[1] < self.hoop_min_height:
                    self.shot_in_progress = True
                    if debug:
                         cv2.putText(frame, "Shot in Progress", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, VANILLA, 2, cv2.LINE_AA)

                    if len(self.pos_list_x) > 1 and len(self.pos_list_y) > 1:
                        angle = self.calculate_angle()
                        if debug:
                            cv2.putText(frame, f"Release Angle: {angle:.2f}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, VANILLA, 2, cv2.LINE_AA)

        # Shot Outcome Logic
        if len(self.pos_list_x) > 3:
            # If ball drops below hoop height and shot was in progress
            if self.pos_list_y[-1] > self.hoop_min_height and self.shot_in_progress:
                average_x_of_last_2 = (self.pos_list_x[-1] + self.pos_list_x[-2]) / 2
                
                # Check if x is within hoop bounds
                if self.hoop_left[0] < average_x_of_last_2 < self.hoop_right[0]:
                    self.shots.append(1)
                    self.fgm += 1
                else:
                    self.shots.append(0)
                self.fga += 1

                self.shot_angles.append(self.calculate_angle())
                self.pos_list_x.clear()
                self.pos_list_y.clear()

                self.shot_in_progress = False
                self.cooldown = 30 # Frames to wait before tracking next shot
            else:
                if debug:
                    self.trace_predicted_path(frame)

        if self.cooldown > 0:
            self.cooldown -= 1
            
        return {
            "fgm": self.fgm,
            "fga": self.fga,
            "fg_percent": (100 * self.fgm / self.fga) if self.fga > 0 else 0.0,
            "frame": frame
        }

    def calculate_angle(self):
        try:
            delta_x = self.pos_list_x[2] - self.pos_list_x[0]
            delta_y = self.pos_list_y[2] - self.pos_list_y[0]
            angle_radians = math.atan2(delta_y, delta_x)
            angle_degrees = -math.degrees(angle_radians)
            
            if 90 < angle_degrees < 180:
                angle_degrees = 180 - angle_degrees
            
            if 0 < angle_degrees < 90:
                return angle_degrees
            else:
                return 0
        except:
            return 0

    def draw_hoop(self, frame):
        cv2.circle(frame, self.hoop_left, 10, GREEN, cv2.FILLED)
        cv2.circle(frame, self.hoop_right, 10, GREEN, cv2.FILLED)
        cv2.line(frame, self.hoop_left, self.hoop_right, GREEN, 2)

    def show_frame_with_ball_circled(self, frame, ball):
        if ball is not None:
            cv2.circle(frame, (ball[0], ball[1]), ball[2], PUMPKIN, 2)
            cv2.putText(frame, f"radius {ball[2]}", (ball[0], ball[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, PUMPKIN, 2, cv2.LINE_AA)

    def trace_predicted_path(self, frame):
        if len(self.pos_list_x) < 3: return
        try:
            A, B, C = np.polyfit(self.pos_list_x, self.pos_list_y, 2)
            width_of_frame = frame.shape[1]
            x_list = [item for item in range(width_of_frame)]
            
            # Draw historical points
            for i, (pos_x, pos_y) in enumerate(zip(self.pos_list_x, self.pos_list_y)):
                pos = (pos_x, pos_y)
                cv2.circle(frame, pos, 10, PUMPKIN, cv2.FILLED)
                if i > 0:
                    cv2.line(frame, pos, (self.pos_list_x[i - 1], self.pos_list_y[i - 1]), PUMPKIN, 5)
            
            # Draw prediction
            for x in x_list:
                y = int(A * x ** 2 + B * x + C)
                # Ensure y is within frame bounds to avoid errors (optional but good practice)
                if 0 <= y < frame.shape[0]:
                    cv2.circle(frame, (x, y), 2, FELDGRAU, cv2.FILLED)
        except:
            pass
