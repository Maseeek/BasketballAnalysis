from flask import Flask, request, jsonify
import os
import cv2
import numpy as np
import math
import json
from flask_cors import CORS
from werkzeug.utils import secure_filename
from cv_core import BasketballTracker

# RUNNING ON PORT 5000 (User Request)

# PYTHON PROGRAM TO PROCESS VIDEO AND DETERMINE BASKETBALL OUTCOMES
MAX_FRAMES = 5000
app = Flask(__name__)

# Configure CORS
allowed_origins = [
    "http://localhost:5173",
    "http://localhost:3000",
    "https://nothingbutnet.online",
    "https://www.nothingbutnet.online",
    os.environ.get("FRONTEND_URL"),
    os.environ.get("PRODUCTION_FRONTEND_URL")
]
allowed_origins = [origin for origin in allowed_origins if origin]

CORS(app, resources={r"/*": {"origins": allowed_origins}}, supports_credentials=True)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Helper functions that were not in cv_core but used by server
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

def calculateAverageAngle(shotAngles, shots):
    shotsMadeAngle = []
    shotsMissedAngle = []

    for i in range(len(shots)):
        if shotAngles[i] != 0:
            if not abs(sum(shotAngles) / len(shotAngles) - shotAngles[i]) > 2 * sum(shotAngles) / len(shotAngles):
                if shots[i] == 1:
                    shotsMadeAngle.append(shotAngles[i])
                else:
                    shotsMissedAngle.append(shotAngles[i])
    try:
        if not shotsMadeAngle and not shotsMissedAngle:
            return 0, 0, 0
            
        total_angles = shotsMadeAngle + shotsMissedAngle
        averageAngle = sum(total_angles) / len(total_angles) if total_angles else 0
        averageMakeAngle = sum(shotsMadeAngle) / len(shotsMadeAngle) if len(shotsMadeAngle) > 0 else 0
        averageMissAngle = sum(shotsMissedAngle) / len(shotsMissedAngle) if len(shotsMissedAngle) > 0 else 0
        return averageAngle, averageMakeAngle, averageMissAngle
    except:
        return 0, 0, 0

def analyze_video(videoPath, hoopLeft, hoopRight, max_frames, accuracy=0.15):
    tracker = BasketballTracker(hoopLeft, hoopRight)
    cap = cv2.VideoCapture(videoPath)

    frame_count = 0

    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        # Skip frames based on accuracy optimization
        # accuracy = 0.15 means 1/0.15 ~= 6.66 => skip ~6 frames? 
        # User code: if frame_count % int(1/accuracy) != 0: continue
        # If accuracy is 0.5 (showAngle=True), 1/0.5 = 2. Process every 2nd frame.
        if frame_count % int(1/accuracy) != 0: 
            continue

        tracker.process_frame(frame, debug=False)

    cap.release()

    shots = tracker.shots
    shotAngles = tracker.shot_angles

    makes = shots.count(1) if shots else 0
    misses = shots.count(0) if shots else 0
    fg_percentage = 100 * makes / len(shots) if shots else 0
    longest_streak = getLongestStreak(shots)
    averageAngle, averageMakeAngle, averageMissAngle = calculateAverageAngle(shotAngles, shots) if shotAngles and shots else (0, 0, 0)

    result = {
        "total_shots": len(shots),
        "makes": makes,
        "misses": misses,
        "fg_percentage": round(fg_percentage, 2),
        "longest_streak": longest_streak,
        "average_angle": round(averageAngle, 2),
        "average_make_angle": round(averageMakeAngle, 2),
        "average_miss_angle": round(averageMissAngle, 2),
        "shot_angles": shotAngles,
        "shots_results": shots
    }

    return result

@app.route('/upload-and-analyze', methods=['POST'])
def upload_and_analyze():
    if 'video' not in request.files:
        return jsonify({'success': False, 'error': 'No video file provided'}), 400

    video_file = request.files['video']

    if video_file.filename == '':
        return jsonify({'success': False, 'error': 'Empty filename'}), 400

    # Get hoop coordinates from request
    try:
        hoopLeftRaw = request.form.get('hoopLeft', '[0, 0]')
        hoopRightRaw = request.form.get('hoopRight', '[100, 0]')
        
        hoopLeft = json.loads(hoopLeftRaw)
        hoopRight = json.loads(hoopRightRaw)
        
        # Ensure they are lists/tuples before passing to tracker
        if isinstance(hoopLeft, dict):
            if 'x' in hoopLeft and 'y' in hoopLeft:
                hoopLeft = [hoopLeft['x'], hoopLeft['y']]
            elif '0' in hoopLeft and '1' in hoopLeft:
                 hoopLeft = [hoopLeft['0'], hoopLeft['1']]

        if isinstance(hoopRight, dict):
             if 'x' in hoopRight and 'y' in hoopRight:
                hoopRight = [hoopRight['x'], hoopRight['y']]
             elif '0' in hoopRight and '1' in hoopRight:
                 hoopRight = [hoopRight['0'], hoopRight['1']]

    except Exception as e:
        return jsonify({'success': False, 'error': 'Invalid hoop coordinates'}), 400

    # Get showAngle status from request
    show_angle = request.form.get('showAngle', 'false').lower() == 'true'
    accuracy = 0.5 if show_angle else 0.15

    # Save the video file
    filename = secure_filename(video_file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    video_file.save(filepath)

    try:
        # Run the analysis
        result = analyze_video(filepath, hoopLeft, hoopRight, MAX_FRAMES, accuracy)
        
        # Clean up the file 
        if os.path.exists(filepath):
             os.remove(filepath)

        return jsonify({
            'success': True,
            'data': result
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"}), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
