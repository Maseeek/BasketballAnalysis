# 🏀 Basketball Shot Analysis System

<div align="center">

**An intelligent computer vision application that analyzes basketball shooting performance using real-time video processing**

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.0+-green.svg)](https://opencv.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Latest-orange.svg)](https://numpy.org/)

</div>

---

## 📋 Overview

This application leverages advanced computer vision techniques to automatically track basketball shots, calculate shooting statistics, and provide real-time performance analytics. The system processes video footage to detect the basketball, track its trajectory, predict its path using polynomial regression, and determine whether shots are successful—all while calculating critical metrics like release angle and field goal percentage.

**Key Capabilities:**
- ✨ Real-time basketball detection and tracking
- 📊 Automatic shot counting and success rate calculation
- 📐 Release angle measurement and optimization analysis
- 🎯 Trajectory prediction using polynomial regression
- 📈 Performance statistics and visualization

---

## 🎥 Demo

The application processes video input and provides real-time visual feedback:
- **Ball Detection**: Identifies and tracks the basketball throughout the video
- **Trajectory Visualization**: Displays actual and predicted ball paths
- **Live Statistics**: Shows FG%, FGM, FGA, and release angles in real-time
- **Shot Analysis**: Automatically detects makes and misses

---

## ✨ Features

### 🔍 Advanced Ball Detection
The system employs a multi-stage detection pipeline:
- **Color Space Transformation**: Converts frames from BGR to grayscale to focus on edges rather than colors
- **Gaussian Blur**: Reduces noise and eliminates false positives from camera grain
- **Hough Circle Transform**: Detects circular objects using gradient information
- **Smart Tracking**: Uses distance-based filtering to maintain consistent ball tracking across frames

### 📉 Trajectory Analysis
- **Real-time Tracking**: Records ball position at each frame
- **Polynomial Regression**: Applies 2nd-degree polynomial fitting to predict ball trajectory
- **Path Visualization**: Displays both actual and predicted paths with visual overlays

### 📊 Performance Metrics
- **Field Goal Percentage (FG%)**: Automatically calculates shooting accuracy
- **Release Angle**: Measures shot angle at release point
- **Streak Tracking**: Identifies longest consecutive make streaks
- **Comparative Analysis**: Separates statistics for made vs. missed shots
- **Optimal Angle Detection**: Analyzes correlation between release angle and success rate

### 📈 Data Visualization
- Statistical charts showing shot distribution
- Angle analysis plots
- Field goal percentage trends
- Performance summaries

---

## 🛠️ Technology Stack

- **Python 3.7+**: Core programming language
- **OpenCV (cv2)**: Computer vision and image processing
- **NumPy**: Numerical computations and array operations
- **Tkinter**: GUI for file selection
- **Matplotlib**: Data visualization and statistical plotting

---

## 📦 Installation

### Prerequisites
Ensure you have Python 3.7 or higher installed on your system.

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/Maseeek/BasketballAnalysis.git
   cd BasketballAnalysis
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python main.py
   ```

---

## 🚀 Usage

1. **Launch the application**
   ```bash
   python main.py
   ```

2. **Select your video file**
   - A file dialog will appear
   - Navigate to and select your basketball shooting video
   - Supported formats: `.mp4`, `.avi`, `.mov`

3. **Calibrate the system**
   - Click on the **left edge** of the basketball hoop when prompted
   - Click on the **right edge** of the basketball hoop when prompted
   - This calibration helps the system determine shot success and ball size

4. **Watch the analysis**
   - The video will play with real-time overlays showing:
     - Detected basketball (orange circle)
     - Ball trajectory (orange trail)
     - Predicted path (gray dots)
     - Live statistics (FG%, FGM, FGA)
     - Release angle for each shot

5. **Exit**
   - Press `q` to stop the analysis at any time

---

## 🧠 Technical Implementation

### Ball Detection Algorithm
The `findBall()` function implements a robust multi-step detection process:

1. **Grayscale Conversion**: Simplifies color information to focus on structural features
2. **Gaussian Blur (17×17 kernel)**: Smooths image to reduce noise
3. **Hough Circle Detection**: Identifies circular objects with configurable radius constraints
4. **Proximity Tracking**: Selects the circle closest to the previous frame's position, ensuring consistent tracking

```python
def findBall(frame, prevCircle, radius):
    # Converts to grayscale and applies Gaussian blur
    # Uses Hough Circle Transform with dynamic radius bounds
    # Returns the most likely basketball position
```

### Trajectory Prediction
The system uses **polynomial regression** to model the parabolic path of the basketball:

- **Data Collection**: Records (x, y) coordinates of the ball at each frame
- **Polynomial Fitting**: Applies `numpy.polyfit()` with degree 2 to compute trajectory coefficients
- **Path Rendering**: Visualizes the predicted path in real-time

```python
A, B, C = np.polyfit(posListX, posListY, 2)
# Models trajectory as: y = Ax² + Bx + C
```

### Shot Detection Logic
The application determines shot outcomes using intelligent heuristics:

- **Shot In Progress**: Triggered when ball rises above hoop level
- **Make Detection**: Verified when ball descends through the hoop's horizontal bounds
- **Release Angle Calculation**: Uses `atan2()` on initial trajectory vectors
- **Cooldown System**: Prevents duplicate detection of the same shot

### Performance Statistics
Comprehensive metrics are calculated and displayed:

- **FG% (Field Goal Percentage)**: `(FGM / FGA) × 100`
- **Average Release Angles**: Computed separately for makes and misses
- **Streak Analysis**: Tracks longest consecutive make sequences
- **Angle Optimization**: Identifies correlation between release angle and success rate

---

## 📊 Output Metrics

The system provides comprehensive shooting analytics:

| Metric | Description |
|--------|-------------|
| **FGM** | Field Goals Made - Total successful shots |
| **FGA** | Field Goal Attempts - Total shots attempted |
| **FG%** | Field Goal Percentage - Success rate |
| **Release Angle** | Shot angle at release (degrees) |
| **Average Angle** | Mean release angle across all shots |
| **Make Angle** | Average angle for successful shots |
| **Miss Angle** | Average angle for missed shots |
| **Longest Streak** | Most consecutive successful shots |

---

## 📁 Project Structure

```
BasketballAnalysis/
│
├── main.py              # Main application entry point
├── README.md            # Project documentation
├── requirements.txt     # Python dependencies
│
├── Resources/
│   ├── Videos/         # Sample basketball videos
│   └── Images/         # Sample images and screenshots
│
├── dist/               # Compiled executable
└── build/              # Build artifacts
```

---

## 🔮 Future Enhancements

- [ ] **Live Webcam Support**: Real-time analysis from camera feed
- [ ] **Machine Learning Integration**: Improved ball detection using neural networks
- [ ] **Multi-player Tracking**: Analyze multiple shooters simultaneously
- [ ] **Shot Heat Maps**: Visualize shooting locations and success rates
- [ ] **Advanced Statistics**: eFG%, True Shooting %, shot charts
- [ ] **Mobile App**: Cross-platform mobile application
- [ ] **Cloud Storage**: Save and compare historical performance data
- [ ] **3D Trajectory Analysis**: Enhanced spatial tracking

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page or submit a pull request.

---

## 📧 Contact

**Project Maintainer**: [Maseeek](https://github.com/Maseeek)

**Project Link**: [https://github.com/Maseeek/BasketballAnalysis](https://github.com/Maseeek/BasketballAnalysis)

---

## 📝 License

This project is available for educational and personal use.

---

<div align="center">

**Built with ❤️ using Python and OpenCV**

*Transforming basketball training through intelligent video analysis*

</div>
