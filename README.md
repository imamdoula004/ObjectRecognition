# Real-Time Object Detection, Pose & Motion Tracker

This project uses YOLOv8, MediaPipe, and OpenCV to build a real-time object detection system enhanced with pose estimation, motion detection, and maneuver tracking. It uses your webcam, visualizes objects with bounding boxes (green for general objects, yellow for humans), and logs detections to an Excel file. Simple hand raise gestures are also recognized.

# Features
•	• Object Detection using YOLOv8
•	• Pose Estimation using MediaPipe (e.g. detects 'Hands Up' pose)
•	• Motion Detection based on frame difference
•	• Saves object detection data to detected_objects.xlsx
•	• All-in-one file – easy to run and customize

# Example Output

- Green boxes = Detected objects
- Yellow boxes = Humans
- Text overlay for hand poses and motion detection

# Dependencies
Install all required libraries with:
pip install -r requirements.txt
Or manually install:
pip install opencv-python numpy pandas openpyxl torch ultralytics mediapipe

# How to Run
•	• Clone the repo:
   git clone https://github.com/yourusername/object-pose-motion-detector.git
   cd object-pose-motion-detector
•	• Run the script:
   python main.py
•	• Press Q to quit the webcam view.
How It Works

- YOLOv8 detects objects frame by frame
- MediaPipe Pose identifies key human joints
- A custom rule checks if both hands are above the nose (for "Hands Up" detection)
- OpenCV computes frame differences to detect general motion
- Detections are saved to a .xlsx spreadsheet with timestamps

# Output File

- detected_objects.xlsx contains:
  - Timestamp of detection
  - Object name
  - Detection confidence

# Requirements
• Python 3.8+
• Windows 10 (tested)
• Webcam
Useful Links
•	YOLOv8 (Ultralytics): https://docs.ultralytics.com/
•	MediaPipe Pose: https://google.github.io/mediapipe/solutions/pose.html
•	OpenCV Documentation: https://docs.opencv.org/
Future Ideas
•	• Export logs to CSV or database
•	• Detect specific gestures like waving, pointing
•	• Add voice feedback using pyttsx3
•	• Build a GUI with tkinter or PyQt

# License
This project is licensed under the MIT License.

# Credits
Created by Imam Ud Doula 

Built with 💡 and ☕ using OpenCV, YOLOv8, and MediaPipe
