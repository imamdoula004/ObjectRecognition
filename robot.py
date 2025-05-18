import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO
import mediapipe as mp
import time
import os
from collections import deque

# Initialize YOLO model
yolo_model = YOLO('yolov8n.pt')  # Best balance between speed and accuracy

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Initialize video capture
cap = cv2.VideoCapture(0)

# Create or load data file
data_file = 'detected_objects.xlsx'
if not os.path.exists(data_file):
    df = pd.DataFrame(columns=['Timestamp', 'Object', 'Confidence'])
    df.to_excel(data_file, index=False)
else:
    df = pd.read_excel(data_file)

# Helper for motion detection
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# Wrist history for waving detection
wrist_history = deque(maxlen=20)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Object detection with YOLO
    results = yolo_model(frame)[0]
    objects_detected = []

    for result in results.boxes:
        cls_id = int(result.cls[0])
        label = yolo_model.model.names[cls_id]
        confidence = float(result.conf[0])
        x1, y1, x2, y2 = map(int, result.xyxy[0])

        # Save to dataframe
        timestamp = pd.Timestamp.now()
        objects_detected.append({'Timestamp': timestamp, 'Object': label, 'Confidence': confidence})

        # Draw boxes
        color = (0, 255, 0)  # Default green
        if label.lower() in ["person", "human"]:
            color = (0, 255, 255)  # Yellow for humans
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Save to Excel
    if objects_detected:
        df = pd.concat([df, pd.DataFrame(objects_detected)], ignore_index=True)
        df.to_excel(data_file, index=False)

    # Pose estimation
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_pose = pose.process(rgb)
    if results_pose.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        landmarks = results_pose.pose_landmarks.landmark
        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        nose = landmarks[mp_pose.PoseLandmark.NOSE]

        # Hands up detection
        if left_wrist.y < nose.y and right_wrist.y < nose.y:
            cv2.putText(frame, "Hands Up", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 3)

        # Crossed hands detection
        if abs(left_wrist.x - right_wrist.x) < 0.05 and abs(left_wrist.y - right_wrist.y) < 0.05:
            cv2.putText(frame, "Hands Crossed", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # Pointing detection (right arm extended)
        if abs(right_wrist.y - right_shoulder.y) < 0.1 and right_wrist.x > right_shoulder.x + 0.1:
            cv2.putText(frame, "Pointing Right", (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)

        # Nodding detection (simple vertical motion of nose)
        if not hasattr(pose, 'nose_y_history'):
            pose.nose_y_history = deque(maxlen=10)
        pose.nose_y_history.append(nose.y)
        if len(pose.nose_y_history) == pose.nose_y_history.maxlen:
            y_diff = max(pose.nose_y_history) - min(pose.nose_y_history)
            if y_diff > 0.05:
                cv2.putText(frame, "Nodding", (30, 170), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

        # Waving detection
        wrist_history.append(right_wrist.x)
        if len(wrist_history) == wrist_history.maxlen:
            x_range = max(wrist_history) - min(wrist_history)
            y_diff = abs(right_wrist.y - right_shoulder.y)
            if x_range > 0.15 and y_diff < 0.1:
                cv2.putText(frame, "Waving", (30, 210), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 3)

    # Motion detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(prev_gray, gray)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    motion_level = np.sum(thresh)
    if motion_level > 500000:
        cv2.putText(frame, "Motion Detected", (30, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    prev_gray = gray

    cv2.imshow("Object, Pose, and Motion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
