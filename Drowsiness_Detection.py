from scipy.spatial import distance as dist
import cv2
import winsound
import time
from deepface import DeepFace
import mediapipe as mp
import numpy as np

import threading

# Beep settings
frequency = 2500
duration = 1000

# Function to calculate Eye Aspect Ratio (EAR)
def eyeAspectRatio(eye):
    
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Initialize parameters
earThresh = 0.25  # Lower EAR threshold for sensitivity
frame_skip = 2   # Process every 2nd frame for better FPS
emotion_interval = 20  # Process emotion every 20 frames
fps_display_interval = 1  # Seconds to calculate FPSu
frame_count = 0

# Mediapipe face mesh setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)



# Start video capture
cam = cv2.VideoCapture(0)
fps_start_time = time.time()
ear_low_start = None  # Timer to track low EAR duration
emotion_analysis_frame = 0  # Track frames for emotion analysis
emotion_result = None  # Sto re emotion result in a thread-safe manner

# Function to run emotion analysis in a separate thread
def analyze_emotion(frame):
    global emotion_result
    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        emotion_result = result[0]['dominant_emotion']
    except Exception:
        emotion_result = None

while True:
    ret, frame = cam.read()
    if not ret:
        break

    # Resize frame to speed up processing
    frame = cv2.resize(frame, (640, 480))
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark

            # Extract landmarks for the left and right eyes
            leftEye = [
                (int(landmarks[i].x * frame.shape[1]), int(landmarks[i].y * frame.shape[0]))
                for i in [33, 160, 158, 133, 153, 144]
            ]
            rightEye = [
                (int(landmarks[i].x * frame.shape[1]), int(landmarks[i].y * frame.shape[0]))
                for i in [362, 385, 387, 263, 373, 380]
            ]

            # Calculate EAR
            leftEAR = eyeAspectRatio(leftEye)
            rightEAR = eyeAspectRatio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0

            # Check drowsiness based on EAR and use time-based alert
            if ear < earThresh:
                if ear_low_start is None:
                    ear_low_start = time.time()
                elif time.time() - ear_low_start >= 1:  # 1-second threshold
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    winsound.Beep(frequency, duration)
            else:
                ear_low_start = None

            # Draw eye contours
            cv2.polylines(frame, [np.array(leftEye, dtype=np.int32)], True, (0, 255, 0), 1)
            cv2.polylines(frame, [np.array(rightEye, dtype=np.int32)], True, (0, 255, 0), 1)

    # Emotion detection every 20 frames using threading
    if frame_count % emotion_interval == 0:
        emotion_thread = threading.Thread(target=analyze_emotion, args=(frame,))
        emotion_thread.start()

    # Display emotion result if available
    if emotion_result:
        cv2.putText(frame, f"Emotion: {emotion_result}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Calculate FPS
    frame_count += 1
    elapsed_time = time.time() - fps_start_time
    if elapsed_time > fps_display_interval:
        fps = frame_count / elapsed_time
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        frame_count = 0
        fps_start_time = time.time()

    # Show the video frame
    cv2.imshow("Drowsiness Detection", frame)

    # Exit on pressing 'q'
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Release resources
cam.release()
cv2.destroyAllWindows()
