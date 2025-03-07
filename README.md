# Real-Time-Drowsiness-Detection

This project is a real-time drowsiness detection system that monitors a person's eye aspect ratio (EAR) using OpenCV and Mediapipe. If the EAR value falls below a threshold, the system triggers an alert sound. Additionally, the system detects emotions using DeepFace.

## Features
- **Real-time face and eye detection** using Mediapipe.
- **Drowsiness detection** based on Eye Aspect Ratio (EAR).
- **Audio alert** when drowsiness is detected.
- **Emotion detection** using DeepFace.
- **FPS calculation** for performance monitoring.

## Requirements
Make sure you have Python installed, then install the required dependencies using:

```bash
pip install opencv-python numpy mediapipe scipy deepface
```

For Windows users, `winsound` is used for audio alerts, which is built into Python.

## Running the Project
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/drowsiness_detection.git
   cd drowsiness_detection
   ```

2. Run the script:
   ```bash
   python drowsiness_detection.py
   ```

3. The program will start capturing video from your webcam. If drowsiness is detected, an alert sound will play.

4. Press `q` to exit the program.

## Project Structure
```
├── drowsiness_detection.py  # Main script
├── README.md                # Documentation
```

## How It Works
1. **Face and Eye Detection**: Uses Mediapipe Face Mesh to locate eyes in real-time.
2. **Drowsiness Detection**: Computes the EAR value and triggers an alert if it stays below a threshold for 1 second.
3. **Emotion Detection**: DeepFace analyzes emotions every 20 frames in a separate thread.
4. **Performance Optimization**: Frame skipping and threading improve FPS.

## Example Output
- When drowsiness is detected:
  ```
DROWSINESS ALERT!
Beep sound plays
  ```
- Detected emotion is displayed on the screen.

## License
This project is open-source and available under the MIT License.

---

Feel free to contribute .

