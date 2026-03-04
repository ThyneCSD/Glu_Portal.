import mediapipe as mp
import cv2
import time
import numpy as np

print("Loading hand tracking model...")

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
Image = mp.Image
ImageFormat = mp.ImageFormat

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2 #aantal handen wowowow.
)

print("Initializing hand tracker...")
hands_tracker = HandLandmarker.create_from_options(options)
print("Hand tracker ready!")

cap = None
for camera_index in range(5):
    test_cap = cv2.VideoCapture(0)
    if test_cap.isOpened():
        cap = test_cap
        print(f"Camera found at index {camera_index}")
        break
    test_cap.release()

if cap is None:
    print("No camera found. Exiting.")
    exit(1)

p_time = 0
frame_count = 0

def draw_landmarks(frame, hand_landmarks):
    """Draw hand landmarks on frame"""
    h, w, c = frame.shape

    for landmark in hand_landmarks:
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

while True:
    success, frame = cap.read()
    if not success:
        print("Camera not working")
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)
    results = hands_tracker.detect_for_video(mp_image, frame_count)
    frame_count += 1

    if results.hand_landmarks:
        for hand_landmarks in results.hand_landmarks:
            draw_landmarks(frame, hand_landmarks)

    c_time = time.time()
    fps = 1 / (c_time - p_time) if c_time != p_time else 0
    p_time = c_time
    cv2.putText(frame, f'FPS: {int(fps)}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Resize the frame
    resized_frame = cv2.resize(frame, (1920, 1200))
    cv2.imshow("Hand Tracking", resized_frame)

    if cv2.waitKey(1) & 0xFF == 27:  #klik op escape om te stoppen.
        break

cap.release()
cv2.destroyAllWindows()