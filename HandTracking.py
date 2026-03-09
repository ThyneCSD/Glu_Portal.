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

def is_scissors(hand_landmarks):
    index_folded = hand_landmarks[8].x < hand_landmarks[6].x
    middle_folded = hand_landmarks[12].x < hand_landmarks[10].x
    ring_folded = hand_landmarks[16].y > hand_landmarks[14].y
    pinky_folded = hand_landmarks[20].y > hand_landmarks[18].y

    return index_folded and middle_folded and ring_folded and pinky_folded

def middle_finger_extended(hand_landmarks):
    return hand_landmarks[12].y < hand_landmarks[10].y

def is_hand_open(hand_landmarks):
    fingers_extended = 0

    # Thumb/Duim
    if hand_landmarks[4].x < hand_landmarks[3].x:
        fingers_extended += 1
        
    # Index/Wijsvinger
    if hand_landmarks[8].y < hand_landmarks[6].y:
        fingers_extended += 1

    # Middle/middelvinger
    if hand_landmarks[12].y < hand_landmarks[10].y:
        fingers_extended += 1

    # Ring/Ringvinger
    if hand_landmarks[16].y < hand_landmarks[14].y:
        fingers_extended += 1

    # Pinky/pink
    if hand_landmarks[20].y < hand_landmarks[18].y:
        fingers_extended += 1

    return fingers_extended >= 4

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

        if is_hand_open(hand_landmarks):
            cv2.putText(frame, "Paper", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        elif is_scissors(hand_landmarks):
                cv2.putText(frame, "Scissors", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)
        else:
            cv2.putText(frame, "Rock", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

    c_time = time.time()
    fps = 1 / (c_time - p_time) if c_time != p_time else 0
    p_time = c_time
    cv2.putText(frame, f'FPS: {int(fps)}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Resize the frame
    resized_frame = cv2.resize(frame, (1550, 1000)) #1920 en 1200 is voor mijn 2e scherm thuis. #1550 en 1000 is voor mijn laptop scherm.
    cv2.imshow("Hand Tracking", resized_frame)

    if cv2.waitKey(1) & 0xFF == 27:  #press escape to quit/klik op escape om te stoppen.
        break

cap.release()
cv2.destroyAllWindows()