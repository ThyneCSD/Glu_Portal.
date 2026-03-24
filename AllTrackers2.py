import mediapipe as mp
import cv2
import time
import numpy as np
import pyvirtualcam
import os
import sys
print("Loading hand tracking model...")

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
Image = mp.Image
ImageFormat = mp.ImageFormat

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') #laad het detectie model in.


script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "hand_landmarker.task")
if not os.path.exists(model_path):
    sys.stderr.write(
        f"Model not found at {model_path}\n"
        "Download 'hand_landmarker.task' from the MediaPipe Tasks model zoo and place it in this folder.\n"
    )
    sys.exit(1)

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1 #aantal handen wowowow.
)

print("Initializing hand tracker...")
hands_tracker = HandLandmarker.create_from_options(options)
print("Hand tracker ready!")

capture = None
for camera_index in range(5):
    test_capture = cv2.VideoCapture(camera_index) #0 staat voor welke webcam je wilt gebruiken. Hierbij is 0 de ingebouwde webcam.
    if test_capture.isOpened():
        capture = test_capture
        print(f"Camera found at index {camera_index}")
        break
    test_capture.release()


# Read logo and resize
logo = cv2.imread('Test.png', cv2.IMREAD_UNCHANGED)
size = 100
if logo is not None:
    logo = cv2.resize(logo, (size, size))
    if logo.shape[2] == 4:
        logo_bgr = logo[:, :, :3]
        alpha = logo[:, :, 3] / 255.0
        mask = (alpha * 255).astype(np.uint8)
    else:
        logo_bgr = logo
        mask = 255 * np.ones(logo_bgr.shape[:2], dtype=np.uint8)
else:
    logo_bgr = None
    mask = None


if capture is None:
    print("No camera found. Exiting.")
    exit(1)

#virtual webcam starten
width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
cam = pyvirtualcam.Camera(width=width, height=height, fps=30)

#fullscreen window maken (lukt niet heel goed)
cv2.namedWindow("Hand Tracking", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Hand Tracking", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

p_time = 0
frame_count = 0

missionActive = True
got_Number = False

def draw_landmarks(frame, hand_landmarks):
    """Draw hand landmarks on frame"""
    h, w, c = frame.shape

    for i in [0,1,2,3,5,6,7,9,10,11,13,14,15,17,18,19]: #20 punten voor op de handen.
        landmark = hand_landmarks[i]
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        cv2.circle(frame, (x, y), 5, (0, 155, 0), -1)

    for i in [4,8,12,16,20]: 
        landmark = hand_landmarks[i]
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)
    

def is_scissors(hand_landmarks):
    index_folded = hand_landmarks[8].x < hand_landmarks[6].x
    middle_folded = hand_landmarks[12].x < hand_landmarks[10].x
    ring_folded = hand_landmarks[16].y > hand_landmarks[14].y
    pinky_folded = hand_landmarks[20].y > hand_landmarks[18].y

    return index_folded and middle_folded and ring_folded and pinky_folded

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

def count_fingers(hand_landmarks):
    count = 0
    tips_pips = ((8, 6), (12, 10), (16, 14), (20, 18))
    for tip, pip in tips_pips:
        if hand_landmarks[tip].y < hand_landmarks[pip].y:
            count += 1

    is_right = hand_landmarks[17].x < hand_landmarks[5].x

# de duim moet anders want je steek die vaak zeiwaarts uit inplaats van omhoog.
    if is_right:
        if hand_landmarks[4].x > hand_landmarks[3].x:
            count += 1
    else:
        if hand_landmarks[4].x < hand_landmarks[3].x:
            count += 1

    return count


while True:
    success, frame = capture.read()
    if not success:
        print("Camera not working")
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)
    results = hands_tracker.detect_for_video(mp_image, frame_count)
    frame_count += 1




    if got_Number == False:
          random_fingerNumber = np.random.randint(1, 6)
          print(f"Mission: put up {random_fingerNumber} fingers!")

    if random_fingerNumber is not None:
        got_Number = True
        cv2.putText(frame, f"gotNumber: {got_Number}", (300, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    if missionActive:
        cv2.putText(frame, f"Mission: put up {random_fingerNumber} fingers!", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        
    paper_detected = False
    faces = []

    if results.hand_landmarks:
        for hand_landmarks in results.hand_landmarks:
            draw_landmarks(frame, hand_landmarks)

            fingers_extended = count_fingers(hand_landmarks)
            if fingers_extended >= 5:
                # paper_detected = True
                cv2.putText(frame, f"Fingers: {fingers_extended} (max i hope.)", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            elif fingers_extended >= 1:
                cv2.putText(frame, f"Fingers: {fingers_extended}", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                if missionActive and fingers_extended == random_fingerNumber:
                    cv2.putText(frame, f"Mission complete!", (50, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, f"Fingers: {fingers_extended}", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        
       
    # alleen gezicht blokkeren als paper gesture: detecteer nu gezichten en blokkeer ze
    if paper_detected:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), -1) # verander de -1 naar 2 of 3 om een rand te maken om het gezicht heen in plaats van het gezicht te blokken.

    c_time = time.time()
    fps = 1 / (c_time - p_time) if c_time != p_time else 0
    p_time = c_time
    # cv2.putText(frame, f'FPS: {int(fps)}', (10, 30),
                #cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    # frame naar virtual webcam sturen (pyvirtualcam verwacht RGB)
    cam.send(frame[:, :, ::-1])
    cam.sleep_until_next_frame()

    #fullscreen laten zien zonder stretch
    cv2.imshow("Hand Tracking", frame)

    if cv2.waitKey(1) & 0xFF == 27:  #press escape to quit/klik op escape om te stoppen.
        break

capture.release()
cv2.destroyAllWindows()