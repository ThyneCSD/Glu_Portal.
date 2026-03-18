import mediapipe as mp
import cv2
import time
import numpy as np
import pyvirtualcam

print("Loading hand tracking model...")

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
Image = mp.Image
ImageFormat = mp.ImageFormat

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') #laad het detectie model in.


options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2 #aantal handen wowowow.
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

#fullscreen window maken
cv2.namedWindow("Hand Tracking", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Hand Tracking", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

p_time = 0
frame_count = 0

def draw_landmarks(frame, hand_landmarks):
    """Draw hand landmarks on frame"""
    h, w, c = frame.shape

    for i in [0,1,2,3,5,6,7,9,10,11,13,14,15,17,18,19]: #20 punten
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
    success, frame = capture.read()
    if not success:
        print("Camera not working")
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)
    results = hands_tracker.detect_for_video(mp_image, frame_count)
    frame_count += 1

    paper_detected = False
    faces = []

    if results.hand_landmarks:
        for hand_landmarks in results.hand_landmarks:
            draw_landmarks(frame, hand_landmarks)

            if is_hand_open(hand_landmarks):
                paper_detected = True
                cv2.putText(frame, "Paper", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            elif is_scissors(hand_landmarks):
                cv2.putText(frame, "Scissors", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)
            else:
                cv2.putText(frame, "Rock", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
       
    #alleen gezicht blokkeren als paper gesture
    if paper_detected:
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), -1) # verander de -1 naar 2 of 3 om een rand te maken om het gezicht heen in plaats van het gezicht te blokken.

    c_time = time.time()
    fps = 1 / (c_time - p_time) if c_time != p_time else 0
    p_time = c_time
    # cv2.putText(frame, f'FPS: {int(fps)}', (10, 30),
                #cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # overlay logo on first detected face (centered) if available; fallback bottom-right
    if logo_bgr is not None and mask is not None:
        if len(faces) > 0:
            x, y, w, h = faces[0]
            # square area at top of face region
            overlay_size = min(size, w, h)
            ox1 = max(0, x + w//2 - overlay_size//2)
            oy1 = max(0, y + h//2 - overlay_size//2)
            ox2 = ox1 + overlay_size
            oy2 = oy1 + overlay_size
            if ox2 <= frame.shape[1] and oy2 <= frame.shape[0]:
                roi = frame[oy1:oy2, ox1:ox2]
                resized_logo = cv2.resize(logo_bgr, (overlay_size, overlay_size))
                resized_mask = cv2.resize(mask, (overlay_size, overlay_size))
                mask_inv = cv2.bitwise_not(resized_mask)
                roi_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
                logo_fg = cv2.bitwise_and(resized_logo, resized_logo, mask=resized_mask)
                frame[oy1:oy2, ox1:ox2] = cv2.add(roi_bg, logo_fg)
        else:
            y1, y2 = frame.shape[0] - size - 10, frame.shape[0] - 10
            x1, x2 = frame.shape[1] - size - 10, frame.shape[1] - 10
            roi = frame[y1:y2, x1:x2]
            if roi.shape[0] == size and roi.shape[1] == size:
                mask_inv = cv2.bitwise_not(mask)
                roi_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
                logo_fg = cv2.bitwise_and(logo_bgr, logo_bgr, mask=mask)
                frame[y1:y2, x1:x2] = cv2.add(roi_bg, logo_fg)

    #frame naar virtual webcam sturen
    cam.send(frame)
    cam.sleep_until_next_frame()

    #fullscreen laten zien zonder stretch
    cv2.imshow("Hand Tracking", frame)

    if cv2.waitKey(1) & 0xFF == 27:  #press escape to quit/klik op escape om te stoppen.
        break

capture.release()
cv2.destroyAllWindows()