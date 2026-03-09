import cv2
def main():

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') #laad het detectie model in.
    
    vidCapture = cv2.VideoCapture(0) #0 staat voor welke webcam je wilt gebruiken. Hierbij is 0 de ingebouwde webcam.
    if not vidCapture.isOpened():
        print("Error: Could not open webcam.")
        return
    
    while True:
        ret, frame = vidCapture.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
       
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), -1) # verander de -1 naar 2 of 3 om een rand te maken om het gezicht heen in plaats van het gezicht te blokken.
        
            # Resize the frame
            resized_frame = cv2.resize(frame, (1550, 1000)) #1920 en 1200 is voor mijn 2e scherm thuis. #1550 en 1000 is voor mijn laptop scherm.
            cv2.imshow("face Tracking", resized_frame)
        
        
        if cv2.waitKey(1) & 0xFF == 27:  #press escape to quit/klik op escape om te stoppen.
          break
    
    vidCapture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()