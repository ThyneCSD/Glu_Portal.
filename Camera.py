import cv2
webcam=cv2.VideoCapture(0)

stop = False
while stop==False:
    ret,frame=webcam.read()

    if ret==True:
        cv2.imshow("Webcam",frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            stop = True