import cv2


haar_cascade =  "model/haarcascade_frontalface_default.xml"

cap = cv2.VideoCapture(0)

cap.set(3,640)
cap.set(4,480)

while True:
    ret, frame = cap.read()
    face_cascades = cv2.CascadeClassifier(haar_cascade)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    face = face_cascades.detectMultiScale(frame_gray, 1.1, 4)
    for (x,y,w,h) in face:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
    
    cv2.imshow("Frame", frame)
    
    if cv2.waitKey(1) == ord('q'):
        break