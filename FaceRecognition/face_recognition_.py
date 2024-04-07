import cv2
from simple_facerec import SimpleFacerec
import face_recognition

# Encode faces from a folder
sfr = SimpleFacerec()
sfr.load_encoding_images("faces/")


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    #Detect faces
    face_locations, face_names = sfr.detect_known_faces(frame)
    for face_loc, name in zip(face_locations, face_names):
        y1, x1, y2, x2 = face_loc[0],face_loc[1],face_loc[2],face_loc[3]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 4)
        cv2.putText(frame, name, (x2, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255), 2)
        
    cv2.imshow("Face Recognition", frame)
    
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
