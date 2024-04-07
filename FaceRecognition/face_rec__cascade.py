import face_recognition
import os, sys
import cv2
import numpy as np
import math

def face_confidence(face_distance, face_match_threshold=0.6):
    range = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (range * 2.0)
    
    if face_distance > face_match_threshold:
        return str(round(linear_val * 100, 2)) + '%'
    else:
        value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5)*2, 0.2)))*100
        return str(round(value, 2)) + '%'
    
class FaceRecognition:
    face_locations = []
    face_encodings = []
    face_names = []
    known_face_encodings = []
    known_face_names = []
    process_current_frame = True
    
    def __init__(self):
        self.encode_faces()
    
    def encode_faces(self):
        for image in os.listdir('faces'):
            face_img = face_recognition.load_image_file(f'faces/{image}')
            face_encoding = face_recognition.face_encodings(face_img)[0]
            #Get name file
            (filename, ext) = os.path.splitext(image)
            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(filename)
        print(self.known_face_names)
        
    def run_recognition(self):
        haar_cascade =  "model/haarcascade_frontalface_default.xml"
        #cap = cv2.VideoCapture("C:/Users/admin/Downloads/class.mp4")
        cap = cv2.VideoCapture(0)
        cap.set(3, 640)  # Set width
        cap.set(4, 480)  # Set height
        
        # Minimum width and height for the window size to be recognized as a face
        minW = 0.1 * cap.get(3)
        minH = 0.1 * cap.get(4)
            
        
        
        if not cap.isOpened():
            sys.exit('Video source not found...')
        
        while True:
            ret, frame = cap.read()
            
            if self.process_current_frame:
                # Resize and change frame to RGB
                #small_frame =  cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
                #rgb_small_frame = small_frame[:, :, ::-1]
                
                #Find all faces in current frame 
                #self.face_locations = face_recognition.face_locations(rgb_small_frame)
                #self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)

                self.face_names = []
                for face_encoding in self.face_encodings:
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                    name = 'Unknown'
                    
                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    
                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                        confident = face_confidence(face_distances[best_match_index])
                        
                    self.face_names.append(f'{name}({confident})')
                    
            self.process_current_frame = not self.process_current_frame
            
            #
            face_cascades = cv2.CascadeClassifier(haar_cascade)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            
            faces = face_cascades.detectMultiScale(
                frame_gray,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(int(minW), int(minH)),
            )
            
            #Display annotations
            for (x,y,w,h) in faces:# , name in zip(faces, self.face_names):                
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                #cv2.putText(frame, name, (w, y - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255), 2)
                for name in self.face_names:
                    print(name)
                #cv2.rectangle(frame, (left, top), (right, bottom ), (0,0,255), 2)
                #cv2.rectangle(frame, (left, bottom - 35 ), (right, bottom ), (0,0,255), -1)
                #cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,255,255), 1)
             
            cv2.imshow('Face Recognition', frame)
            
            if cv2.waitKey(1) == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        
if __name__ == '__main__':
    fr = FaceRecognition()
    fr.run_recognition()
    