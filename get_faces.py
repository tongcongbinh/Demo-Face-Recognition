import cv2
from facenet_pytorch import MTCNN
import torch
import os

device =  torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Device used: " + str(device))
# Create a face data folder
DATA_DIR = "./Data/Faces"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Create a folder of face names
face_name = input("Enter your name: ")
if not os.path.exists(os.path.join(DATA_DIR, str(face_name))):
    os.makedirs(os.path.join(DATA_DIR, str(face_name)))    

print('Collecting data for face "{}"...'.format(face_name))

FACE_PATH = os.path.join(DATA_DIR, face_name)
dataset_size = 30

mtcnn = MTCNN(thresholds= [0.6, 0.7, 0.7],
              margin = 20,
              keep_all=False, 
              select_largest = True, 
              post_process=False, 
              device = device
)
url1 = "rtsp://admin:L22478F1@192.168.88.221:554/cam/realmonitor?channel=1&subtype=1"
cap = cv2.VideoCapture(0)

directions = ['AHEAD','to the LEFT', 'to the RIGHT', 'UP','DOWN', ]   
direction_type = ''


for direction in directions:
    # if direction == 'AHEAD': direction_type = 'A'
    # elif direction == 'to the LEFT': direction_type = 'L'
    # elif direction == 'to the RIGHT': direction_type = 'R'
    # elif direction == 'UP': direction_type = 'U'
    # elif direction == 'DOWN': direction_type = 'D'

    match direction:
        case 'AHEAD': direction_type = 'A'
        case 'to the LEFT': direction_type = 'L'
        case 'to the RIGHT': direction_type = 'R'
        case 'UP': direction_type = 'U'
        case 'DOWN': direction_type = 'D'

    while True:
        isSuccess, frame = cap.read()
        
        if isSuccess:
            boxes, _ = mtcnn.detect(frame)
            if boxes is not None:
                for box in boxes:
                    bbox = list(map(int,box.tolist()))
                    frame = cv2.rectangle(frame,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,0,255),3)
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, 'Please look {}!'.format(direction), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (155,231,20), 2,cv2.LINE_AA)
        cv2.putText(frame, 'Press "S" to start collecting!', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (155,231,20), 2,cv2.LINE_AA)
        cv2.imshow('Face Capturing', frame)
        if cv2.waitKey(20) == ord('s'):
            break   
    
    # Save pictures
    count = 0
    while mtcnn(frame) is not None and count < dataset_size:
        isSuccess, frame = cap.read()
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.putText(frame, 'Collecting...', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (155,231,20), 2,cv2.LINE_AA)
        cv2.imshow('Face Capturing', frame)
        path = str(FACE_PATH + '/{}_{}.jpg'.format(direction_type, str(count)))
        cv2.waitKey(30)
        face_img = mtcnn(frame_rgb, save_path = path)
        count+=1

print("Done!")
cap.release()
cv2.destroyAllWindows()
