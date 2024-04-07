# Demo-Face-Recognition

# face_recognition:
### Requirements: 
* Dlib:
```
pip install https://github.com/jloh02/dlib/releases/download/v19.22/dlib-19.22.99-cp310-cp310-win_amd64.whl
```
* face_recognition:
```
pip install face_recognition
```
# Haar Cascades:
### Steps:

`face_taker.py`
1) Take pictures using the `face_taker.py` script. After you enter the ID number, the script will save 30 images of your face in the `images` folder. The ID number represents a single face. The ID MUST be an integer and incremental starting with 1, then 2, 3, ...
Note: Make sure your face is centered. The window will collapse when all the 30 pictures are taken.


`face_train.py`

2) The `face_tain.py` script will train a model to recognize all the faces from the 30 images taken using `face_taker.py` and save the training output in the `training.yml` file.


`face_recognizer.py`

3) The `face_recognizer.py` is the main script. You need to append each person's name in the index equal to the ID provided in `face_taker.py` script. The program will recognize the face according to the ID. i.e., If Joe has an id 1, his name should appear in the list as index 1 like such:

`names = ['None', 'Joe']. # Don't remove the None`

Requirements:

- `pip install opencv-python`
- `pip install opencv-contrib-python --upgrade` or `pip install opencv-contrib-python`
