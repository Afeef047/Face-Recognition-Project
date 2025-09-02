 Real-Time Face Recognition using OpenCV

A Python project to detect and recognize faces in real-time from a webcam feed using the LBPH algorithm.


Requirements

You will need the following Python libraries:
* opencv-contrib-python
* numpy
* Pillow

You can install them all with pip:
```bash
pip install opencv-contrib-python numpy pillow
```

---
## How to Run

Follow these three steps in order.

### Step 1: Generate Face Data
-   Open `Data_Generator.py`.
-   Change the `user_id` to a unique number for the person you want to add.
-   Run the script. It will open your webcam and save 100 images to the `data/` folder.
```bash
python Data_Generator.py
```

### Step 2: Train the Model
-   Run the `Classifier.py` script.
-   This script reads the images from the `data/` folder and creates a `classifier.yml` file, which is your trained model.
```bash
python Classifier.py
```

### Step 3: Recognize Faces
-   Run the `Recognize.py` script to start the webcam.
-   The program will detect faces and draw a label with the name of the recognized person.
```bash
python Recognize.py
```

---

##  File Descriptions

* `Data_Generator.py`: Captures and saves face images from a webcam for training.
* `Classifier.py`: Trains the face recognizer on the saved images and creates `classifier.yml`.
* `Recognize.py`: Uses the webcam and the trained `classifier.yml` to perform real-time face recognition.
* `haarcascade_*.xml`: Pre-trained models from OpenCV used for detecting objects like faces and eyes.
