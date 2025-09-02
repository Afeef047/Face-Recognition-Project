
````markdown
# Real-Time Face Recognition System

![Project Demo GIF](https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExdDk5em1idXJqMmJrbmVwZ2EwbDN2MG56ZzVjcTRiaW5iaHN2dGcxOSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/L3pr9V2y2oA5g5nsyI/giphy.gif)
*(**Important:** Replace the placeholder GIF above with a screen recording of your own project in action! This is the best way to showcase your work.)*

## Overview

This project is a real-time face recognition system built using **Python** and **OpenCV**. It can detect faces from a live video stream, identify known individuals, and has the capability to detect other facial features like eyes, nose, and mouth. The recognition is based on the **Local Binary Patterns Histograms (LBPH)** algorithm.

This system is broken down into three main stages:
1.  **Data Collection**: Capturing facial images to build a dataset for training.
2.  **Model Training**: Training the LBPH recognizer on the collected dataset.
3.  **Real-Time Recognition**: Detecting and identifying faces from a live webcam feed.

---

##  Key Features

* **Real-Time Face Detection**: Uses Haar Cascades to efficiently detect faces in every frame of the video stream.
* **LBPH Face Recognition**: Trains a model to recognize and label individuals it has been trained on.
* **Easy Data Generation**: Includes a script to easily capture and store face samples for training new individuals.
* **Facial Feature Detection**: `Detector.py` script showcases the ability to identify eyes, nose, and mouth within a detected face region.
* **Performance Evaluation**: `evaluation.py` script generates a confusion matrix and other metrics to evaluate the model's accuracy.

---

##  Technologies Used

* **Language**: Python 3.8+
* **Core Libraries**:
    * [OpenCV](https://opencv.org/) (`opencv-contrib-python`) - For all computer vision tasks (image processing, video capture, face detection/recognition).
    * [NumPy](https://numpy.org/) - For numerical operations and handling image arrays.
    * [Pillow (PIL)](https://python-pillow.org/) - For image manipulation.
* **For Evaluation**:
    * [scikit-learn](https://scikit-learn.org/stable/) - For generating the confusion matrix and performance metrics.
    * [Matplotlib](https://matplotlib.org/) & [Seaborn](https://seaborn.pydata.org/) - For data visualization and plotting results.
    * [Pandas](https://pandas.pydata.org/) - For data manipulation.

---

##  Setup and Installation

Follow these steps to set up the project on your local machine.

**1. Clone the Repository**
```bash
git clone [https://github.com/your-username/face-recognition-system.git](https://github.com/your-username/face-recognition-system.git)
cd face-recognition-system
```

**2.Create a Virtual Environment (Recommended)**
A virtual environment keeps your project dependencies isolated.
```bash
# For Windows
python -m venv venv
.\venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

**3. Install Dependencies**
All required libraries can be installed using the `requirements.txt` file.
```bash
pip install opencv-contrib-python numpy pillow scikit-learn matplotlib seaborn pandas
```
*(Alternatively, you can create a `requirements.txt` file with the libraries listed above and run `pip install -r requirements.txt`)*

---

## How to Use

The process is divided into three simple steps. Run the scripts in the following order:

**Step 1: Generate the Dataset (`Data_Generator.py`)**
* Run the data generator script to capture images of a person's face.
* Before running, you **must** edit the file and change the `user_id` variable to a unique number for each new person.
* The script will capture 100 images via webcam and save them in a `data/` folder (you may need to create this folder first).
```bash
python Data_Generator.py
```

**Step 2: Train the Classifier (`Classifier.py`)**
* Once you have collected the data, run the training script.
* This will read the images from the `data/` folder, train the LBPH recognizer, and save the trained model as `classifier.yml` in the root directory.
```bash
python Classifier.py
```

**Step 3: Run the Recognition (`Recognize.py`)**
* You're all set! Run the recognition script to start your webcam.
* The system will draw a box around detected faces and label them with the name of the recognized person based on the `if/elif` conditions in the script.
```bash
python Recognize.py
```

---

## Evaluating the Model

If you want to measure the model's performance, you can use the provided evaluation scripts.

1.  **Collect Results**: First, run the modified recognition script (`mod_recognize.py`) which saves predictions to a `recognition_results.csv` file. You may need to manually input the true labels during this process.
2.  **Generate Plots**: Then, run the evaluation script to generate a confusion matrix and other performance plots.
```bash
python evaluation.py
```
This will display plots that help you understand the model's accuracy, precision, and recall.

---

