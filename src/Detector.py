import cv2

# Drawing Boxes Around the Face and Features
def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)
    coordinates = []
    for (x, y, w, h) in features:
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
        cv2.putText(img, text, (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        coordinates.append((x, y, w, h))
    return coordinates, img

# Detect Function
def detect(img, faceCascade, eyesCascade, mouthCascade, noseCascade):
    color = {"Blue": (255, 0, 0), "Red": (0, 0, 255), "Green": (0, 255, 0), "Purple": (128, 0, 128), "Orange": (255, 165, 0)}
    face_coordinates, img_with_boxes = draw_boundary(img, faceCascade, 1.1, 10, color["Green"], "Face")
    
    # Detect Face
    if face_coordinates:
        for (x, y, w, h) in face_coordinates:
            if w > 0 and h > 0:
                cropped_img = img[y:y+h, x:x+w]
                gray_cropped = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
                
                # Detect Eyes
                eyes = eyesCascade.detectMultiScale(gray_cropped, scaleFactor=1.1, minNeighbors=15)
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(cropped_img, (ex, ey), (ex+ew, ey+eh), color["Red"], 2)
                    cv2.putText(cropped_img, "Eye", (ex, ey-4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color["Red"], 1, cv2.LINE_AA)
                
                # Detect Mouth
                mouth_roi = cropped_img[int(0.6*h):int(0.9*h), :]
                gray_mouth_roi = cv2.cvtColor(mouth_roi, cv2.COLOR_BGR2GRAY)
                mouths = mouthCascade.detectMultiScale(gray_mouth_roi, scaleFactor=1.1, minNeighbors=15)
                for (mx, my, mw, mh) in mouths:
                    cv2.rectangle(mouth_roi, (mx, my), (mx+mw, my+mh), color["Blue"], 2)
                    cv2.putText(mouth_roi, "Mouth", (mx, my-4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color["Blue"], 1, cv2.LINE_AA)
                
                # Detect Nose
                nose_roi = cropped_img[int(0.3*h):int(0.6*h), int(0.3*w):int(0.7*w)]  
                gray_nose_roi = cv2.cvtColor(nose_roi, cv2.COLOR_BGR2GRAY)
                noses = noseCascade.detectMultiScale(gray_nose_roi, scaleFactor=1.1, minNeighbors=10)
                
                for (nx, ny, nw, nh) in noses:
                    cv2.rectangle(nose_roi, (nx, ny), (nx+nw, ny+nh), color["Purple"], 2)
                    cv2.putText(nose_roi, "Nose", (nx, ny-4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color["Purple"], 1, cv2.LINE_AA)

                
                
                cropped_img[int(0.3*h):int(0.6*h), int(0.3*w):int(0.7*w)] = nose_roi
                cropped_img[int(0.6*h):int(0.9*h), :] = mouth_roi
                img[y:y+h, x:x+w] = cropped_img
    return img_with_boxes

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eyesCascade = cv2.CascadeClassifier("haarcascade_eye.xml")
mouthCascade = cv2.CascadeClassifier("Mouth.xml")
noseCascade = cv2.CascadeClassifier("Nariz.xml")

# 0 if using laptop cam otherwise -1/1
video_capture = cv2.VideoCapture(0)

while True:
    ret, img = video_capture.read()
    if not ret:
        print("Failed to capture image")
        break

    img_with_boxes = detect(img, faceCascade, eyesCascade, mouthCascade, noseCascade)
    
    if img_with_boxes is not None and img_with_boxes.size > 0:
        cv2.imshow("Face Detector", img_with_boxes)

    if cv2.waitKey(1) & 0xFF == ord('s'):
        break

video_capture.release()
cv2.destroyAllWindows()
