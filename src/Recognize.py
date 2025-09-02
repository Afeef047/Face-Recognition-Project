import cv2

def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, clf):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)
    for (x, y, w, h) in features:
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        id, _ = clf.predict(gray_img[y:y + h, x:x + w])
        
        if id == 1:
            cv2.putText(img, "Afeef", (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
            
        elif id == 2:
            cv2.putText(img, "Lakshay", (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        elif id == 3:
            
            cv2.putText(img, "Dr. Abhishek", (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
            
        elif id == 4:
            cv2.putText(img, "Ziyan Malik", (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        
        elif id == 5:
            cv2.putText(img, "Alok Kumar", (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
            
        elif id == 6:
            cv2.putText(img, "Yash", (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA) 
            
        elif id == 7:
            cv2.putText(img, "Rashid", (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
            
        elif id == 8:
            cv2.putText(img, "Ravi", (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)    
        
        elif id == 9:
            cv2.putText(img, "Aydindczx", (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)    

 
def recognize(img, clf, faceCascade):
    color = {"blue": (255, 0, 0), "red": (0, 0, 255), "green": (0, 255, 0), "white": (255, 255, 255), "purple": (128, 0, 128)}
    draw_boundary(img, faceCascade, 1.1, 10, color["white"], clf)
    return img

# Load classifiers
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
clf = cv2.face.LBPHFaceRecognizer_create()
clf.read("classifier.yml")

video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    print("Error: Could not open video capture.")
    exit()

while True:
    ret, img = video_capture.read()
    if not ret:
        print("Error: Failed to capture image.")
        break
    
    img = recognize(img, clf, faceCascade)
    
    cv2.imshow("Face Detection", img)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        break

video_capture.release()
cv2.destroyAllWindows()
