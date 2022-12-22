import cv2                                                       
import dlib 
from scipy.spatial import distance                                                              
from imutils import face_utils                                          
import numpy as np
from keras.models import load_model
import argparse

# Parsing command-line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-vw", "--isVideoWriter", type=bool, default=False)
args = vars(ap.parse_args())

# Emotion Handling
emotion_offsets = (20, 40)
emotions = {
    0: {
        "emotion": "Angry",
        "color": (193, 69, 42)
    },
    1: {
        "emotion": "Disgust",
        "color": (164, 175, 49)
    },
    2: {
        "emotion": "Fear",
        "color": (40, 52, 155)
    },
    3: {
        "emotion": "Happy",
        "color": (23, 164, 28)
    },
    4: {
        "emotion": "Sad",
        "color": (164, 93, 23)
    },
    5: {
        "emotion": "Suprise",
        "color": (218, 229, 97)
    },
    6: {
        "emotion": "Neutral",
        "color": (108, 72, 200)
    }
}

#iterating over its parts and extracting the x and y coordinates of each part
def shapePoints(shape):
    coords = np.zeros((68, 2), dtype="int")
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

#tuple containing the coordinates and dimensions of the rectangle is then returned by the function
def rectPoints(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x, y, w, h)


def eye_aspect_ratio(eye):
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])

	C = distance.euclidean(eye[0], eye[3])
	eye = (A + B) / (2.0 * C)

	return eye


faceLandmarks = "faceDetection/models/dlib/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

emotionModelPath = 'models/emotionModel.hdf5'  
emotionClassifier = load_model(emotionModelPath, compile=False)
emotionTargetSize = emotionClassifier.input_shape[1:3]

# starting the webcam through cv2 module
cap = cv2.VideoCapture(0)

count = 0
total = 0

if args["isVideoWriter"] == True:
    fourrcc = cv2.VideoWriter_fourcc("M", "J", "P", "G")
    capWidth = int(cap.get(3))
    capHeight = int(cap.get(4))
    videoWrite = cv2.VideoWriter("output.avi", fourrcc, 22,
                                 (capWidth, capHeight))

while True:
    # stores video in img
    success,img = cap.read()
    # converts img into black and white
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # dlib lib 
    faces = detector(imgGray)

    for face in faces:
        landmarks = predictor(imgGray,face)

        landmarks = face_utils.shape_to_np(landmarks)
        leftEye = landmarks[42:48]
        rightEye = landmarks[36:42]

        leftEye = eye_aspect_ratio(leftEye)
        rightEye = eye_aspect_ratio(rightEye)

        eye = (leftEye + rightEye) / 2.0

        if eye<0.3:
            count+=1
        else:
            if count>=3:
                total+=1

            count=0
        
    # Showing text in video output
    cv2.putText(img, "Blink Count: {}".format(total), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
   
    img = cv2.resize(img, (720, 480))

    grayFrame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(grayFrame, 0)
    for rect in rects:
        shape = predictor(grayFrame, rect)
        points = shapePoints(shape)
        (x, y, w, h) = rectPoints(rect)
        grayFace = grayFrame[y:y + h, x:x + w]
        try:
            grayFace = cv2.resize(grayFace, (emotionTargetSize))
        except:
            continue

        grayFace = grayFace.astype('float32')
        grayFace = grayFace / 255.0
        grayFace = (grayFace - 0.5) * 2.0
        grayFace = np.expand_dims(grayFace, 0)
        grayFace = np.expand_dims(grayFace, -1)
        emotion_prediction = emotionClassifier.predict(grayFace)
        emotion_probability = np.max(emotion_prediction)
        if (emotion_probability > 0.36):
            emotion_label_arg = np.argmax(emotion_prediction)
            color = emotions[emotion_label_arg]['color']
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.line(img, (x, y + h), (x + 20, y + h + 20),
                     color,
                     thickness=2)
            cv2.rectangle(img, (x + 20, y + h + 20), (x + 110, y + h + 40),
                         color, -1)
            cv2.putText(img, emotions[emotion_label_arg]['emotion'],
                        (x + 25, y + h + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)
        else:
            color = (255, 255, 255)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

    # showing the video output
    cv2.imshow("video", img)
    if cv2.waitKey(1) & 0xff==ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
