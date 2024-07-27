import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('../models/emotion_detection_model.h5')

# Define emotions
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']


def predict_emotion(face):
    """
    Predict the emotion from a face image using the pre-trained model.

    Args:
    face (numpy array): Preprocessed face image.

    Returns:
    str: Predicted emotion.
    """
    prediction = model.predict(face)
    emotion = emotions[np.argmax(prediction)]
    return emotion


# Capture video from webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(gray, (48, 48))
    face = face / 255.0
    face = np.expand_dims(face, axis=-1)
    face = np.expand_dims(face, axis=0)

    # Predict emotion
    emotion = predict_emotion(face)

    # Display the result
    cv2.putText(frame, emotion, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('Emotion Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
