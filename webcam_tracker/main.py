import cv2
import mediapipe as mp
import numpy as np

from eyes import EyeTracker
from mouth import MouthTracker
from micController import MicController

# Initialize MediaPipe Face Mesh module
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Capture video from the default camera
cap = cv2.VideoCapture(0)

# Initiliaze MicController
mic_controller = MicController()

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    # Convert the BGR image to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Flip the image horizontally
    image = cv2.flip(image, 1)
    # Run Mediapipe Face Mesh
    results = face_mesh.process(image)
    # Convert the RGB image to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            
            # Eye Tracking
            eyes = EyeTracker(image, face_landmarks.landmark)
            eyes_focus = eyes.track_focus()
            
            # Mouth Tracking
            # finds current distance between top and bottom lip
            mouth = MouthTracker(image, face_landmarks.landmark)
            mouth_opening = mouth.track_mouth()
            
            # Microphone Control
            # Use eye_focus and mouth_opening to determine when mic should be on
            mic_status = mic_controller.set_mic_status(mouth_opening, eyes_focus, image)

            # Put text on image
            if mic_status == "Listening":
                cv2.putText(image, mic_status, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)  # Green Text
            else:
                cv2.putText(image, mic_status, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)  # Red Text

    # Display the annotated image
    cv2.imshow('webcamTracker', image)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()