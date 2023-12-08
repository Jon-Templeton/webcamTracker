import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Mesh module
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize MediaPipe drawing module
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Function to calculate distance between two landmarks
def calculate_distance(landmark1, landmark2):
    x1, y1 = landmark1.x, landmark1.y
    x2, y2 = landmark2.x, landmark2.y
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def find_pupil(image, landmarks, direction:str):
    height, width, _ = image.shape
    
    if direction == 'right':
        eye_landmarks = [263,466,388,387,386,385,384,398,362,382,381,380,374,373,390,249]
    else:
        eye_landmarks = [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246]
    
    # Define shape of the eye
    eye_points = [landmarks[point] for point in eye_landmarks]
    hull = cv2.convexHull(np.array([(landmark.x * width, landmark.y * height) for landmark in eye_points], dtype=np.int32))

    mask = np.zeros((height, width), np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)

    # Apply the mask to the original image
    eye_image = cv2.bitwise_and(image, image, mask=mask)

    # Preprocess the eye image
    gray_eye = cv2.cvtColor(eye_image, cv2.COLOR_BGR2GRAY)
    blurred_eye = cv2.GaussianBlur(gray_eye, (3, 3), 0)

    # Thresholding to isolate the darker pupil region
    _, thresholded_eye = cv2.threshold(blurred_eye, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(thresholded_eye, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Pupil is 2nd largest contour
    if contours and len(contours) >= 2:
        # Sort the contours by area in descending order
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
        pupil_contour = sorted_contours[1]

        # Calculate the moments of the pupil contour
        M = cv2.moments(pupil_contour)

        # Calculate the x and y coordinates of the center of the contour
        if M["m00"] != 0:
            eye_x = int(M["m10"] / M["m00"])
            eye_y = int(M["m01"] / M["m00"])

            # Determine if the pupil is looking forward or not
            # Calculate horizontal and vertical eye sizes
            if direction == 'right':
                horizontal_point1, horizontal_point2 = landmarks[263], landmarks[362]
                vertical_point1, vertical_point2 = landmarks[386], landmarks[374]
            else:
                horizontal_point1, horizontal_point2 = landmarks[133], landmarks[33]
                vertical_point1, vertical_point2 = landmarks[159], landmarks[145]

            horizontal_eye_size = np.sqrt((horizontal_point1.x * width - horizontal_point2.x * width) ** 2 + 
                                        (horizontal_point1.y * height - horizontal_point2.y * height) ** 2)
            vertical_eye_size = np.sqrt((vertical_point1.x * width - vertical_point2.x * width) ** 2 + 
                                        (vertical_point1.y * height - vertical_point2.y * height) ** 2)

            # Set thresholds
            horizontal_threshold = horizontal_eye_size / 4
            vertical_threshold = vertical_eye_size / 4

            # Find the center of the eye
            eye_center_norm = np.mean([(landmarks[point].x, landmarks[point].y) for point in eye_landmarks], axis=0)
            eye_center = (int(eye_center_norm[0] * width), int(eye_center_norm[1] * height))

            # Calculate horizontal and vertical distances from the center
            horizontal_distance = abs(eye_x - eye_center[0])
            vertical_distance = abs(eye_y - eye_center[1])
            
            # How close is the pupil to the center?
            # Positive value = eye is centered
            # Negative value = eye is not centered
            horizontal_difference = horizontal_threshold - horizontal_distance
            vertical_difference = vertical_threshold - vertical_distance
            
            return (horizontal_difference, vertical_difference)
    return ()

# Capture video from the default camera
cap = cv2.VideoCapture(0)

# Variables to hold previous mouth opening
prev_mouth_opening = 0
mouth_opening_change_threshold = .005
talking_grace_period = 30 # frames of video
talking_counter = 0

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    # Convert the BGR image to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Flip the image horizontally
    image = cv2.flip(image, 1)
    
    results = face_mesh.process(image)

    # Draw the face mesh annotations on the image
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            
            # Eye Tracking
            pupil_centering_R = find_pupil(image, face_landmarks.landmark, "right")
            pupil_centering_L = find_pupil(image, face_landmarks.landmark, "left")

            # Average differences if both eyes are visible
            if not pupil_centering_R and not pupil_centering_L:
                # No eyes detected
                horizontal_difference = -1
                vertical_difference = -1
            elif not pupil_centering_R:
                horizontal_difference = pupil_centering_L[0]
                vertical_difference = pupil_centering_L[1]
            elif not pupil_centering_L:
                horizontal_difference = pupil_centering_R[0]
                vertical_difference = pupil_centering_R[1]
            else:
                # Find the average of the horizontal and vertical differences
                horizontal_difference = np.mean([pupil_centering_R[0], pupil_centering_L[0]])
                vertical_difference = np.mean([pupil_centering_R[1], pupil_centering_L[1]])

            cv2.putText(image, f'Both:{horizontal_difference}, {vertical_difference}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Check if the pupil is centered
            if horizontal_difference > 0 and vertical_difference > 0:
                status_text = 'Looking Forward'
            else:
                status_text = 'Not Looking Forward !!!!!!!'

            # Put text on image
            cv2.putText(image, status_text, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            # Mouth Movement Tracking
            top_lip = face_landmarks.landmark[13]
            bottom_lip = face_landmarks.landmark[14]
            
            # Calculate mouth opening
            mouth_opening = calculate_distance(top_lip, bottom_lip)
            
            # Check if the mouth opening has changed significantly
            if abs(mouth_opening - prev_mouth_opening) > mouth_opening_change_threshold:
                talking_counter = talking_grace_period

            prev_mouth_opening = mouth_opening
            
            # Determine talking status based on the counter
            if talking_counter > 0:
                status_text = 'Talking'
                talking_counter -= 1
            else:
                status_text = 'Not Talking'

            # Put text on image
            cv2.putText(image, status_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)


    # Display the annotated image
    cv2.imshow('FaceMesh', image)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()