import cv2 as cv
import numpy as np

class EyeTracker:
    def __init__(self, image, landmarks):
        self.image = image
        self.focus = False
        
        self.landmarks = landmarks
        self.landmarks_num_R = [263,466,388,387,386,385,384,398,362,382,381,380,374,373,390,249]
        self.landmarks_num_L = [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246]
        
        self.horizontal_thresh_ratio = 0.25
        self.vertical_thresh_ratio = 0.25
        
        
    def track_focus(self) -> bool:
        # Eye Tracking
        pupil_centering_R = self._find_pupil("right")
        pupil_centering_L = self._find_pupil("left")

        # Average values if both eyes are visible
        if not pupil_centering_L and not pupil_centering_R:
            # No eyes detected
            horizontal_centering = -1
            vertical_centering = -1
        elif not pupil_centering_R:
            # Only left eye detected
            horizontal_centering = pupil_centering_L[0]
            vertical_centering = pupil_centering_L[1]
        elif not pupil_centering_L:
            # Only right eye detected
            horizontal_centering = pupil_centering_R[0]
            vertical_centering = pupil_centering_R[1]
        else:
            # Both eyes detected
            # Find the average of the horizontal and vertical differences
            horizontal_centering = np.mean([pupil_centering_L[0], pupil_centering_R[0]])
            vertical_centering = np.mean([pupil_centering_L[1], pupil_centering_R[1]])

        # Check if the pupil is centered
        if horizontal_centering > 0 and vertical_centering > 0:
            self.focus = True
        else:
            self.focus = False

        return self.focus

    def _process_image(self, direction:str) -> np.ndarray:
        height, width, _ = self.image.shape
        eye_landmarks = self.landmarks_num_R if direction == 'right' else self.landmarks_num_L
        
        # Define shape of the eye
        eye_points = [self.landmarks[point] for point in eye_landmarks]
        hull = cv.convexHull(np.array([(landmark.x * width, landmark.y * height) for landmark in eye_points], dtype=np.int32))

        mask = np.zeros((height, width), np.uint8)
        cv.fillConvexPoly(mask, hull, 255)

        # Apply the mask to the original image
        eye_image = cv.bitwise_and(self.image, self.image, mask=mask)

        # Preprocess the eye image
        gray_eye = cv.cvtColor(eye_image, cv.COLOR_BGR2GRAY)
        blurred_eye = cv.GaussianBlur(gray_eye, (3, 3), 0)

        # Thresholding to isolate the darker pupil region
        _, thresholded_eye = cv.threshold(blurred_eye, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)

        # Find contours
        contours, _ = cv.findContours(thresholded_eye, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        # Pupil is 2nd largest contour
        if contours and len(contours) >= 2:
            # Sort the contours by area in descending order
            sorted_contours = sorted(contours, key=cv.contourArea, reverse=True)
            pupil_contour = sorted_contours[1]
            return pupil_contour
        return None

    def _find_pupil(self, direction:str) -> tuple:
        height, width, _ = self.image.shape
        eye_landmarks = self.landmarks_num_R if direction == 'right' else self.landmarks_num_L
        
        # Find Pupil Contour
        pupil_contour = self._process_image(direction)
        if pupil_contour is None:
            return None

        # Calculate the moments of the pupil contour
        pupil_moments = cv.moments(pupil_contour)

        # Calculate the x and y coordinates of the center of the contour
        if pupil_moments["m00"] != 0:
            eye_x = int(pupil_moments["m10"] / pupil_moments["m00"])
            eye_y = int(pupil_moments["m01"] / pupil_moments["m00"])

            # Determine if the pupil is looking forward or not
            # Calculate horizontal and vertical eye sizes
            if direction == 'right':
                horizontal_point1, horizontal_point2 = self.landmarks[263], self.landmarks[362]
                vertical_point1, vertical_point2 = self.landmarks[386], self.landmarks[374]
            else:
                horizontal_point1, horizontal_point2 = self.landmarks[133], self.landmarks[33]
                vertical_point1, vertical_point2 = self.landmarks[159], self.landmarks[145]

            horizontal_eye_size = np.sqrt((horizontal_point1.x * width - horizontal_point2.x * width) ** 2 + 
                                        (horizontal_point1.y * height - horizontal_point2.y * height) ** 2)
            vertical_eye_size = np.sqrt((vertical_point1.x * width - vertical_point2.x * width) ** 2 + 
                                        (vertical_point1.y * height - vertical_point2.y * height) ** 2)

            # Set thresholds
            horizontal_threshold = horizontal_eye_size * self.horizontal_thresh_ratio
            vertical_threshold = vertical_eye_size * self.vertical_thresh_ratio

            # Find the center of the eye
            eye_center_norm = np.mean([(self.landmarks[point].x, self.landmarks[point].y) for point in eye_landmarks], axis=0)
            eye_center = (int(eye_center_norm[0] * width), int(eye_center_norm[1] * height))

            # Calculate horizontal and vertical distances from the center
            horizontal_distance = abs(eye_x - eye_center[0])
            vertical_distance = abs(eye_y - eye_center[1])
            
            # How close is the pupil to the center?
            # Positive value = eye is centered
            # Negative value = eye is not centered
            horizontal_centering = horizontal_threshold - horizontal_distance
            vertical_centering = vertical_threshold - vertical_distance
            
            return (horizontal_centering, vertical_centering)
        return None
