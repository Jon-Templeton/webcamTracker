import numpy as np

class MouthTracker:
    """A class for tracking mouth movement in an image using facial landmarks."""
    
    def __init__(self, image, landmarks):
        """
        Initialize the MouthTracker with an image and facial landmarks.

        param image: The image to process.
        param landmarks: Facial landmarks detected in the image.
        """
        self.image = image
        self.landmarks = landmarks

    def track_mouth(self) -> float:
        """
        Track the mouth movement by measuring the distance between the top and bottom lip.

        return: The distance between the top and bottom lip, indicating mouth opening.
        """
        # Landmarks
        top_lip = self.landmarks[13]
        bottom_lip = self.landmarks[14]

        # Calculate mouth opening
        x1, y1 = top_lip.x, top_lip.y
        x2, y2 = bottom_lip.x, bottom_lip.y
        
        # Distance equation between two points
        mouth_opening = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return mouth_opening