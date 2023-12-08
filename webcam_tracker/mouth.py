import numpy as np

class MouthTracker:
    def __init__(self, image, landmarks):
        self.image = image
        self.landmarks = landmarks

    def track_mouth(self) -> float:
        # Mouth Movement Tracking
        top_lip = self.landmarks[13]
        bottom_lip = self.landmarks[14]

        # Calculate mouth opening
        x1, y1 = top_lip.x, top_lip.y
        x2, y2 = bottom_lip.x, bottom_lip.y
        
        # Distance equation between two points
        mouth_opening = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        
        return mouth_opening

