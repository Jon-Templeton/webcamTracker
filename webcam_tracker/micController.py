from datetime import datetime
import cv2 as cv

class MicController:
    """A class to control the microphone status based on eye focus and mouth movement analysis."""

    def __init__(self):
        """
        Initializes the MicController with default settings and timestamps.
        """
        self.current_time = datetime.now().timestamp()
        
        # Mic Properties
        self.mic_on = False
        self.mic_status_list = []
        self.speaking_buffer_sec = 4
        
        # Eye Properties
        self.eyes_focus = False
        self.eye_focus_list = []
        
        # Mouth Properties
        self.talking = False
        self.mouth_opening_previous = 0.0
        self.mouth_threshold = .001
        self.talking_list = []
        self.last_speak_time = datetime.now().timestamp() - 60
        
    def _determine_talking(self, mouth_opening:float):
        """
        Determines if the subject is talking based on recent movement of the mouth.

        param mouth_opening: The current mouth opening measurement.
        """
        # Determine if lips moved from last frame
        mouth_change = abs(self.mouth_opening_previous - mouth_opening)
        talking = mouth_change > self.mouth_threshold
        self.mouth_opening_previous = mouth_opening
        
        # Track last 31 frames of talking
        self.talking_list.append(talking)
        if len(self.talking_list) > 30:
            self.talking_list.pop(0)
        else:
            # Need 31 frames of data before making a decision
            self.talking = False
            return
        
        # talking if True for 66% of frames
        self.talking = self.talking_list.count(True) >= (len(self.talking_list) * .66)
                
    def _determine_eyes_focus(self, current_eyes_focus:bool):
        """
        Determines if the subject's eyes are focused based on recent eye tracking data.

        param current_eyes_focus: Boolean indicating the current focus status of the eyes.
        """
        # Track last 31 frames of eye focus
        self.eye_focus_list.append(current_eyes_focus)
        if len(self.eye_focus_list) > 30:
            self.eye_focus_list.pop(0)
        else:
            # Need 31 frames of data before making a decision
            self.eyes_focus = False
            return
        
        # If True for 66% of frames, eyes are focused
        self.eyes_focus = self.eye_focus_list.count(True) >= (len(self.eye_focus_list) * .66)
            
    def set_mic_status(self, mouth_opening:float, current_eyes_focus:bool, image) -> str:
        """
        Sets the microphone status based on eye focus and mouth movement.

        param mouth_opening: The current mouth opening measurement.
        param current_eyes_focus: Boolean indicating the current focus status of the eyes.
        param image: The image where the status text will be displayed.
        return: A string indicating the current microphone status.
        """
        # Update object properties
        self.current_time = datetime.now().timestamp()
        
        # Update Eye Properties
        self._determine_eyes_focus(current_eyes_focus)
        if self.eyes_focus:
            cv.putText(image, "Eyes Focused", (50, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
        else:
            cv.putText(image, "Eyes Not Focused", (50, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)
        
        # Update Mouth Properties
        self._determine_talking(mouth_opening)
        if self.talking:
            cv.putText(image, "Talking", (50, 150), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
        else:
            cv.putText(image, "Not Talking", (50, 150), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)
        
        if self.talking and self.eyes_focus: self.last_speak_time = self.current_time
        
        # Set Mic Status
        self.mic_status_list.append(self.eyes_focus and self.talking)
        if len(self.mic_status_list) > 30:
            self.mic_status_list.pop(0)
        else: 
            # Need 31 frames of data before making a decision
            # Leave mic status as is
            return "Gathering Data"
        
        if self.mic_status_list.count(True) >= (len(self.mic_status_list) * .66):
            # Mic should be on
            self.mic_on = True
            return "Listening"
        # TO MUTE: Time since last speaking > buffer
        elif (self.current_time - self.last_speak_time) > self.speaking_buffer_sec:
            # Mic should be off  
            self.mic_on = False
            return "Muted"
        else:
            # Speaking or focus is False, but last_speak_time still within buffer
            # Mic should be on
            self.mic_on = True
            return "Listening"