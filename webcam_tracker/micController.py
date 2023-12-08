from datetime import datetime

class MicController:
    def __init__(self):
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
        self.mouth_threshold = .005
        self.talking_list = []
        self.last_speak_time = datetime.now().timestamp() - 60
        
    def _determine_talking(self, mouth_opening:float):
        # Determine if lips moved from last frame
        mouth_change = abs(self.mouth_opening_previous - mouth_opening)
        talking = mouth_change > self.mouth_threshold
        
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
            
    def set_mic_status(self, mouth_opening:float, current_eyes_focus:bool) -> str:
        # Update object properties
        self.current_time = datetime.now().timestamp()
        
        # Update Eye Properties
        self._determine_eyes_focus(current_eyes_focus)           
        
        # Update Mouth Properties
        self._determine_talking(mouth_opening)
        
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