import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def set_microphone_volume(volume):
    # AppleScript command to set the input volume
    script = f"""
    set volume input volume {volume}
    """
    os.system(f"osascript -e '{script}'")


