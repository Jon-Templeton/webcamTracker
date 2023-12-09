import os

# How to mute/unmute the microphone on macOS
# Not currently used in the project
# Volume is a value between 0 and 100

def set_microphone_volume(volume):
    # AppleScript command to set the input volume
    script = f"""
    set volume input volume {volume}
    """
    os.system(f"osascript -e '{script}'")