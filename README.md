# Webcam-Based Interaction Controller

This project demonstrates a unique approach to control a microphone using webcam-based interaction, focusing on pupil and mouth movements. The code, written in Python, leverages OpenCV and MediaPipe for sophisticated facial feature tracking, providing a unique way to manage audio input during various user activities.

## Table of Contents

- [Demo](#demo)
- [Project Overview](#project-overview)
- [Features](#features)
- [Files](#files)
- [Requirements](#requirements)
- [Usage](#usage)

## Demo

https://github.com/Jon-Templeton/webcamTracker/assets/68302842/f03e1af3-7d84-49f6-9f0e-a15aff956d52

## Project Overview

MediaPipe's FaceMesh is utilized to accurately track the position of the lips and the outline of the eyes. However, FaceMesh does not inherently track pupil movements. To overcome this, additional computer vision techniques such as masking, blurring, thresholding, and contour detection have been implemented to enable precise pupil tracking. This enhanced eye tracking is pivotal in determining the user's focus.

Moreover, the system monitors mouth movements by calculating the variance in lip distance frame-to-frame, which is a reliable indicator of speech. This fully vision-based approach ensures that the microphone management is solely responsive to the user's actions, thereby avoiding unintended microphone activation due to ambient noise, music, or other voices.

The logic of the system is finely tuned to consider the practical aspects of speech and attention. It includes a grace period for mic muting, preventing the microphone from cutting out during brief pauses in speech. Additionally, decision-making is based not on a single frame but on the analysis of a sequence of frames. This results in increased accuracy, as it accounts for brief, natural fluctuations in focus and speech.

For optimal performance, it is recommended that the user's face be well-lit. This ensures the precision of the facial feature tracking, thereby enhancing the responsiveness and reliability of the microphone control.

## Features

- **Eye Tracking**: Determines if the user is focusing by tracking the position of the pupils.
- **Mouth Movement Tracking**: Measures the distance between the top and bottom lips to detect if the user is speaking.
- **Microphone Control**: Turns the microphone on or off based on the user's focus and speech activity, with a grace period and averaged frame analysis for enhanced accuracy.


## Files
- **'main.py'**: The main script to run the webcam tracker.
- **'eyes.py'**: Contains the EyeTracker class for pupil tracking and determining eye focus.
- **'mouth.py'**: Contains the MouthTracker class for monitoring mouth movement.
- **'micController.py'**: Contains the MicController class. Uses eye focus and mouth movements to decide when microphone should listen.
- **'mic_mute.py'**: Contains function to modify MacOS microphone input volume. *(Not currently implemented)*

## Requirements

- Python 3.6+
- OpenCV
- MediaPipe
- NumPy

## Usage
1. Run **'main.py'** to start the webcam tracker.
2. The program will display the webcam feed with real-time annotations indicating eye focus, speech activity, and microphone status.
3. Press 'q' to quit the application.