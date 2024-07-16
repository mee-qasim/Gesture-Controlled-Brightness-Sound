import cv2
import numpy as np
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import screen_brightness_control as sbc
import threading

import mediapipe as mp

# Initialize MediaPipe Hands model
mpHands = mp.solutions.hands
hands = mpHands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

# Video capture setup
vidObj = cv2.VideoCapture(0)
vidObj.set(cv2.CAP_PROP_FRAME_WIDTH, 1800)
vidObj.set(cv2.CAP_PROP_FRAME_HEIGHT, 900)

# Audio volume setup
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volRange = volume.GetVolumeRange()
minVolume = volRange[0]
maxVolume = volRange[1]

# Screen brightness setup
minBrightness = 0
maxBrightness = 100

# Function to set volume based on hand distance
def setVolume(dist):
    vol = np.interp(int(dist), [35, 215], [minVolume, maxVolume])
    volume.SetMasterVolumeLevel(vol, None)

# Function to set brightness based on hand distance
def setBrightness(dist):
    brightness = np.interp(int(dist), [35, 230], [minBrightness, maxBrightness])
    sbc.set_brightness(int(brightness))

while True:
    _, frame = vidObj.read()
    frame = cv2.flip(frame, 1)

    # Convert the BGR frame to RGB for processing with MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)

            # Calculate distances between specific landmarks (for example purposes)
            xr1, yr1 = int(hand_landmarks.landmark[4].x * frame.shape[1]), int(hand_landmarks.landmark[4].y * frame.shape[0])
            xr2, yr2 = int(hand_landmarks.landmark[8].x * frame.shape[1]), int(hand_landmarks.landmark[8].y * frame.shape[0])
            dist = math.hypot(xr2 - xr1, yr2 - yr1)

            # Adjust volume or brightness based on detected hand
            if results.multi_handedness[0].classification[0].label == 'Left':
                setBrightness(dist)
            elif results.multi_handedness[0].classification[0].label == 'Right':
                setVolume(dist)
            elif results.multi_handedness[0].classification[0].label == 'Both':
                # Example of handling both hands simultaneously using threading
                xl1, yl1 = int(hand_landmarks.landmark[4].x * frame.shape[1]), int(hand_landmarks.landmark[4].y * frame.shape[0])
                xl2, yl2 = int(hand_landmarks.landmark[8].x * frame.shape[1]), int(hand_landmarks.landmark[8].y * frame.shape[0])
                distl = math.hypot(xl2 - xl1, yl2 - yl1)

                t1 = threading.Thread(target=setVolume, args=(dist,))
                t2 = threading.Thread(target=setBrightness, args=(distl,))
                
                t1.start()
                t2.start()

    cv2.imshow("stream", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
vidObj.release()
cv2.destroyAllWindows()
