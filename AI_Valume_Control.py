import cv2
import mediapipe as mp
from math import hypot
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from comtypes import CoInitialize
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import numpy as np
import comtypes

# Initialize the camera
cap = cv2.VideoCapture(0)  # You may need to adjust the camera index

# Check if the camera is opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Initialize COM for pycaw
CoInitialize()

# Initialize Mediapipe hands
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

# Get the speakers
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(comtypes.GUID("{5CDF2C82-841E-4546-9722-0CF74078229A}"), CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

volMin, volMax, _ = volume.GetVolumeRange()  # Unpack the tuple correctly
current_vol = volume.GetMasterVolumeLevel()

# Smoothing parameters
alpha = 0.2  # Adjust this value for smoother changes

while True:
    success, img = cap.read()

    if not success:
        print("Error: Failed to read frame from the camera.")
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    lmList = []
    if results.multi_hand_landmarks:
        for hand_landmark in results.multi_hand_landmarks:
            for id, lm in enumerate(hand_landmark.landmark):
                h, w, _ = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
            mpDraw.draw_landmarks(img, hand_landmark, mpHands.HAND_CONNECTIONS)

    if lmList:
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]

        cv2.circle(img, (x1, y1), 4, (255, 0, 0), cv2.FILLED)
        cv2.circle(img, (x2, y2), 4, (255, 0, 0), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)

        length = hypot(x2 - x1, y2 - y1)

        target_vol = np.interp(length, [15, 220], [volMin, volMax])
        current_vol = alpha * target_vol + (1 - alpha) * current_vol  # Smoothing
        volume.SetMasterVolumeLevel(current_vol, None)

    cv2.putText(img, f"Vol: {int(round(current_vol * 100))}%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('Hand Tracking', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()