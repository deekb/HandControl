import cv2
import mediapipe as mp
import pyautogui
import tkinter
import os
import math
import time

root = tkinter.Tk()

pyautogui.PAUSE = 0
pyautogui.FAILSAFE = False

BUFFER = 0.75
SHOW_IMAGE = True

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

smoothedValues = []

screen_width, screen_height = (root.winfo_screenwidth(), root.winfo_screenheight())  # (1920, 1080)
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
        model_complexity=0,
        max_num_hands=1,
        min_detection_confidence=0.75,
        min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        image = cv2.flip(image, 1)
        if not success:
            print("[WARN]: Got blank or invalid frame from camera")
            continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
                
                middleX = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x
                middleY = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
                
                indexX = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x
                indexY = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
                
                thumbX = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x
                thumbY = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y
                
                
                
                normalizedX = (indexX + thumbX) / 2
                normalizedY = (indexY + thumbY) / 2
                
                smoothedValues.append((normalizedX, normalizedY))
                if len(smoothedValues) > 10:
                    smoothedValues = smoothedValues[-10:]
                
                smoothedX = sum([pair[0] for pair in smoothedValues]) / len(smoothedValues)
                smoothedY = sum([pair[1] for pair in smoothedValues]) / len(smoothedValues)
                                
                if math.sqrt((abs(indexX - thumbX) + abs(indexY - thumbY))) < 0.4:
                    if math.sqrt((abs(middleX - indexX) + abs(middleY - indexY))) < 0.2 and middleY < indexY + 10:
                        pyautogui.mouseDown(button='right')
                    else:
                        pyautogui.mouseDown(button='left')
                elif not(math.sqrt((abs(indexX - thumbX) + abs(indexY - thumbY))) < 0.4):
                    pyautogui.mouseUp(button='left')
                    pyautogui.mouseUp(button='right')
                
                
                indexFingerTipX = (((smoothedX - 0.5) / BUFFER) + 0.5) * screen_width
                indexFingerTipY = (((smoothedY - 0.5) / BUFFER) + 0.5) * screen_height

                print(f"\r({indexFingerTipX}, {indexFingerTipY})", end=(" " * 100))
                indexFingerTipX = sorted((0, indexFingerTipX, screen_width))[1]  # clamp the coordinates to the screen
                indexFingerTipY = sorted((0, indexFingerTipY, screen_height))[1]

                pyautogui.moveTo(int(indexFingerTipX), int(indexFingerTipY))

        if SHOW_IMAGE: cv2.imshow("Image", image)
        if cv2.waitKey(1) & 0xFF == 27:
            break
cap.release()
os._exit(0)
