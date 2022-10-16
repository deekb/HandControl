import cv2
import mediapipe as mp
import pyautogui
import tkinter
root = tkinter.Tk()

pyautogui.PAUSE = 0
pyautogui.FAILSAFE = False

BUFFER = 0.9
SHOW_IMAGE = True

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

screen_width, screen_height = (root.winfo_screenwidth(),root.winfo_screenheight()) #(1920, 1080)
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        image = cv2.flip(image, 1)
        if not success:
            print("Ignoring empty camera frame.")
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

                indexFingerTipX = (((hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x - 0.5) / BUFFER) + 0.5) * screen_width
                indexFingerTipY = (((hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y - 0.5) / BUFFER) + 0.5) * screen_height
                print(f"\r({indexFingerTipX}, {indexFingerTipY})", end="")
                if indexFingerTipX > screen_width:
                    indexFingerTipX = screen_width
                elif indexFingerTipX < 0:
                    indexFingerTipX = 0
                if indexFingerTipY > screen_height:
                    indexFingerTipY = screen_height
                elif indexFingerTipY < 0:
                    indexFingerTipY = 0


                pyautogui.moveTo(int(indexFingerTipX), int(indexFingerTipY))


        if SHOW_IMAGE: cv2.imshow("Image", image)
        if cv2.waitKey(1) & 0xFF == 27:
            break
cap.release()