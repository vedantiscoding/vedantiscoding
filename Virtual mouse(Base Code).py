import cv2
import mediapipe as mp
import pyautogui
from pynput.mouse import Controller, Button

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mouse = Controller()

# Initialize the hand tracker
with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
    cap = cv2.VideoCapture(0)
    
    # Get screen dimensions
    screen_width, screen_height = pyautogui.size()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        # BGR 2 RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Flip on horizontal
        image = cv2.flip(image, 1)

        # Set flag
        image.flags.writeable = False

        # Detections
        results = hands.process(image)

        # Set flag to true
        image.flags.writeable = True

        # RGB 2 BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Rendering results
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get the coordinates of the index finger tip (you can use other landmarks for mouse control)
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                height, width, _ = image.shape
                x, y = int(index_finger_tip.x * width), int(index_finger_tip.y * height)

                # Map the hand coordinates to screen coordinates manually
                screen_x = int(x * screen_width / width)
                screen_y = int(y * screen_height / height)

                # Set the mouse cursor position
                mouse.position = (screen_x, screen_y)

                # Check for left-click gesture (e.g., index finger and thumb touching)
                if hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y > hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y:
                    mouse.press(Button.left)
                    mouse.release(Button.left)

                # Check for right-click gesture (e.g., middle finger and thumb touching)
                if hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y > hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y:
                    mouse.press(Button.right)
                    mouse.release(Button.right)

                # Check for scrolling gesture (e.g., scroll with two fingers)
                if hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y:
                    mouse.scroll(0, 2)  # Scroll up
                elif hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y > hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y:
                    mouse.scroll(0, -2)  # Scroll down

        cv2.imshow('Hand Tracking', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
