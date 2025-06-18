import cv2
import mediapipe as mp
import numpy as np
import joblib

# Load model and scaler
clf = joblib.load('gesture_model.joblib')
scaler = joblib.load('scaler.joblib')

# List of gesture names in label order (update as needed)
GESTURES = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "plus", "minus", "multiply", "divide", "equals", "clear"
]
# Map gesture names to calculator symbols
GESTURE_TO_SYMBOL = {
    "0": "0", "1": "1", "2": "2", "3": "3", "4": "4", "5": "5",
    "6": "6", "7": "7", "8": "8", "9": "9",
    "plus": "+", "minus": "-", "multiply": "*", "divide": "/",
    "equals": "=", "clear": "C"
}

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=2
)

cap = cv2.VideoCapture(1)  # Change index if needed

def extract_landmarks(multi_hand_landmarks):
    # Always return a 126-length vector (2 hands x 21 landmarks x 3)
    feature_vector = np.zeros(126)
    for i, hand_landmarks in enumerate(multi_hand_landmarks):
        coords = []
        for lm in hand_landmarks.landmark:
            coords.extend([lm.x, lm.y, lm.z])
        feature_vector[i*63:(i+1)*63] = coords
    return feature_vector

# Calculator state
current_input = ""
operator = ""
result = ""
last_gesture = None
gesture_cooldown = 15  # frames to wait before accepting the same gesture again
cooldown_counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    gesture_name = None

    if results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
        if len(results.multi_hand_landmarks) == 2:
            mp_drawing.draw_landmarks(frame, results.multi_hand_landmarks[1], mp_hands.HAND_CONNECTIONS)
        feature_vector = extract_landmarks(results.multi_hand_landmarks)
        feature_vector = scaler.transform([feature_vector])
        pred = clf.predict(feature_vector)[0]
        gesture_name = GESTURES[int(pred)]

        # Only process gesture if it's different from the last one and cooldown expired
        if (gesture_name != last_gesture) and (cooldown_counter == 0):
            symbol = GESTURE_TO_SYMBOL[gesture_name]
            if symbol.isdigit():
                if operator == "":
                    current_input += symbol
                else:
                    current_input += symbol
            elif symbol in "+-*/":
                if current_input and (not operator or result):
                    if result:
                        current_input = str(result)
                        result = ""
                    operator = symbol
                    current_input += " " + operator + " "
            elif symbol == "=":
                try:
                    result = str(eval(current_input))
                except Exception:
                    result = "Error"
                operator = ""
            elif symbol == "C":
                current_input = ""
                operator = ""
                result = ""
            last_gesture = gesture_name
            cooldown_counter = gesture_cooldown
    else:
        gesture_name = None

    if cooldown_counter > 0:
        cooldown_counter -= 1
    elif gesture_name != last_gesture:
        last_gesture = None  # Reset last_gesture when hand is removed

    # Display
    y0 = 40
    cv2.putText(frame, f'Gesture: {gesture_name if gesture_name else ""}', (10, y0),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f'Input: {current_input}', (10, y0+40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f'Result: {result}', (10, y0+80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('Hand Gesture Calculator', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()