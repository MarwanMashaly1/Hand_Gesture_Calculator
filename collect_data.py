import cv2
import mediapipe as mp
import numpy as np
import os
import time

# Configuration
DATA_DIR = 'data/raw_landmarks'
NUM_SAMPLES = 150
GESTURES = {
    "0": "Closed_Fist",
    "1": "Index_Finger",
    "2": "Two_Fingers",
    "3": "Three_Fingers",
    "4": "Four_Fingers",
    "5": "Open_Palm",
    "6": "Left_5_Right_1",   # Left hand open, Right hand index finger
    "7": "Left_5_Right_2",   # Left hand open, Right hand two fingers
    "8": "Left_5_Right_3",   # Left hand open, Right hand three fingers
    "9": "Left_5_Right_4", 
    "plus": "cross fingers",
    "minus": "two_fingers_side",
    "multiply": "thumbs_up",
    "divide": "thumbs_down",
    "equals": "hand_side_back",
    "clear": "hand_down_closed"
}
TWO_HAND_GESTURES = ["6", "7", "8", "9"]

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=2 
)

# Create Directories
for gesture_name in GESTURES.keys():
    os.makedirs(os.path.join(DATA_DIR, gesture_name), exist_ok=True)
    print(f"Ensured directory: {os.path.join(DATA_DIR, gesture_name)}")

def extract_landmarks(hand_landmarks):
    # Extracts (x, y, z) for 21 landmarks and flattens them into a 1D array (63 values)
    landmark_coords = []
    for lm in hand_landmarks.landmark:
        # MediaPipe already normalizes x,y,z relative to the image/hand detection box.
        landmark_coords.extend([lm.x, lm.y, lm.z])
    return np.array(landmark_coords)

def collect_gesture_data():
    cap = cv2.VideoCapture(1) # 0 for default webcam

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("--- Starting Data Collection ---")
    print("Instructions:")
    print("1. Perform the gesture displayed on screen.")
    print(f"2. Press 's' to start collecting {NUM_SAMPLES} samples.")
    print("3. Press 'q' to quit.")
    print("----------------------------------")

    for gesture_key, gesture_display_name in GESTURES.items():
        is_two_hand_gesture = gesture_key in TWO_HAND_GESTURES
        print(f"\n--- Get ready for gesture: '{gesture_display_name}' (Label: '{gesture_key}') ---")
        if is_two_hand_gesture:
            print(">>> This gesture requires TWO hands. Please show both hands clearly. <<<")
        else:
            print(">>> This gesture requires ONE hand. Please show one hand clearly. <<<")

        count = 0
        collecting = False

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Flip frame horizontally for a mirrored view (optional, but common)
            frame = cv2.flip(frame, 1)
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the frame with MediaPipe Hands
            results = hands.process(image_rgb)

            display_frame = frame.copy() # Create a copy for drawing

            hand_landmarks_list = []
            if results.multi_hand_landmarks:
                # Store hand landmarks with their wrist x-coordinate for sorting
                for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    wrist_x = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x
                    hand_landmarks_list.append((wrist_x, hand_landmarks))
                    # Draw landmarks on the display frame
                    mp_drawing.draw_landmarks(
                        display_frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )

                # Sort hands by their wrist's x-coordinate (left hand first, right hand second)
                hand_landmarks_list.sort(key=lambda x: x[0])
                sorted_hand_landmarks = [item[1] for item in hand_landmarks_list]

                num_detected_hands = len(sorted_hand_landmarks)

                # Data Saving Logic
                if collecting:
                    feature_vector = np.zeros(21 * 3 * 2) # Initialize with zeros for 2 hands (126 features)

                    if is_two_hand_gesture:
                        if num_detected_hands == 2:
                            # Concatenate landmarks from both hands
                            left_hand_features = extract_landmarks(sorted_hand_landmarks[0])
                            right_hand_features = extract_landmarks(sorted_hand_landmarks[1])
                            feature_vector = np.concatenate((left_hand_features, right_hand_features))
                        else:
                            # Display warning if not enough hands detected for a two-hand gesture
                            cv2.putText(display_frame, "Show BOTH hands!", (display_frame.shape[1] // 2 - 100, 50),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                            continue # Skip saving this frame if not enough hands
                    else: # Single hand gesture
                        if num_detected_hands >= 1:
                            # Use only the first detected hand (e.g., the dominant one)
                            feature_vector[:63] = extract_landmarks(sorted_hand_landmarks[0])
                        else:
                            # This case should ideally not happen if min_detection_confidence is good
                            continue # Skip if no hand detected for a single-hand gesture


                    # Save the landmark data
                    file_name = os.path.join(DATA_DIR, gesture_key, f'hand_{int(time.time() * 10000)}_{count}.npy')
                    np.save(file_name, feature_vector)
                    count += 1
                    cv2.putText(display_frame, f"Collecting: {gesture_display_name} ({count}/{NUM_SAMPLES})",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            else: # No hands detected
                if collecting:
                     cv2.putText(display_frame, "No hand(s) detected!", (display_frame.shape[1] // 2 - 100, 50),
                                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)


            # Display instructions/status
            if not collecting:
                 cv2.putText(display_frame, f"Gesture: {gesture_display_name}", (10, 30),
                             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                 cv2.putText(display_frame, "Press 's' to start collecting", (10, 70),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)


            cv2.imshow('Hand Gesture Data Collection', display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quitting data collection.")
                cap.release()
                cv2.destroyAllWindows()
                return # Exit the function completely

            if key == ord('s') and not collecting:
                print(f"Starting collection for '{gesture_display_name}'...")
                collecting = True

            if collecting and count >= NUM_SAMPLES:
                print(f"Finished collecting {NUM_SAMPLES} samples for '{gesture_display_name}'.")
                collecting = False
                # Add a small delay before moving to the next gesture to give user time to react
                time.sleep(2)
                break # Move to the next gesture

    cap.release()
    cv2.destroyAllWindows()
    print("\n--- Data Collection Complete for all gestures! ---")

if __name__ == "__main__":
    collect_gesture_data()