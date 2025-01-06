import pickle
import cv2
import mediapipe as mp
import numpy as np
import time

# Load the model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'K', 10: 'L', 11: 'M',
    12: 'N', 13: 'O', 14: 'P', 15: 'Q', 16: 'R', 17: 'S', 18: 'T', 19: 'U', 20: 'V', 21: 'W', 22: 'X', 23: 'Y'
}

flag = False
# Timer variables
last_prediction_time = time.time()
current_character = None
window_duration = 2  # 2 seconds
acceptance_duration = 0.5  # 0.5 seconds for "Accepted" display
accepted_until = 0  # Time until "Accepted" is displayed
no_hand_start_time = time.time()  # Time since no hand detected
detected_characters = []  # List for detected characters
recognition_active = True

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    if not ret:
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if recognition_active:
        results = hands.process(frame_rgb)
        current_time = time.time()

        if results.multi_hand_landmarks:
            # Reset timer if a hand is detected
            no_hand_start_time = current_time

            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,  # Image to draw on
                    hand_landmarks,  # Model output
                    mp_hands.HAND_CONNECTIONS,  # Hand connections
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            # Limit data_aux to 42 features
            max_features = 42

            if len(data_aux) > max_features:
                data_aux = data_aux[:max_features]
            elif len(data_aux) < max_features:
                data_aux.extend([0] * (max_features - len(data_aux)))

            # Make a prediction
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]

            # Check if the character has remained constant
            if predicted_character == current_character:
                if current_time - last_prediction_time >= window_duration:
                    # Add character to list if constant and time window is over
                    detected_characters.append(predicted_character)
                    accepted_until = current_time + acceptance_duration
                    print("Recognized character:", predicted_character)
                    current_character = None  # Reset current character
                    flag = True
            else:
                # Update time and character if it changes
                current_character = predicted_character
                last_prediction_time = current_time

            # Calculate bounding box for the hand
            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            # Display prediction on the image
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, f'Predicted: {predicted_character}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

            # Show "Accepted" if within time
            if current_time <= accepted_until:
                cv2.putText(frame, 'Accepted', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 0), 2, cv2.LINE_AA)

    # Display the detected characters
    word = ''.join(detected_characters)
    if word:
        cv2.putText(frame, f'Text: {word}', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
