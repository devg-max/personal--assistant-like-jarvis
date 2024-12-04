import cv2
import mediapipe as mp
import numpy as np
import pyttsx3

# Initialize the MediaPipe Hands class
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils  # For drawing hand landmarks on the image

# Initialize OpenCV for video capture
cap = cv2.VideoCapture(0)

# Initialize text-to-speech engine (optional)
engine = pyttsx3.init()

# Function to determine if hand is raised (gesture to say "Hi")
def is_hand_raised(landmarks, height, width):
    # Check if the y-coordinate of the tip of the middle finger is higher than a threshold (e.g., above the middle of the image)
    middle_finger_tip_y = landmarks[9].y * height
    palm_base_y = landmarks[0].y * height  # Palm base (wrist) position

    # A simple heuristic: If the middle finger tip is significantly above the wrist
    if middle_finger_tip_y < palm_base_y - height * 0.2:  # Check if hand is raised
        return True
    return False

# Main loop for processing video frames
while True:
    success, img = cap.read()  # Capture frame-by-frame from the webcam
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert the image to RGB format (required by MediaPipe)
    
    # Process the frame to detect hands
    result = hands.process(img_rgb)

    height, width, _ = img.shape  # Get the dimensions of the video frame
    
    # Check if hand landmarks are detected
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw landmarks on the hand
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Convert the landmarks to pixel values
            landmarks = hand_landmarks.landmark

            # Check if the hand is raised
            if is_hand_raised(landmarks, height, width):
                # Display "Hi" on the screen
                cv2.putText(img, "Hi!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
                
                # Optionally say "Hi" using text-to-speech
                engine.say("Hi")
                engine.runAndWait()

    # Display the result
    cv2.imshow('Hand Gesture Recognition', img)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
