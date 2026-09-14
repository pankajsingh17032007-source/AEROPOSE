import cv2
import mediapipe as mp
import mediapipe.python.solutions.drawing_utils as mp_drawing
import mediapipe.python.solutions.pose as mp_pose

# Connect to your Mac's default webcam
cap = cv2.VideoCapture(0,cv2.CAP_AVFOUNDATION)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        # Convert image to RGB for MediaPipe processing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False

        # Extract pose landmarks
        results = pose.process(frame_rgb)

        # Draw the skeleton lines over the original image
        frame.flags.writeable = True
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, 
                results.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS
            )

        # Display window
        cv2.imshow("Badminton Pose Engine - Test", frame)

        # Press 'q' on your keyboard to exit
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()