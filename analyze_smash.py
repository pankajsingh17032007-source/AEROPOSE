import cv2
import mediapipe as mp
import mediapipe.python.solutions.drawing_utils as mp_drawing
import mediapipe.python.solutions.pose as mp_pose
import numpy as np

def calculate_angle(a, b, c):
    """Calculates the 2D angle between three joints with point 'b' as vertex."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360.0 - angle
        
    return round(angle, 1)

video_path = "smash1.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open {video_path}. Make sure it is inside the smash-ai folder.")
    exit()

# Extract video properties for the writer
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0 or np.isnan(fps):
    fps = 30.0

# Initialize VideoWriter using the cross-platform mp4v codec
output_filename = "annotated_smash1.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))

# Configure display window
window_name = "Badminton Smash Analysis"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 720, 960)

highest_wrist_y = 1.0
impact_frame = None
best_elbow_angle = 0.0
frame_count = 0

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        h, w, _ = frame.shape
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = pose.process(rgb_frame)
        frame.flags.writeable = True

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w,
                        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h]
            elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * w,
                     landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * h]
            wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x * w,
                     landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * h]
            
            elbow_angle = calculate_angle(shoulder, elbow, wrist)
            
            wrist_y = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y
            if wrist_y < highest_wrist_y:
                highest_wrist_y = wrist_y
                impact_frame = frame.copy()
                best_elbow_angle = elbow_angle

            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            cv2.putText(frame, f"{elbow_angle} deg", 
                        (int(elbow[0]) + 10, int(elbow[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Write the annotated frame to the output video file
        out.write(frame)

        cv2.imshow(window_name, frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

cap.release()
out.release()
cv2.destroyAllWindows()

# --- Post-Analysis Report ---
print("\n" + "="*40)
print("         SMASH ANALYSIS REPORT          ")
print("="*40)
print(f"Total frames processed: {frame_count}")
print(f"Arm extension at impact: {best_elbow_angle}°")

if best_elbow_angle >= 165:
    score = 100.0
    feedback = "Textbook smash! Arm was fully extended at impact."
elif best_elbow_angle >= 140:
    score = round(max(0.0, 100.0 - (165.0 - best_elbow_angle) * 2.5), 1)
    feedback = "Good shot, but reach higher for a steeper hit."
else:
    score = round(max(0.0, 50.0 - (140.0 - best_elbow_angle)), 1)
    feedback = "Arm was bent excessively. Hit at peak overhead reach."

print(f"Perfection Score: {score}%")
print(f"Feedback: {feedback}")
print("="*40)
print(f"Annotated video saved successfully as: {output_filename}")

if impact_frame is not None:
    cv2.imwrite("impact_snapshot.jpg", impact_frame)
    print("Snapshot saved as 'impact_snapshot.jpg'.")