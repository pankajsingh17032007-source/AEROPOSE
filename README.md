 AEROPOSE - Badminton Smash AI Analyzer

A computer vision tool built with OpenCV and MediaPipe Pose to analyze badminton smash biomechanics, measure overhead arm extension angles, identify peak contact height, and calculate a technique perfection score.

Features:-
- Human pose estimation with MediaPipe Pose.
- Biomechanical joint angle calculation (Shoulder-Elbow-Wrist).
- Automated impact frame detection based on peak vertical wrist reach.
- Angle feedback and scoring rubric based on elbow extension.
- Annotated video export with on-screen angle displays.

Setup & Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/pankajsingh17032007-source/AEROPOSE.git](https://github.com/pankajsingh17032007-source/AEROPOSE.git)
   cd AEROPOSE
   ```

2. Create and activate virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install opencv-python mediapipe numpy
   ```

#Usage

1. Webcam Test:
   ```bash
   python test_webcam.py
   ```

2. Run Smash Analysis:
   Place your video inside the folder as `smash.mp4` and run:
   ```bash
   python analyze_smash.py
   ```
