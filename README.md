# Posture Analysis System Using MediaPipe Pose

This project implements a posture analysis system using **MediaPipe Pose**, **OpenCV**, and the **DRV2605L Haptic Motor Controller**. It detects body posture through a webcam and gives visual feedback on good and bad posture. A haptic motor provides tactile feedback for bad posture detected over time.

## Features
- Detects shoulder, ear, and hip landmarks using MediaPipe Pose.
- Analyzes neck and torso inclination to classify posture as good or bad.
- Provides visual feedback via color-coded annotations on the live video feed.
- Delivers tactile feedback via the DRV2605L haptic motor when bad posture is maintained for over 5 seconds.

## Requirements
This project is compatible with the following:
- **Python 3.x**
- A webcam for posture detection
- NVIDIA Jetson Orin (or compatible hardware with I2C support)
- DRV2605L Haptic Motor Controller

## Installation

### Clone the Repository
```bash
git clone https://github.com/<your-github-username>/posture-analysis-system.git
cd posture-analysis-system

