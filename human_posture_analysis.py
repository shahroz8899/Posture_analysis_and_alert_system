import cv2
import time
import math as m
import mediapipe as mp
import board
import busio
import adafruit_drv2605
import numpy as np
import pyautogui

# Initialize I2C using default Jetson Orin pins (Pin 3 for SDA, Pin 5 for SCL)
i2c = busio.I2C(board.SCL, board.SDA)

# Initialize the DRV2605L
try:
    drv = adafruit_drv2605.DRV2605(i2c)
    print("DRV2605L successfully initialized.")
except Exception as e:
    print(f"Failed to initialize DRV2605L: {e}")
    exit()

def testing(trigger):
    if trigger == 1:
        drv.sequence[0] = adafruit_drv2605.Effect(1)
        drv.play()

def findDistance(x1, y1, x2, y2):
    return m.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def findAngle(x1, y1, x2, y2):
    try:
        a = [x1, y1]
        b = [x2, y2]
        vertical = [x1, y1 - 100]
        ab = [b[0] - a[0], b[1] - a[1]]
        av = [vertical[0] - a[0], vertical[1] - a[1]]
        dot = ab[0] * av[0] + ab[1] * av[1]
        mag_ab = m.sqrt(ab[0]**2 + ab[1]**2)
        mag_av = m.sqrt(av[0]**2 + av[1]**2)
        angle_rad = m.acos(dot / (mag_ab * mag_av))
        return int(m.degrees(angle_rad))
    except:
        return 0

font = cv2.FONT_HERSHEY_SIMPLEX
colors = {
    "blue": (255, 127, 0),
    "red": (50, 50, 255),
    "green": (127, 255, 0),
    "light_green": (127, 233, 100),
    "yellow": (0, 255, 255),
    "pink": (255, 0, 255)
}

mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(static_image_mode=False, model_complexity=1)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def get_camera():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1000)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)
    return cap

cap = get_camera()
cv2.namedWindow("MediaPipe Pose", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("MediaPipe Pose", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

bad_frames = 0
good_frames = 0
screen_width, screen_height = pyautogui.size()

while True:
    success, image = cap.read()
    if not success or image is None:
        cap.release()
        while True:
            cap = get_camera()
            success, image = cap.read()
            if success and image is not None:
                break
            blank_img = 255 * np.ones((screen_height, screen_width, 3), dtype=np.uint8)
            cv2.putText(blank_img, "Camera not connected", (200, 240), font, 1, colors["red"], 2)
            cv2.imshow("MediaPipe Pose", blank_img)
            if cv2.waitKey(1000) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                exit()
        continue

    image = cv2.resize(image, (screen_width, screen_height))
    h, w = image.shape[:2]
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = holistic.process(image_rgb)

    lm = results.pose_landmarks
    face_lm = results.face_landmarks

    is_valid = True
    visible_landmarks = []

    if lm:
        visible_landmarks += [l for l in lm.landmark if l.visibility >= 0.9]

    if face_lm:
        visible_landmarks += face_lm.landmark[:3]  # Assume first 20 face landmarks are enough to count

    if len(visible_landmarks) < 17:
        is_valid = False

    if not is_valid:
        cv2.putText(image, "Full human not detected", (10, 30), font, 1, colors["red"], 2)
        cv2.imshow('MediaPipe Pose', image)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
        continue

    mp_drawing.draw_landmarks(image, lm, mp_holistic.POSE_CONNECTIONS)

    try:
        l_shldr_x = int(lm.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].x * w)
        l_shldr_y = int(lm.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].y * h)
        r_shldr_x = int(lm.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER].x * w)
        r_shldr_y = int(lm.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER].y * h)
        l_ear_x = int(lm.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x * w)
        l_ear_y = int(lm.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y * h)
        l_hip_x = int(lm.landmark[mp_holistic.PoseLandmark.LEFT_HIP].x * w)
        l_hip_y = int(lm.landmark[mp_holistic.PoseLandmark.LEFT_HIP].y * h)
        l_knee_x = int(lm.landmark[mp_holistic.PoseLandmark.LEFT_KNEE].x * w)
        l_knee_y = int(lm.landmark[mp_holistic.PoseLandmark.LEFT_KNEE].y * h)
    except:
        continue

    hip_knee_angle = findAngle(l_hip_x, l_hip_y, l_knee_x, l_knee_y)
    cv2.putText(image, f"Hip-Knee: {hip_knee_angle}°", (l_hip_x + 10, l_hip_y), font, 0.5, colors["green"], 2)
    if 170 <= hip_knee_angle <= 190:
        cv2.putText(image, "Please sit on a chair for posture analysis", (10, 30), font, 1, colors["red"], 2)
        cv2.imshow('MediaPipe Pose', image)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
        continue

    offset = findDistance(l_shldr_x, l_shldr_y, r_shldr_x, r_shldr_y)
    if offset < 100:
        cv2.putText(image, str(int(offset)) + ' Aligned', (w - 150, 30), font, 0.9, colors["green"], 2)
    else:
        cv2.putText(image, str(int(offset)) + ' Not Aligned', (w - 150, 30), font, 0.9, colors["red"], 2)

    neck_inclination = findAngle(l_shldr_x, l_shldr_y, l_ear_x, l_ear_y)
    body_inclination = findAngle(l_hip_x, l_hip_y, l_shldr_x, l_shldr_y)
    angle_text = f'Neck: {neck_inclination}°  Body: {body_inclination}°'

    if 10 < neck_inclination < 50 and body_inclination < 20:
        bad_frames = 0
        good_frames += 1
        color = colors["light_green"]
    else:
        good_frames = 0
        bad_frames += 1
        color = colors["red"]

    cv2.putText(image, angle_text, (10, 30), font, 0.9, color, 2)
    cv2.putText(image, str(neck_inclination), (l_shldr_x + 10, l_shldr_y), font, 0.9, color, 2)
    cv2.putText(image, str(body_inclination), (l_hip_x + 10, l_hip_y), font, 0.9, color, 2)
    cv2.line(image, (l_shldr_x, l_shldr_y), (l_ear_x, l_ear_y), color, 4)
    cv2.line(image, (l_shldr_x, l_shldr_y), (l_shldr_x, l_shldr_y - 100), color, 4)
    cv2.line(image, (l_hip_x, l_hip_y), (l_shldr_x, l_shldr_y), color, 4)
    cv2.line(image, (l_hip_x, l_hip_y), (l_hip_x, l_hip_y - 100), color, 4)

    good_time = (1 / 30.0) * good_frames
    bad_time = (1 / 30.0) * bad_frames

    if good_time > 0:
        cv2.putText(image, f'Good Posture Time: {round(good_time, 1)}s', (10, h - 20), font, 0.9, colors["green"], 2)
    else:
        cv2.putText(image, f'Bad Posture Time: {round(bad_time, 1)}s', (10, h - 20), font, 0.9, colors["red"], 2)

    if bad_time > 5:
        testing(1)

    if face_lm:
        mp_drawing.draw_landmarks(
            image,
            face_lm,
            mp_holistic.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_tesselation_style()
        )

    cv2.imshow('MediaPipe Pose', image)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

