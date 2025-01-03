import cv2
import time
import math as m
import mediapipe as mp
import board
import busio
import adafruit_drv2605




# Initialize I2C using default Jetson Orin pins (Pin 3 for SDA, Pin 5 for SCL)
i2c = busio.I2C(board.SCL, board.SDA)

# Initialize the DRV2605L
try:
    drv = adafruit_drv2605.DRV2605(i2c)
    print("DRV2605L successfully initialized.")
except Exception as e:
    print(f"Failed to initialize DRV2605L: {e}")
    exit()


# Function to play the vibration effect
def testing(trigger):
    """
    Play the vibration motor 3 times if trigger is 1.

    Parameters:
    trigger (int): If 1, play the vibration effect 3 times.
    """
    if trigger == 1:
        #print("Trigger received. Running vibration effect 3 times...")
       # for _ in range(3):
            drv.sequence[0] = adafruit_drv2605.Effect(1)  # Set effect (e.g., Strong Click 100%)
            drv.play()  # Start the effect
            #time.sleep(1)  # Let the effect play for its duration
           # drv.stop()  # Stop the effect
            #time.sleep(0.1)  # Wait 0.5 seconds before playing again
      #  print("Vibration effect played 3 times.")
   # else:
        #print("No trigger received. Exiting function.")

# Calculate distance
def findDistance(x1, y1, x2, y2):
    dist = m.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return dist


# Calculate angle.
def findAngle(x1, y1, x2, y2):
    theta = m.acos((y2 - y1) * (-y1) / (m.sqrt(
        (x2 - x1) ** 2 + (y2 - y1) ** 2) * y1))
    degree = int(180 / m.pi) * theta
    return degree


# Function to send alert. Use this function to send alert when bad posture detected.
#def sendWarning(x):
   # pass


# =============================CONSTANTS and INITIALIZATIONS=====================================#
good_frames = 0
bad_frames = 0
font = cv2.FONT_HERSHEY_SIMPLEX
blue = (255, 127, 0)
red = (50, 50, 255)
green = (127, 255, 0)
dark_blue = (127, 20, 0)
light_green = (127, 233, 100)
yellow = (0, 255, 255)
pink = (255, 0, 255)

# Initialize mediapipe pose class.
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# ===============================================================================================#

if __name__ == "__main__":
    # Initialize webcam (0 for default webcam).
    cap = cv2.VideoCapture(0)

    # Meta information.
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (width, height)

    while cap.isOpened():
        # Capture frames.
        success, image = cap.read()
        if not success:
            print("Please connect camera")
            break

        # Get height and width of the frame.
        h, w = image.shape[:2]

        # Convert the BGR image to RGB.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image with MediaPipe Pose.
        keypoints = pose.process(image)

        # Convert back to BGR.
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        lm = keypoints.pose_landmarks
        lmPose = mp_pose.PoseLandmark

        # Get landmark coordinates.
        l_shldr_x = int(lm.landmark[lmPose.LEFT_SHOULDER].x * w)
        l_shldr_y = int(lm.landmark[lmPose.LEFT_SHOULDER].y * h)
        r_shldr_x = int(lm.landmark[lmPose.RIGHT_SHOULDER].x * w)
        r_shldr_y = int(lm.landmark[lmPose.RIGHT_SHOULDER].y * h)
        l_ear_x = int(lm.landmark[lmPose.LEFT_EAR].x * w)
        l_ear_y = int(lm.landmark[lmPose.LEFT_EAR].y * h)
        l_hip_x = int(lm.landmark[lmPose.LEFT_HIP].x * w)
        l_hip_y = int(lm.landmark[lmPose.LEFT_HIP].y * h)

        # Calculate distance between left and right shoulder.
        offset = findDistance(l_shldr_x, l_shldr_y, r_shldr_x, r_shldr_y)

        # Check camera alignment.
        if offset < 100:
            cv2.putText(image, str(int(offset)) + ' Aligned', (w - 150, 30), font, 0.9, green, 2)
        else:
            cv2.putText(image, str(int(offset)) + ' Not Aligned', (w - 150, 30), font, 0.9, red, 2)

        # Calculate angles for neck and torso inclinations.
        neck_inclination = findAngle(l_shldr_x, l_shldr_y, l_ear_x, l_ear_y)
        torso_inclination = findAngle(l_hip_x, l_hip_y, l_shldr_x, l_shldr_y)

        # Draw landmarks.
        cv2.circle(image, (l_shldr_x, l_shldr_y), 7, yellow, -1)
        cv2.circle(image, (l_ear_x, l_ear_y), 7, yellow, -1)
        cv2.circle(image, (l_shldr_x, l_shldr_y - 100), 7, yellow, -1)
        cv2.circle(image, (r_shldr_x, r_shldr_y), 7, pink, -1)
        cv2.circle(image, (l_hip_x, l_hip_y), 7, yellow, -1)
        cv2.circle(image, (l_hip_x, l_hip_y - 100), 7, yellow, -1)

        # Display angle and determine posture.
        angle_text_string = 'Neck : ' + str(int(neck_inclination)) + '  Torso : ' + str(int(torso_inclination))
        if 35 < neck_inclination < 50 and torso_inclination < 10:

            bad_frames = 0
            good_frames += 1
            cv2.putText(image, angle_text_string, (10, 30), font, 0.9, light_green, 2)
            cv2.putText(image, str(int(neck_inclination)), (l_shldr_x + 10, l_shldr_y), font, 0.9, light_green, 2)
            cv2.putText(image, str(int(torso_inclination)), (l_hip_x + 10, l_hip_y), font, 0.9, light_green, 2)
            cv2.line(image, (l_shldr_x, l_shldr_y), (l_ear_x, l_ear_y), green, 4)
            cv2.line(image, (l_shldr_x, l_shldr_y), (l_shldr_x, l_shldr_y - 100), green, 4)
            cv2.line(image, (l_hip_x, l_hip_y), (l_shldr_x, l_shldr_y), green, 4)
            cv2.line(image, (l_hip_x, l_hip_y), (l_hip_x, l_hip_y - 100), green, 4)
        else:
            good_frames = 0
            bad_frames += 1
            cv2.putText(image, angle_text_string, (10, 30), font, 0.9, red, 2)
            cv2.putText(image, str(int(neck_inclination)), (l_shldr_x + 10, l_shldr_y), font, 0.9, red, 2)
            cv2.putText(image, str(int(torso_inclination)), (l_hip_x + 10, l_hip_y), font, 0.9, red, 2)
            cv2.line(image, (l_shldr_x, l_shldr_y), (l_ear_x, l_ear_y), red, 4)
            cv2.line(image, (l_shldr_x, l_shldr_y), (l_shldr_x, l_shldr_y - 100), red, 4)
            cv2.line(image, (l_hip_x, l_hip_y), (l_shldr_x, l_shldr_y), red, 4)
            cv2.line(image, (l_hip_x, l_hip_y), (l_hip_x, l_hip_y - 100), red, 4)

        # Calculate the time of remaining in a particular posture.
        good_time = (1 / fps) * good_frames
        bad_time = (1 / fps) * bad_frames

        # Pose time.
        if good_time > 0:
            time_string_good = 'Good Posture Time : ' + str(round(good_time, 1)) + 's'
            cv2.putText(image, time_string_good, (10, h - 20), font, 0.9, green, 2)
        else:
            time_string_bad = 'Bad Posture Time : ' + str(round(bad_time, 1)) + 's'
            cv2.putText(image, time_string_bad, (10, h - 20), font, 0.9, red, 2)

        # If bad posture is detected for over 5 seconds, send an alert.
        if bad_time > 5:
            testing(1)



        # Display the frame.
        cv2.imshow('MediaPipe Pose', image)

        # Break the loop on pressing 'q'.
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
