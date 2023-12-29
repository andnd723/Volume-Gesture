import cv2
import numpy as np
import time
import mediapipe as mp
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# MediaPipe Setup, setting the utilities for drawing landmarks and connections on the image with
# 'mp_hands' as the module for hand tracking
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Initializing the volume control using pycaw. Get the volume range and set initial values for
# volume, volume bar, and volume percentage.

# Return a collection of audio playback devices (speakers)
devices = AudioUtilities.GetSpeakers()
# Activate the volume control interface for the speaker and return an interface to control volume
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
# Cast the interface to control the volume
volume = cast(interface, POINTER(IAudioEndpointVolume))
# Retrieve the range of volume of the speaker and return a tuple of min and max volume values
volRange = volume.GetVolumeRange()
# Initialize the volume values for control parameters
minVol , maxVol , volBar, volPer = volRange[0] , volRange[1], 400, 0

# Camera set up using OpenCV
wCam, hCam = 1920, 1080
cam = cv2.VideoCapture(0)
cam.set(3, wCam)
cam.set(4, hCam)
pTime = 0

# Mediapipe Hand Landmark Model that sets up specific configurations for model complexity,
# minimum detection confidence, and minimum tracking confidence
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

# FPS Counter
  while cam.isOpened():
    success, image = cam.read()

    # Add a check for successful frame capture
    if not success:
      continue

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(image, f'FPS: {int(fps)}', (60, 120), cv2.FONT_HERSHEY_COMPLEX, 3, (255,0,255), 3)
    cv2.waitKey(1)    


    # Convert image to RGB format and process it using the MediaPipe Hand module, and convert
    # back to BGR format

    # Convert the color space of an image using the OpenCV library from the img (frame of camera),
    # to RGB format
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Process the RGB image using the hand tracking model, should contain information on
    # the position and detected hand landmarks
    results = hands.process(image)
    # After processing the image, convert back to BGR so that OpenCV can process it
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Draw hand landmarks, if hand landmarks are detected, draw them on the image

    # Parse through the list of detected hand landmarks
    if results.multi_hand_landmarks:
      # Iterate through each set of hand landmarks if there is any
      for hand_landmarks in results.multi_hand_landmarks:
        # Use MediaPipe to draw landmarks on the image with all those parameters
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
            )

    # Extract and store the positions of hand landmarks in a list    
    lmList = []
    # Check if hand landmarks are detected in the frame
    if results.multi_hand_landmarks:
      # If detected, get the first detected hand assuming there is one hand in the frame
      myHand = results.multi_hand_landmarks[0]
      # Iterate through each landmark of the hand to get both the index ('id') and the landmark
      for id, lm in enumerate(myHand.landmark):
        # Return height, width, and number of channels o fthe image
        h, w, c = image.shape
        # Convert the normalized coordinates to pixel coordinates using the image dimensions
        cx, cy = int(lm.x * w), int(lm.y * h)
        # Append a list containing the landmark index and its pixel coordinates
        lmList.append([id, cx, cy])          

    # Assigning variables for Thumb and Index finger position. Analyze hand gestures
    # and calculate the length between thumb and index finger, and mark them on the image
    if len(lmList) != 0:
      # If the list is not empty, assign the pixel coordinates of the thumb and index finger
      # variables x1, y1 and x2, y2. Index them from the tip of the thumb (4) and tip of
      # the index finger (8)
      x1, y1 = lmList[4][1], lmList[4][2]
      x2, y2 = lmList[8][1], lmList[8][2]

      # Marking Thumb and Index finger by drawing circles around the finger tips
      cv2.circle(image, (x1,y1),15,(255,255,255))  
      cv2.circle(image, (x2,y2),15,(255,255,255))
      # Draw a line starting from the tip of the thumb to the index finger   
      cv2.line(image,(x1,y1),(x2,y2),(0,255,0),3)
      # Calculate the Euclidean distance between two points, representing the length between
      # the fingers
      length = math.hypot(x2-x1,y2-y1)
      # If the length is less than 50 pixels, draw an additional line between the thumb
      # and index finger tips
      if length < 50:
        cv2.line(image,(x1,y1),(x2,y2),(0,0,255),3)

      # Interpolate the hand gesture length to control the volume, update volume, volume bar
      # position, and volume percentage

      # Use NumPy function for linear interpolation to map the length of minimum 50 to maximum
      # 220 pixels
      vol = np.interp(length, [50, 220], [minVol, maxVol])
      # Apply the hand gesture length to the system's audio volume
      volume.SetMasterVolumeLevel(vol, None)
      # Maps the length of the hand gesture to a position on the volume bar
      volBar = np.interp(length, [50, 220], [400, 150])
      # Maps the length of the hand gesture to a percentage value
      volPer = np.interp(length, [50, 220], [0, 100])

      # Display the Volume Bar

      # Draws the outline of the volume bar
      cv2.rectangle(image, (50, 150), (85, 400), (0, 0, 0), 3)
      # Fill the volume bar
      cv2.rectangle(image, (50, int(volBar)), (85, 400), (0, 0, 0), cv2.FILLED)
      # Display the volume percentage as a text on the screen
      cv2.putText(image, f'{int(volPer)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX,
                1, (0, 0, 0), 3)
    
    # Display the image and quit when pressing 'q'
    cv2.imshow('handDetector', image) 
    # Check for the 'q' input and break if someone quits
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

cam.release()