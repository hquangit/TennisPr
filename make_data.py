from unittest import result
import cv2
import numpy as np
import mediapipe as mp
import pandas as pd


def make_landmark_timestep(resukts):
    c_lm = []
    for id, lm in enumerate(results.pose_landmarks.landmark):
        c_lm.append(lm.x)
        c_lm.append(lm.y)
        c_lm.append(lm.z)
        c_lm.append(lm.visibility)
    return c_lm

def draw_landmark_on_image(mpDraw, results, img):
    
    mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

    for id, lm in enumerate(results.pose_landmarks.landmark):
        h, w, c = img.shape
        print(id, lm)
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(img, (cx, cy), 3, (0, 0, 255), cv2.FILLED)
    return img 

lm_list = []
label = "SERVE"
no_of_frames = 600

cap = cv2.VideoCapture("./video/video-v2/serve.mp4")

# mediapipe library
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")

# Read until video is completed
while(cap.isOpened() and len(lm_list) <= no_of_frames): #  and len(lm_list) <= no_of_frames
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
        # frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame)

        if results.pose_landmarks:
            # identufy pose
            lm = make_landmark_timestep(results)
            lm_list.append(lm)

            # draw pose
            frame = draw_landmark_on_image(mpDraw, results, frame) 

        # Display the resulting frame
        cv2.imshow('Frame',frame)

        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Break the loop
    else: 
        break
# print("len: ", len(lm_list))
df = pd.DataFrame(lm_list)
df.to_csv("./data/data-v2/" + label +".txt")

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()