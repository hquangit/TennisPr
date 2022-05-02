
import cv2
from cv2 import norm
import numpy as np
import mediapipe as mp
import pandas as pd
import tensorflow as tf
import threading
from tkinter import *
from PIL import ImageTk, Image

import matplotlib.pyplot as plt
# import numpy as np
from matplotlib.patches import Circle
import matplotlib.cbook as cbook
import os
import math
import time
import random

def make_landmark_timestep(results):
    c_lm = []
    for id, lm in enumerate(results.pose_landmarks.landmark):
        c_lm.append(lm.x)
        c_lm.append(lm.y)
        c_lm.append(lm.z)
        c_lm.append(lm.visibility)
    return c_lm

def draw_landmark_on_image(mpDraw, results, img):
    
    mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

    cx_head = 0
    cy_head = 0

    foot_1 = [0,0]
    foot_2 = [0,0]

    foot_1_cz = 0
    foot_2_cz = 0

    vis_foot1 = 0
    vis_foot2 = 0

    vis_head = 0

    for id, lm in enumerate(results.pose_landmarks.landmark):
        h, w, c = img.shape
        print(id, lm)
        cx, cy, cz = int(lm.x * w), int(lm.y * h), float(abs(lm.z * w))
        # z_ratio = float(lm.z)
        if id == 0:
            cx_head = cx
            cy_head = cy
            vis_head = float(lm.visibility)

        if id ==29:
            foot_1[0] = cx
            foot_1[1] = cy
            foot_1_cz = cz
            vis_foot1 = float(lm.visibility)

        if id == 30:
            foot_2[0] = cx
            foot_2[1] = cy
            foot_2_cz = cz
            vis_foot2 = float(lm.visibility)
        cv2.circle(img, (cx, cy), 3, (0, 0, 255), cv2.FILLED)

    z_ratio = (foot_1_cz+foot_2_cz)/(2*w)
    return img, cx_head, cy_head, foot_1, foot_2, z_ratio, vis_foot1, vis_foot2, vis_head

def draw_class_on_image(label, img, coordinates):
    font = cv2.FONT_HERSHEY_SIMPLEX
    # bottomLeftCornerOfText = (10, 30)
    bottomLeftCornerOfText = (10, 30)
    fontScale = 1
    fontColor = (0, 255, 0)
    thickness = 2
    lineType = 2
    cv2.putText(img, label,
                coordinates,
                # bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                thickness,
                lineType)
    return img

def draw_logo(label, img, coordinates):
    font = cv2.FONT_HERSHEY_SIMPLEX
    # bottomLeftCornerOfText = (10, 30)
    bottomLeftCornerOfText = (10, 30)
    fontScale = 1
    fontColor = (0, 9, 255)
    thickness = 2
    lineType = 2
    cv2.putText(img, label,
                coordinates,
                # bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                thickness,
                lineType)
    return img

def detect(model, lm_list):
    global label
    lm_list = np.array(lm_list)
    lm_list = np.expand_dims(lm_list, axis=0)
    print(lm_list.shape)
    results = model.predict(lm_list)
    # print(results)
    score = np.max(results)
    classes = np.argmax(results, axis = 1)
    print("score========================================.: ", score)
    if score <=0.9:
        label = "..."
    else:
        if classes == 0: 
            label = "SERVE"
        elif classes == 1:
            label = "BACK_HAND"
        elif classes == 2:
            label = "FORE_HAND"
        return label

def distance(coordinates_1, coordinates_2):
    x = coordinates_2[0] - coordinates_1[0]
    # y = coordinates_2[1] - coordinates_1[1]
    y = coordinates_1[1] - coordinates_2[1]
    coordinates = (x,y)
    return coordinates

def gen_distance(point1, point2, ):
    u1, u2, u3, u4 = (21,1024), (574, 701), (1837, 1024), (1315, 701)
    v1, v2, v3, v4 = (90, 769), (90, 413), (357, 769), (357, 413)
    u_center_1 = (933, 1017)
    u_center_2 = (933, 695)

    v_center_1 = (225, 770)
    v_center_2 = (225, 412)

    ratio_x = float(4.11/point_to_point(v1,v3))
    ratio_y = float((23.78/2)/point_to_point(v_center_1, v_center_2))


    dis_1 = float(point_to_point(point1,(point2[0], point1[1])) * ratio_y)
    dis_2 = float(point_to_point(point2,(point2[0], point1[1])) * ratio_x)

    # result = 

    return float(math.sqrt(pow(dis_1,2) + pow(dis_2,2)))

def read_video():
    lm_list = []
    # # label = "Warmup...."
    n_time_steps = 10

    # mpPose = mp.solutions.pose
    # pose = mpPose.Pose()
    # mpDraw = mp.solutions.drawing_utils

    model = tf.keras.models.load_model("./model/model-v2/model.h5")
    
    cap = cv2.VideoCapture("./video/video-v3/don.mp4")

    image_course = cv2.imread("./course_image/course.PNG")
    w,h = image_course.shape[0], image_course.shape[1]
    print("-----------------------------------------------------------------------")

    print("Width: ", w)
    print("Height: ", h)
    coo_before = []
    color_before = []


    u1, u2, u3, u4 = (21,1024), (574, 701), (1837, 1024), (1315, 701)
    v1, v2, v3, v4 = (90, 769), (90, 413), (357, 769), (357, 413)

    u_center_1 = (933, 1017)
    u_center_2 = (933, 695)

    left_boder_line= (u2, u1)
    center_line = (u_center_1,u_center_2)
    boder_down_center = ((v1[0]+v3[0])/2, (v1[1]+v3[1])/2)
    half_boder_down_line = point_to_point(v1, boder_down_center)
    print("half_boder_down_line: ",half_boder_down_line)

    centerPoint_to_downLine = point_to_point((937,803), boder_down_center)

    count = 0
    coordinates_start=[]
    time_start = time.time()

    text_speed = ""
    while(cap.isOpened()):
        success, img = cap.read()
        
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(imgRGB)
        coordinates=(0,0)
        people = (0,0)
        # coordinates_root = (91,861)
        coordinates_root = (975,864)
        
        if success == True:
            
            print("Start detect....")
            
            if results.pose_landmarks:
                c_lm = make_landmark_timestep(results)

                lm_list.append(c_lm)
                if len(lm_list) == n_time_steps:
                    # predict
                    t1 = threading.Thread(target=detect, args=(model, lm_list,))               
                    t1.start()
                
                    lm_list = []

                img, cx, cy, foot_1, foot_2, z_rat, vis1, vis2, vis_head = draw_landmark_on_image(mpDraw, results, img)
                coordinates_people = ((foot_1[0]+foot_2[0])/2, (foot_1[1]+foot_2[1])/2)

                # speed = 0
                next_time = time.time()
                if(len(coordinates_start) == 0):
                    # coordinates_start.append(coordinates_people[0])
                    # coordinates_start.append(coordinates_people[1])
                    coordinates_start.append(cx)
                    coordinates_start.append(cy)
                elif(next_time-time_start >=0.5):
                    print("Da vao next step")
                    point_tmp = (coordinates_start[0], coordinates_start[1])
                    dis = gen_distance(point_tmp, coordinates_people)
                    speed = int(dis/(next_time-time_start))
                    coordinates_start.clear()
                    # coordinates_start.append(coordinates_people[0])
                    # coordinates_start.append(coordinates_people[1])
                    coordinates_start.append(cx)
                    coordinates_start.append(cy)
                    time_start = next_time

                    text_speed = "speed: {:.1f}".format(speed)
                    # text_speed = "Da vao next step"
                    # img = draw_class_on_image(text_speed, img, (coordinates[0]-35, coordinates[1]-35))


                x_rat, x_label = x_ratio(coordinates_people, left_boder_line, center_line)

                print("--------------------------------")
                print("X-rate: ", x_rat)
                if(x_label == "left"):
                    x_tmp = int((1-x_rat) * half_boder_down_line)+ 90
                else:
                    x_tmp = int((1+x_rat) * half_boder_down_line) +90 

                y_rat = y_ratio(coordinates_people, (u1,u3))
                y_tmp = int(824-y_rat*(824/4)) -30

                print("Z ratio: ", z_rat)

                # t1, t2 = coordinates_people[0], coordinates_people[1]
                people = distance(coordinates_root,coordinates_people)

                coordinates = (cx, cy)

            # text_coordinates = "[" + str(people[0]) + "," + str(people[1]) + "]"
            # img = draw_class_on_image(text_coordinates, img, (coordinates[0]-35, coordinates[1]-35))
            img = draw_class_on_image("Speed: ", img, (coordinates[0]-35, coordinates[1]-35))
            if(vis_head>=0.8):
                img = draw_class_on_image(text_speed, img, (coordinates[0]-35, coordinates[1]-35))
            # img = draw_class_on_image(text_speed, img, (coordinates[0]-35, coordinates[1]-35))
            img = draw_class_on_image(label, img, coordinates)
            img = draw_logo("RIKAI - Confidential", img, (1500,50))  
    

            cv2.imshow("Image", img)

            if(vis_head >=0.8):
                if len(coo_before)>0 and len(color_before)>0:
                    cv2.circle(image_course, (coo_before[0], coo_before[1]), 5, (color_before[0], color_before[1], color_before[2]), cv2.FILLED)
                
                coo_before.clear()
                color_before.clear()
                print("X-tmp: ", x_tmp)
                print("Y-tmp: ", y_tmp)
                coo_before.append(x_tmp)
                coo_before.append(y_tmp)
                color_before.append(float(image_course[y_tmp-1, x_tmp-1, 0]))
                color_before.append(float(image_course[y_tmp-1, x_tmp-1, 1]))
                color_before.append(float(image_course[y_tmp-1, x_tmp-1, 2]))

                cv2.circle(image_course, (x_tmp, y_tmp), 5, (255, 0, 0), cv2.FILLED)
                cv2.imshow("Course", image_course)


            if cv2.waitKey(1) == ord('q'):
                break
        else:
            break

    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()       


# coo.append((coordinates_people, x_rat, z_rat, label))

def create_circle(x, y, r): #center coordinates, radius
    x0 = x - r
    y0 = y - r
    x1 = x + r  
    y1 = y + r
    # return canvas.create_oval(x0, y0, x1, y1)
    return (x0, y0, x1, y1)



def line_intersection(line1, line2):
    print("line 1: ",line1)
    print("line 2", line2)
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return (x, y)


def point_to_line(poin, line):
    p1 = np.asarray(line[0])
    p2 = np.asarray(line[1])
    p3 = np.asarray(poin)
    d = norm(np.cross(p2-p1, p1-p3))/norm(p2-p1)
    return d

def x_ratio(point, left_boder_line, center_line):
    d1 = point_to_line(point, center_line)
    tmp = (center_line[0][0], point[1])
    print("boder line: ", left_boder_line)
    print("center line: ", center_line)
    print("point: ", point)
    print("tmp: ", tmp)
    tmp2 = line_intersection(left_boder_line, (point, tmp))
    print("tmp point: ", tmp2)
    d2 = point_to_line(tmp2, center_line)

    label = ""
    if point[0] < center_line[0][0]:
        x_label = "left"
    else:
        x_label = "right"

    print("label: ", x_label)
    return d1/d2, x_label

def y_ratio(point, down_boder_line):
    d = point_to_line(point, down_boder_line)
    d2 = point_to_line((937,803), down_boder_line)
    return d/d2

def point_to_point(point1, point2):
    return math.hypot(point2[0]-point1[0], point2[1]-point1[1])

if __name__ == "__main__":
    global label
    # lm_list = []
    label = "Warmup...."
    # n_time_steps = 10

    mpPose = mp.solutions.pose
    pose = mpPose.Pose()
    mpDraw = mp.solutions.drawing_utils
    read_video()


