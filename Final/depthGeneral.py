import cv2
import pyrealsense2
import numpy as np
import math
import os
import sys
import time
import torch
from realsense_depth import *

point = (400, 300)

def show_distance(event, x, y, args, params):
    global point
    point = (x, y)

# Initialize Camera Intel Realsense
dc = DepthCamera()

# Target object selection
SELECTED_OBJECT = 'bottle' #bottle or weed

if SELECTED_OBJECT == 'bottle':
    model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
    time.sleep(5)  
    print('Model has been downloaded and created') 
else:
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='path/to/old-best.pt')
    time.sleep(5)  
    print('Model has been downloaded and created') 

# Create mouse event
cv2.namedWindow("Color frame")
cv2.setMouseCallback("Color frame", show_distance)

while True:
    ret, color_frame= dc.get_frame()

    result = model(color_frame)
    objs = result.pandas().xyxy[0]
    objs_name = objs.loc[objs['name'] == SELECTED_OBJECT]
    
    height = color_frame.shape[0]
    width = color_frame.shape[1]
    
    x_middle = 0
    y_middle = 0
    try:
        obj = objs_name.iloc[0]
        x_middle = obj.xmin + (obj.xmax-obj.xmin)/2
        y_middle = obj.ymin + (obj.ymax-obj.ymin)/2
        #print("x: ", x_middle, "y:", y_middle)
        x_middle = round(x_middle, 0)
        y_middle = round(y_middle, 0)
        
        cv2.rectangle(color_frame, (int(obj.xmin), int(obj.ymin)), (int(obj.xmax), int(obj.ymax)), (0,255,0),2)
        cv2.circle(color_frame, (int(x_middle), int(y_middle)), 5, (0, 255, 0), 2)
        cv2.circle(color_frame, (int(width/2), int(height/2)), 5, (0, 0, 255), 2)
        cv2.line(color_frame, (int(x_middle), int(y_middle)), (int(width/2), int(height/2)), (0,0,255), 2)

        ret, distance_frame_depth = dc.get_aligned_frame()

        distance = distance_frame_depth.get_distance(int(x_middle), int(y_middle))
        distance = round(distance*100, 2)
        print("Distance: ", distance)
        #depth = depth_frame[int(x_middle), int(y_middle)] 
        #dx ,dy, dz = rs.rs2_deproject_pixel_to_point(depth_frame, [x_middle,y_middle], depth)
        #distance = math.sqrt(((dx)**2) + ((dy)**2) + ((dz)**2))
        #print("Distance: ", distance)
        cv2.putText(color_frame, "{}cm".format(distance), (int(x_middle), int(y_middle) - 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)

    except:
        print("Object not detected")
    # Show distance for a specific point
    #cv2.circle(color_frame, point, 4, (0, 0, 255))
    #distance = depth_frame[x_middle, y_middle]
    #print("Distance: ", distance)
    
    cv2.imshow("Color frame", color_frame)
    key = cv2.waitKey(1)
    if key == 27:
        break