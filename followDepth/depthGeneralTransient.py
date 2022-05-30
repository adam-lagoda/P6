TIMEOUT_DURATION = 20

import os
import sys
import time
import numpy as np
import cv2
import torch
import math
import threading
import pyrealsense2 as rs
import matplotlib.pyplot as plt
import pandas as pd
import utilities
from realsense_depth import *
from time import sleep, perf_counter_ns
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import Base_pb2, BaseCyclic_pb2, Common_pb2
#################################################
def check_for_end_or_abort(e):
    """Return a closure checking for END or ABORT notifications

    Args:
    e -- event to signnnal when the action is completed
    (will be set when an END or ABORT occurs)
    """
    def check(notification, e=e):
        print("EVENT :" + \
            Base_pb2.ActionEvent.Name(notification.action_event))
        if notification.action_event == Base_pb2.ACTION_END \
        or notification.action_event == Base_pb2.ACTION_ABORT:
            e.set()
    return check

def sendSpeed(base, velocities):
    #Use the Twist command with a Cartesian frame as a reference
    command = Base_pb2.TwistCommand()
    command.reference_frame = Base_pb2.CARTESIAN_REFERENCE_FRAME_TOOL
    command.duration = 0
    #Extrude data of velocies in X, Y, Z, and <X, <Y, <Z from the function's input
    twist = command.twist
    twist.linear_x = velocities[0]
    twist.linear_y = velocities[1]
    twist.linear_z = velocities[2]
    twist.angular_x = velocities[3]
    twist.angular_y = velocities[4]
    twist.angular_z = velocities[5]
    #Sedn values to the base
    base.SendTwistCommand(command)

def singleLevelServoingMode():
    #Set the servoing mode of the arm to the Single Level (High Level) Servoing Mode
    base_servo_mode = Base_pb2.ServoingModeInformation()
    base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
    base.SetServoingMode(base_servo_mode)
    
def home_position(base):
    # Move the arm to the initial position
    print("Moving arm to a safe position")
    action_type = Base_pb2.RequestedActionType()
    action_type.action_type = Base_pb2.REACH_JOINT_ANGLES
    action_list = base.ReadAllActions(action_type)
    action_handle = None
    for action in action_list.action_list: #search through the library of saved movement
        if action.name == "sweepHome":
            action_handle = action.handle
            
    if action_handle == None:
        print("Can't reach safe position. Not in the list. Exiting")
        sys.exit(0)
    
    e = threading.Event()
    notification_handle = base.OnNotificationActionTopic(check_for_end_or_abort(e), Base_pb2.NotificationOptions()    )
    base.ExecuteActionFromReference(action_handle)
    
    #Leave time to action to complete
    finished = e.wait(TIMEOUT_DURATION)
    base.Unsubscribe(notification_handle)
    
    if finished:
        print("Safe position reached")
    else:
        print("Timeout on action notification wait")
    return finished

def show_distance(event, x, y, args, params):
    global point
    point = (x, y)

#################################################
point = (400, 300)

dataX = []
dataY = []
timePlot = []

left = True


# Initialize Camera Intel Realsense
dc = DepthCamera()

model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
#model = torch.hub.load('ultralytics/yolov5', 'custom', path='path/to/best.pt')
#model = torch.hub.load('ultralytics/yolov5', 'custom', path='path/to/old-best.pt')
#model = torch.hub.load('ultralytics/yolov5', 'custom', path='path/to/best1.pt')
time.sleep(5)  
print('Model has been downloaded and created') 

# Create mouse event
cv2.namedWindow("Color frame")
cv2.setMouseCallback("Color frame", show_distance)
prevTime = perf_counter_ns()#/1e9

args = utilities.parseConnectionArguments()
while True:
    with utilities.DeviceConnection.createTcpConnection(args) as router:
        #Initialize the connection and log in to the Web API
        base = BaseClient(router)
        base_cyclic = BaseCyclicClient(router)
        #Set the Servoing Mode
        singleLevelServoingMode()
        #Move the robot to the home position
        home_position(base)
        #velocities = [1, 0, 0, 0, 0 ,0]
        #sendSpeed(base, velocities)
        #time.sleep(0.5)
        #velocities = [0, 0, 0, 0, 0, 0]
        #sendSpeed(base, velocities)

        try:
            while True:
                ret, color_frame = dc.get_frame()

                result = model(color_frame)
                objs = result.pandas().xyxy[0]
                objs_name = objs.loc[objs['name'] == 'bottle']
                
                height = color_frame.shape[0]
                width = color_frame.shape[1]
                
                x_middle = 0
                y_middle = 0
                try:
                    currentTime = perf_counter_ns()#/1e9
                    obj = objs_name.iloc[0]
                    x_middle = obj.xmin + (obj.xmax-obj.xmin)/2
                    y_middle = obj.ymin + (obj.ymax-obj.ymin)/2
                    #print("x: ", x_middle, "y:", y_middle)
                    x_middle = round(x_middle, 0)
                    y_middle = round(y_middle, 0)
                    
                    x_distance = x_middle-width/2
                    y_distance = y_middle-height/2
                    
                    dataX.append(x_distance)
                    dataY.append(y_distance)
                    plotTime = currentTime/1e9
                    timePlot.append(plotTime)
                    
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

                    if (currentTime - prevTime < 1000000000):
                        velocities = [0, 0, 0, 0, 0 ,0]
                        sendSpeed(base, velocities)
                    else:
                        velocities = [0.1, 0, 0, 0, 0, 0]
                        sendSpeed(base, velocities)
                    
                except:
                    print("Object not detected")
                    velocities = [0, 0, 0, 0, 0, 0]
                    sendSpeed(base, velocities)
                # Show distance for a specific point
                #cv2.circle(color_frame, point, 4, (0, 0, 255))
                #distance = depth_frame[x_middle, y_middle]
                #print("Distance: ", distance)
                
                #cv2.imshow("depth frame", depth_frame)
                cv2.imshow("Color frame", color_frame)
                key = cv2.waitKey(1)
                if key == 27:
                    break
        finally:
            #Save the feedback data to a excel sheet
            #Stop streaming the camera preview
            data = pd.DataFrame({tuple(timePlot),tuple(dataX)})
            data.to_excel('CameraTransientWithTime.xlsx', sheet_name='sheet1', index=False)
            dc.release()