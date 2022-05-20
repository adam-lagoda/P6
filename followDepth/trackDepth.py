#################################################
# Visual Servoing Robotic Arm
# Created by: AL, TAP, SFMS // AIE6-3
# Bachelor Thesis
# Aalborg Univeristy Esbjerg 2022
# Hardware:
#   Kinova Kortex Lite Gen 3
#   Intel L515
##############################################KP=0.005KI=0KD=0###
# Change the object here \/
SELECTED_OBJECT = 'bottle' #bottle or weed
##############################################
if SELECTED_OBJECT == 'bottle':
    TARGET_X = 0
    TARGET_Y = 170 #110 plant 170 bottle
    DISTANCE_HARDCODED_TIME = 3.75 #3.75 dor bottle 3 for plant

else:
    TARGET_X = 0
    TARGET_Y = 20 #110 plant 170 bottle
    DISTANCE_HARDCODED_TIME = 3.75 #3.75 dor bottle 3 for plant
##############################################
#PID coefficients setup
KP=0.025 #0.025 #0.05 oscillating #o.008 old
KI=0 #0.001
KD=0 #0.001


THRESHOLD_CENTERING = 35 #25

DISTANCE_HARDCODED_THRESHOLD = 27

TARGET_Z = 10
KP_Z = 0.005
#################################################
#Camera lens parameters setup
P_PIXEL_WIDTH = 162      #Pixel of object with distance D_initial #43 for fisheye #94 without #plant 157
D_KNOWN_DISTANCE = 56   #Measured distance from camera to object in cm #52 for fishye #40 without #plant 46
W_KNOWN_WIDTH = 17.5     #Known width of object in cm 5.7 #plant 6
#################################################
#Constants definitions
TIMEOUT_DURATION = 20
#################################################
#Importing the libraries
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
#import utilities #2nd script, holds neccessary functions
from realsense_depth import *
from time import sleep, perf_counter_ns
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import Base_pb2, BaseCyclic_pb2, Common_pb2
#################################################
#Functions used for distance calculation based on the focal length and fixed, known object size
def focal_length(pixelWidth, knownDistance, knownWidth):
    focalLength = (pixelWidth * knownDistance) / knownWidth
    return focalLength
#################################################
def distance_to_camera (focalLength, knownWidth, pixelWidth):
    distance = (knownWidth * focalLength) / pixelWidth
    return distance

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

    base.SendTwistCommand(command)

class GripperCommandExample:
    def __init__(self, router, proportional_gain = 2.0):

        self.proportional_gain = proportional_gain
        self.router = router

        # Create base client using TCP router
        self.base = BaseClient(self.router)


    def open_gripper(self):
        # Create the GripperCommand we will send
        gripper_command = Base_pb2.GripperCommand()
        finger = gripper_command.gripper.finger.add()

        gripper_command.mode = Base_pb2.GRIPPER_SPEED
        finger.value = 0.2
        self.base.SendGripperCommand(gripper_command)
        gripper_request = Base_pb2.GripperRequest()    

        # Wait for reported position to be opened
        gripper_request.mode = Base_pb2.GRIPPER_POSITION
        while True:
            gripper_measure = self.base.GetMeasuredGripperMovement(gripper_request)
            if len (gripper_measure.finger):
                print("Current position is : {0}".format(gripper_measure.finger[0].value))
                if gripper_measure.finger[0].value < 0.01:
                    break
            else: # Else, no finger present in answer, end loop
                break

    def close_gripper(self):
        # Create the GripperCommand we will send
        gripper_command = Base_pb2.GripperCommand()
        finger = gripper_command.gripper.finger.add()            

        gripper_command.mode = Base_pb2.GRIPPER_SPEED
        finger.value = -0.2
        self.base.SendGripperCommand(gripper_command)
        gripper_request = Base_pb2.GripperRequest()    
        # Wait for reported speed to be 0
        gripper_request.mode = Base_pb2.GRIPPER_SPEED
        while True:
            gripper_measure = self.base.GetMeasuredGripperMovement(gripper_request)
            if len (gripper_measure.finger):
                print("Current speed is : {0}".format(gripper_measure.finger[0].value))
                if gripper_measure.finger[0].value == 0.00:
                    print("Gripper closed")
                    break
            else: # Else, no finger present in answer, end loop
                break

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
        if action.name == "sweepHome":#bottlehome
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

def saturation(variable, limit):
    if(variable > limit):
        variable = limit
    if(variable < -limit):
        variable = -limit
    return variable

def feedbackPosition(base, base_cyclic):
    feedback = base_cyclic.RefreshFeedback()
    feedbackX.append(feedback.base.tool_pose_x)         # (meters)
    feedbackY.append(feedback.base.tool_pose_y) 
    feedbackZ.append(feedback.base.tool_pose_z)

def show_distance(event, x, y, args, params):
    global point
    point = (x, y)

#Define initial valaues for the variables
follow = False
grip = False
closeToObject = False
gripperClosed = False
runOnce = False
centered = False
startTime = perf_counter_ns()
prevLoopTime = perf_counter_ns()
integral_X=0
integral_Y=0
derivative_X=0
derivative_Y=0
PIDoutput_X=0
PIDoutput_X=0
prevError_X=0
prevError_Y=0

timePlot = []
dataX = []
dataY = []
feedbackX = []
feedbackY = []
feedbackZ = []

point = (400, 300)
    
dc = DepthCamera()
#Load the YOLO object detection model
if SELECTED_OBJECT == 'bottle':
    model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
else:
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='path/to/old-best.pt')  # local model
#model = torch.hub.load('ultralytics/yolov5', 'custom', path='path/to/best1.pt')

print('Model has been downloaded and created')
time.sleep(5)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import utilities

args = utilities.parseConnectionArguments()
#Start the main loop
while True:
    with utilities.DeviceConnection.createTcpConnection(args) as router:
        #Initialize the connection and log in to the Web API
        base = BaseClient(router)
        base_cyclic = BaseCyclicClient(router)
        #Set the Servoing Mode
        singleLevelServoingMode()
        #Move the robot to the home position
        home_position(base)
        #Initialize the gripper and make sure it is open
        example = GripperCommandExample(router)
        print("before gripper open")
        example.open_gripper()
        print("after gripper open")
        time.sleep(5)
        try:
            while True:
                #Capture timestamps, used for performance measurement and PID calculation
                timeDelta = (perf_counter_ns() - prevLoopTime) / 1e9  #[sec]
                prevLoopTime = perf_counter_ns() #[nanosec]
                timePlot.append(prevLoopTime)
                secondsSinceStart = (perf_counter_ns() - startTime) / 1e9 #[sec]
                st = time.time()
                
                #Wait for the frames to arrive from the camera and save them
                if runOnce == False:
                    time.sleep(5)
                    runOnce = True
                    
                
                ret, depth_frame, color_frame, depth_image = dc.get_frame()

                result = model(color_frame)
                objs = result.pandas().xyxy[0]
                objs_name = objs.loc[objs['name'] == SELECTED_OBJECT]
                
                height = color_frame.shape[0]
                width = color_frame.shape[1]
                
                x_middle = 0
                y_middle = 0
                try:
                    #Calculate the middle point of the detected object, based on its bounding box dimensions
                    obj = objs_name.iloc[0]
                    x_middle = obj.xmin + (obj.xmax-obj.xmin)/2
                    y_middle = obj.ymin + (obj.ymax-obj.ymin)/2
                    
                    x_middle = round(x_middle, 0)
                    y_middle = round(y_middle, 0)
                    #Calculate the distance from the middle of the camera frame view, to the middle of the object
                    x_distance = x_middle-width/2
                    y_distance = y_middle-height/2
                    dataX.append(x_distance)
                    dataY.append(y_distance)
                    timePlot.append(prevLoopTime)
                    feedbackPosition(base, base_cyclic)
                    
                    #Mark the middle of the camera view and the middle of the object, with a line connecting them, as well as object's bounding box
                    cv2.rectangle(color_frame, (int(obj.xmin), int(obj.ymin)), (int(obj.xmax), int(obj.ymax)), (0,255,0),2)
                    cv2.circle(color_frame, (int(x_middle), int(y_middle)), 5, (0, 255, 0), 2)
                    cv2.circle(color_frame, (int(width/2), int(height/2)), 5, (0, 0, 255), 2)
                    cv2.line(color_frame, (int(x_middle), int(y_middle)), (int(width/2), int(height/2)), (0,0,255), 2)
                    
                    #Calculate the distance to the center of the object based on the depth camera
                    ret, depth_aligned_frame, color_aligned_frame, distance_frame_depth = dc.get_aligned_frame()

                    D = distance_frame_depth.get_distance(int(x_middle), int(y_middle))
                    D = round(D*100, 2)
                                   
                    #PID Controller X
                    #target_X = 0
                    error_X = TARGET_X - x_distance
                    integral_X += error_X * timeDelta
                    integral_X = saturation(integral_X, 100)
                    derivative_X = (error_X - prevError_X) / timeDelta
                    prevError_X = error_X
                    PIDoutput_X = KP * error_X + KI * integral_X + KD * derivative_X
                    PIDoutput_X = saturation(PIDoutput_X, 1)
                    
                    #PID Controller Y
                    #target_Y = 150
                    error_Y = TARGET_Y - y_distance
                    integral_Y += error_Y * timeDelta
                    integral_Y = saturation(integral_Y, 100)
                    derivative_Y = (error_Y - prevError_Y) / timeDelta
                    prevError_Y = error_Y
                    PIDoutput_Y = KP * error_Y + KI * integral_Y + KD * derivative_Y
                    PIDoutput_Y = saturation(PIDoutput_Y, 1)
                    
                    #P Controller Z
                    error_Z = D - TARGET_Z
                    Poutput_Z = KP_Z * error_Z
                    Poutput_Z = saturation(Poutput_Z, 0.05)
                    
                    #Print data for debugging
                    print('X distance: ' + str(round(x_distance,2)) + '\t' + 'Y distance: ' + str(round(y_distance,2)) + '\t' + 'Distance:' + str(D) + '\t' + "timeDelta: " + str(timeDelta) + '\t' + 'Speed: ' + str(Poutput_Z))    
                         
                    
                    #Saturate the maximum output from the PID for both axis to 2m/s, as a sanity check
                    #If the distance to the object is more then 11cm, start centering and moving towards it, provided the boundaries are met
                    if D<=DISTANCE_HARDCODED_THRESHOLD and centered:
                        closeToObject = True
                        velocities = [0, 0, 0.05, 0 ,0 ,0]
                        sendSpeed(base, velocities)
                        time.sleep(DISTANCE_HARDCODED_TIME)
                    elif(D>DISTANCE_HARDCODED_THRESHOLD) or not centered:
                        closeToObject = False
                        if(abs(x_distance) > abs(TARGET_X + THRESHOLD_CENTERING) or abs(y_distance) > abs(TARGET_Y + THRESHOLD_CENTERING) or abs(y_distance) < abs(TARGET_Y - THRESHOLD_CENTERING)):
                            centered = False
                            velocities = [PIDoutput_X/50, PIDoutput_Y/50, 0, -PIDoutput_Y*2, PIDoutput_X*2, 0]
                            sendSpeed(base, velocities)
                        else:
                            centered = True
                            velocities = [PIDoutput_X/50, PIDoutput_Y/50, Poutput_Z, -PIDoutput_Y*2, PIDoutput_X*2, 0]
                            sendSpeed(base, velocities)
                        
            
                    #If the distance to the object is less then 11cm, stop moving and close the gripper, grabbing the object
                    if closeToObject:
                        example = GripperCommandExample(router)
                        velocities = [0, 0, 0, 0, 0, 0]
                        sendSpeed(base, velocities)   
                        example.close_gripper()
                        #time.sleep(5)
                        gripperClosed = True
                    
                    #If the end effector is close to the object and the gripper is closed, break the inner loop and start over, by returning to home position
                    if closeToObject and gripperClosed:
                        break
                except:
                    velocities = [0, 0, 0, 0, 0, 0]
                    sendSpeed(base, velocities)

                #Display a window with a camera preview
                cv2.namedWindow('Camera Preview', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('Camera Preview', color_frame)
                #cv2.imshow("depth frame", depth_frame)
                key = cv2.waitKey(1)

                if key & 0xFF == ord('q') or key == 27:
                    cv2.destroyAllWindows()
                    break
                if key & 0xFF == ord('g'): #or key == 27:
                    if(grip ==True):
                        grip = False
                    else:
                        grip = True
                if key & 0xFF == ord('f'): #or key == 27:
                    if(follow ==True):
                        follow = False
                    else:
                        follow = True        
        finally:
            #Stop streaming the camera preview
            #plt.plot(timePlot, dataX)
            data = pd.DataFrame({tuple(timePlot), tuple(dataX), tuple(dataY), tuple(feedbackX), tuple(feedbackY), tuple(feedbackZ)})
            data.to_excel('sample_data.xlsx', sheet_name='sheet1', index=False)
            dc.release()
