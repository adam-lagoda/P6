#################################################
# Visual Servoing Robotic Arm
# Created by: AL, TAP, SFMS
# Hardware:
#   Kinova Kortex Lite Gen 3
#   Intel L515
#################################################
#PID coefficients setup
KP=0.1
KI=100
KD=50
#################################################
#Camera lens parameters setup
P_PIXEL_WIDTH = 43      #Pixel of object with distance D_initial #43 for fisheye #94 without
D_KNOWN_DISTANCE = 52   #Measured distance from camera to object in cm #52 for fishye #40 without
W_KNOWN_WIDTH = 5.7     #Known width of object in cm
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
import threading
import pyrealsense2 as rs
import utilities #2nd script, holds neccessary functions
from charset_normalizer import detect
from time import *
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import Base_pb2, BaseCyclic_pb2, Common_pb2
#################################################
#Functions used for distance calculation based on the focal length and fixed, known object size
def focal_length(pixelWidth, knownDistance, knownWidth):
    focalLength = (pixelWidth * knownDistance) / knownWidth
    return focalLength

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
        finger.value = 0.1
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
        finger.value = -0.1
        self.base.SendGripperCommand(gripper_command)
        gripper_request = Base_pb2.GripperRequest()    
        # Wait for reported speed to be 0
        gripper_request.mode = Base_pb2.GRIPPER_SPEED
        while True:
            gripper_measure = self.base.GetMeasuredGripperMovement(gripper_request)
            if len (gripper_measure.finger):
                print("Current speed is : {0}".format(gripper_measure.finger[0].value))
                if gripper_measure.finger[0].value == 0.0:
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

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

#Define initial valaues for the variables
follow = False
moveTowards = False
closeToObject = False
gripperClosed = False
runOnce = False
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

#Setup the camera output parameters: 940x540 resolution, 8-bit brg color format, 30fps
config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)

#Start streaming the camera preview
profile = pipeline.start(config)

#Load the YOLO object detection model
model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
time.sleep(5)
print('Model has been downloaded and created')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

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
        example.open_gripper()
        time.sleep(5)
        try:
            while True:
                #Capture timestamps, used for performance measurement and PID calculation
                timeDelta = (perf_counter_ns() - prevLoopTime) / 1e9  #[sec]
                prevLoopTime = perf_counter_ns() #[nanosec]
                secondsSinceStart = (perf_counter_ns() - startTime) / 1e9 #[sec]
                
                #st = time.time()
                
                #Wait for the frames to arrive from the camera and save them
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                
                if not color_frame:
                    continue
                
                #Convert image to numpy arrays
                color_image = np.asanyarray(color_frame.get_data())
                color_colormap_dim = color_image.shape
                color_dim = color_image.shape
                
                #Camera image dimensions
                height = color_image.shape[0]
                width = color_image.shape[1]
                
                #Use the created model to detect the object in the image, in this case a bottle
                result = model(color_image)
                objs = result.pandas().xyxy[0]
                objs_name = objs.loc[objs['name'] == 'bottle'] #people
                
                try:
                    #Calculate the middle point of the detected object, based on its bounding box dimensions
                    obj = objs_name.iloc[0]
                    x_middle = obj.xmin + (obj.xmax-obj.xmin)/2
                    y_middle = obj.ymin + (obj.ymax-obj.ymin)/2
                    
                    #Calculate the distance from the middle of the camera frame view, to the middle of the object
                    x_distance = x_middle-width/2
                    y_distance = y_middle-height/2
                    
                    #Calculate the distance to the object, based on its width and camera's focal length
                    F = focal_length(P_PIXEL_WIDTH, D_KNOWN_DISTANCE, W_KNOWN_WIDTH)
                    W_object_width = obj.xmax-obj.xmin
                    D = distance_to_camera(F, W_KNOWN_WIDTH, W_object_width)
                    
                    #Print known data for debugging
                    print('X distance: ' + str(round(x_distance,2)) + '\t' + 'Y distance: ' + str(round(y_distance,2)) + '\t' + 'Object Width:' + str(W_object_width) + '\t' + 'Distance:' + str(D) + '\t' + "timeDelta: " + str(timeDelta))    
                    
                    #Mark the middle of the camera view and the middle of the object, with a line connecting them, as well as object's bounding box
                    cv2.rectangle(color_image, (int(obj.xmin), int(obj.ymin)), (int(obj.xmax), int(obj.ymax)), (0,255,0),2)
                    cv2.circle(color_image, (int(x_middle), int(y_middle)), 5, (0, 255, 0), 2)
                    cv2.circle(color_image, (int(width/2), int(height/2)), 5, (0, 0, 255), 2)
                    cv2.line(color_image, (int(x_middle), int(y_middle)), (int(width/2), int(height/2)), (0,0,255), 2)

                    #PID Controller X
                    target_X = 0
                    error_X = target_X - x_distance
                    integral_X += error_X * timeDelta
                    derivative_X = (error_X - prevError_X) / timeDelta
                    prevError_X = error_X
                    PIDoutput_X = KP * error_X + KI * integral_X + KD * derivative_X
                    
                    #PID Controller Y
                    target_Y = 150
                    error_Y = target_Y - y_distance
                    integral_Y += error_Y * timeDelta
                    derivative_Y = (error_Y - prevError_Y) / timeDelta
                    prevError_Y = error_Y
                    PIDoutput_Y = KP * error_Y + KI * integral_Y + KD * derivative_Y
                    
                    #Saturate the maximum output from the PID for both axis to 2m/s, as a sanity check
                    deadband = 2 
                    if(PIDoutput_X > deadband):
                        PIDoutput_X = deadband
                    if(PIDoutput_X < -deadband):
                        PIDoutput_X = -deadband
                    if(PIDoutput_Y > deadband):
                        PIDoutput_Y = deadband
                    if(PIDoutput_Y < -deadband):
                        PIDoutput_Y = -deadband
                    
                    #If the distance to the object is more then 11cm, start centering and moving towards it, provided the boundaries are met
                    if(D > 11):
                        closeToObject = False
                        if(abs(x_distance) > (target_X + 5) or abs(y_distance) > (target_Y + 15) or abs(y_distance) < (target_Y - 15)):
                            velocities = [PIDoutput_X/50, PIDoutput_Y/50, 0, -PIDoutput_Y*2, PIDoutput_X*2, 0]
                            sendSpeed(base, velocities)
                        else:
                            velocities = [PIDoutput_X/50, PIDoutput_Y/50, 0.01, -PIDoutput_Y*2, PIDoutput_X*2, 0]
                            sendSpeed(base, velocities)                       
                    else:
                        closeToObject = True
                    
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
                cv2.imshow('Camera Preview', color_image)
                key = cv2.waitKey(1)

                if key & 0xFF == ord('q') or key == 27:
                    cv2.destroyAllWindows()
                    break
                if key & 0xFF == ord('m'): #or key == 27:
                    if(moveTowards ==True):
                        moveTowards = False
                    else:
                        moveTowards = True
                if key & 0xFF == ord('f'): #or key == 27:
                    if(follow ==True):
                        follow = False
                    else:
                        follow = True        
        finally:
            #Stop streaming the camera preview
            pipeline.stop()
