from charset_normalizer import detect
import pyrealsense2 as rs
import numpy as np
import cv2
import torch
import time
import sys
import os
import threading

P_pixel_Width = 43      #Pixel of object with distance D_initial #43 for fisheye #94 without
D_known_distance = 52   #Measured distance from camera to object in cm #52 for fishye #40 without
W_Known_width = 5.7     #Known width of object in cm

def focal_length(pixelWidth, knownDistance, knownWidth):
    focalLength = (pixelWidth * knownDistance) / knownWidth
    return focalLength

def distance_to_camera (focalLength, knownWidth, pixelWidth):
    distance = (knownWidth * focalLength) / pixelWidth
    return distance

from time import perf_counter_ns
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient

from kortex_api.autogen.messages import Base_pb2, BaseCyclic_pb2, Common_pb2

TIMEOUT_DURATION = 20

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

def sendSpeed(base, speeds):
    command = Base_pb2.TwistCommand()
    command.reference_frame = Base_pb2.CARTESIAN_REFERENCE_FRAME_TOOL
    command.duration = 0

    twist = command.twist
    twist.linear_x = speeds[0]
    twist.linear_y = speeds[1]
    twist.linear_z = speeds[2]
    twist.angular_x = speeds[3]
    twist.angular_y = speeds[4]
    twist.angular_z = speeds[5]

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
                print("Current position is : {0}".format(gripper_measure.finger[0].value) + "4444444444444444444444")
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
                print("Current speed is : {0}".format(gripper_measure.finger[0].value) + "66666666666666666666666")
                if gripper_measure.finger[0].value == 0.0:
                    break
            else: # Else, no finger present in answer, end loop
                break

def singleLevelServoingMode():
    # Make sure the arm is in Single LEvel Servoing Mode
    base_servo_mode = Base_pb2.ServoingModeInformation()
    base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
    base.SetServoingMode(base_servo_mode)
    
def example_move_to_home_position(base):
    # Move arm to ready position
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
    notification_handle = base.OnNotificationActionTopic(
        check_for_end_or_abort(e),
        Base_pb2.NotificationOptions()
    )
    
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

follow = False
moveTowards = False
closeToObject = False
gripperClosed = False
runOnce = False
detectTime = perf_counter_ns()
lastDetectTime = perf_counter_ns()
loopTime = perf_counter_ns()
lastLoopTime = perf_counter_ns()

config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
time.sleep(5)
print('model created')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import utilities

args = utilities.parseConnectionArguments()
while True:
    with utilities.DeviceConnection.createTcpConnection(args) as router:
        base = BaseClient(router)
        base_cyclic = BaseCyclicClient(router)
        singleLevelServoingMode()
        example_move_to_home_position(base)
        example = GripperCommandExample(router)
        example.open_gripper()
        time.sleep(5)
        try:
            while True:
                loopTime = perf_counter_ns()
                loopRate = (loopTime - lastLoopTime)/ 1e9
                lastLoopTime = loopTime
                st = time.time()
                # Wait for coherent pair of frames: depth and color
                #if not runOnce:
                #    frames = pipeline.wait_for_frames()
                #    runOnce = True
                frames = pipeline.wait_for_frames()

                #aligned_frames = align.process(frames)
                #aligned_depth_frame = aligned_frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                
                if not color_frame:
                    continue
                
                #Convert image to numpy arrays
                #depth_image = np.asanyarray(aligned_depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())
                
                color_colormap_dim = color_image.shape
                
                color_dim = color_image.shape
                
                height = color_image.shape[0]
                width = color_image.shape[1]
                #start_inf = time.time()
                result = model(color_image)
            #stop_inf = time.time()
                objs = result.pandas().xyxy[0]
                #objs_name = objs.loc[objs['name'] == 'bottle']
                objs_name = objs.loc[objs['name'] == 'bottle'] #people
                
                try:
                    detectTime = perf_counter_ns()
                    obj = objs_name.iloc[0]
                    middle_x = (obj.xmax-obj.xmin)/2 + obj.xmin
                    middle_y = (obj.ymax-obj.ymin)/2 + obj.ymin
                    xdist = middle_x-width/2
                    ydist = middle_y-height/2

                    F = focal_length(P_pixel_Width, D_known_distance, W_Known_width)
                    W_object_width = obj.xmax-obj.xmin
                    D = distance_to_camera(F, W_Known_width, W_object_width)



                    detectRate = (detectTime-lastDetectTime)/ 1e9
                    lastDetectTime=detectTime
                    print('xdist: ' + str(round(xdist,2)) + '\t' + 'ydist: ' + str(round(ydist,2)) + '\t' + 'object width:' + str(W_object_width) + '\t' + 'Distance:' + str(D) + '\t' + 'Focal Length:' + (str(F)))    
                    
                    cv2.rectangle(color_image, (int(obj.xmin), int(obj.ymin)), (int(obj.xmax), int(obj.ymax)), (0,255,0),2)
                    cv2.circle(color_image, (int(middle_x), int(middle_y)), 5, (0, 255, 0), 2)
                    cv2.circle(color_image, (int(width/2), int(height/2)), 5, (0, 0, 255), 2)
                    cv2.line(color_image, (int(middle_x), int(middle_y)), (int(width/2), int(height/2)), (0,0,255), 2)

                    #if(abs(xdist) < 10 and abs(ydist) < 10):
                        #xdist = 0
                        #ydist = 0
                    deadband = 2    
                    yOffset = 150
                    Kp = 0.1
                    speedx = -Kp * xdist
                    speedy = -Kp * (ydist-yOffset)
                    
                    if(speedx > deadband):
                        speedx = deadband
                    if(speedx < -deadband):
                        speedx = -deadband
                    if(speedy > deadband):
                        speedy = deadband
                    if(speedy < -deadband):
                        speedy = -deadband
                    #if(abs(xdist) > 7 and abs(ydist) > 7):
                    if(D > 11):
                        closeToObject = False
                        if(abs(xdist) > 5 or abs(ydist) > 165 or abs(ydist) < 135):
                            speeds = [speedx/100, speedy/100, 0, -speedy*2, speedx*2, 0]
                            sendSpeed(base, speeds)
                        else:
                            #speeds = [speedx/50, speedy/50, 0.05, -speedy*2, speedx*2, 0]                        
                            speeds = [0, 0, 0.05, 0, 0, 0]
                            sendSpeed(base, speeds)                       
                    else:
                        closeToObject = True
                        #example.ExampleSendGripperCommands()
                    if closeToObject:
                        example = GripperCommandExample(router)
                        speeds = [0, 0, 0, 0, 0, 0]
                        sendSpeed(base, speeds)   
                        example.close_gripper()
                        #time.sleep(5)
                        gripperClosed = True
                    if closeToObject and gripperClosed:
                        break
                        #time.sleep(5)
                        #example_move_to_home_position(base)
                        #time.sleep(5)
                        #example = GripperCommandExample(router)
                        #example.open_gripper()
                        #sendSpeed(base, speeds)

                    pxheight = obj.ymax - obj.ymin
                    pxwidth = obj.xmax - obj.xmin
                    heightratio = pxheight/height
                    widthratio = pxwidth/width
                except:
                    speeds = [0, 0, 0, 0, 0, 0]
                    sendSpeed(base, speeds)

                cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('RealSense', color_image)
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
            pipeline.stop()