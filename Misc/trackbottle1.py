#################################################
# Visual Servoing Robotic Arm
# Created by: AL, TAP, SFMS
# Hardware:
#   Kinova Kortex Lite Gen 3
#   Intel L515
#################################################
#settings
KP=0.1
KI=100
KD=50
#################################################
import pyrealsense2 as rs
import numpy as np
import cv2
import torch
import time
import sys
import os
import threading

from time import perf_counter_ns, sleep

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

def example_move_to_home_position(base):
    # Make sure the arm is in Single LEvel Servoing Mode
    base_servo_mode = Base_pb2.ServoingModeInformation()
    base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
    base.SetServoingMode(base_servo_mode)

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

config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
time.sleep(5)
print('model created')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import utilities

args = utilities.parseConnectionArguments()
with utilities.DeviceConnection.createTcpConnection(args) as router:
    base = BaseClient(router)
    base_cyclic = BaseCyclicClient(router)
    example_move_to_home_position(base)
    time.sleep(5)
    try:
        while True:
            timeDelta = (perf_counter_ns() - prevLoopTime) / 1e9  #[sec]
            prevLoopTime = perf_counter_ns()
            secondsSinceStart = (perf_counter_ns() - startTime) / 1e9
            st = time.time()
            # Wait for coherent pair of frames: depth and color
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
                #detectTime = perf_counter_ns()
                obj = objs_name.iloc[0]
                middle_x = (obj.xmax-obj.xmin)/2 + obj.xmin
                middle_y = (obj.ymax-obj.ymin)/2 + obj.ymin
                xdist = middle_x-width/2
                ydist = middle_y-height/2
                #print('xdist: ' + str(round(xdist,2)) + '\t' + 'ydist: ' + str(round(ydist,2)))
            #    anglex, angley = get_angles(int(middle_x), int(middle_y), width, height)
            #    dist = round(depth_image[int(middle_y)][int(middle_x)] * depth_scale,2)
            #    x,y,z = get_cart(anglex, angley, dist)
                #detectRate = (detectTime-lastDetectTime)/ 1e9
                #lastDetectTime=detectTime
                print('xdist: ' + str(round(xdist,2)) + '\t' + 'ydist: ' + str(round(ydist,2)) + '\t' + "timeDelta: " + str(timeDelta))
                
                cv2.rectangle(color_image, (int(obj.xmin), int(obj.ymin)), (int(obj.xmax), int(obj.ymax)), (0,255,0),2)
                cv2.circle(color_image, (int(middle_x), int(middle_y)), 5, (0, 255, 0), 2)
                cv2.circle(color_image, (int(width/2), int(height/2)), 5, (0, 0, 255), 2)
                cv2.line(color_image, (int(middle_x), int(middle_y)), (int(width/2), int(height/2)), (0,0,255), 2)

                 #PID Controller X
                target_X = 0
                error_X = target_X - xdist
                integral_X += error_X * timeDelta
                derivative_X = (error_X - prevError_X) / timeDelta
                prevError_X = error_X
                PIDoutput_X = KP * error_X + KI * integral_X + KD * derivative_X
                
                #PID Controller Y
                target_Y = 150
                error_Y = target_Y - ydist
                integral_Y += error_Y * timeDelta
                derivative_Y = (error_Y - prevError_Y) / timeDelta
                prevError_Y = error_Y
                PIDoutput_Y = KP * error_Y + KI * integral_Y + KD * derivative_Y
                print("ok")
                yOffset = 150
                Kp = 0.1
                speedx = -Kp * xdist
                #speedy = -Kp * (ydist-yOffset)
                speedy = 0
                deadband = 2 
                if(speedx > deadband):
                    speedx = deadband
                if(speedx < -deadband):
                    speedx = -deadband
                if(speedy > deadband):
                    speedy = deadband
                if(speedy < -deadband):
                    speedy = -deadband
                #if(abs(xdist) > 7 and abs(ydist) > 7):
                #if(abs(xdist) > 5 or abs(ydist) > 165 or abs(ydist) < 135):
                #speeds = [speedx/50, speedy/50, 0.01, -speedy*2, speedx*2, 0]
                time.sleep(0.14)
                if(moveTowards == True):
                    speeds = [speedx/50, speedy/50, 0.01, -speedy*2, speedx*2, 0]
                else:
                    speeds = [speedx/50, speedy/50, 0, -speedy*2, speedx*2, 0]
                
                sendSpeed(base, speeds)

                
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
        #stop streaming
        pipeline.stop()