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
import utilities
from realsense_depth import *
from time import sleep, perf_counter_ns
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import Base_pb2, BaseCyclic_pb2, Common_pb2
#################################################
#Functions
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

#################################################
#Start the main loop
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
    
        velocities = [0, 1, 0, 0 ,0 ,0]
        sendSpeed(base, velocities)
        time.sleep(1)
        velocities = [0, -1, 0, 0 ,0 ,0]
        sendSpeed(base, velocities)
        time.sleep(1)   