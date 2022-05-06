'''
    File name         : objTracking.py
    Description       : Main file for object tracking
    Author            : Rahmad Sadli
    Date created      : 20/02/2020
    Python Version    : 3.7
'''

import cv2
import torch
import sys
import time
import os
import pyrealsense2 as rs
import numpy as np
import threading
from Detector import detect
from KalmanFilter import KalmanFilter

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
time.sleep(5)
print('model created')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
def main():

    # Create opencv video capture object
    #VideoCap = cv2.VideoCapture(0)

    #Variable used to control the speed of reading the video
    ControlSpeedVar = 100  #Lowest: 1 - Highest:100

    HiSpeed = 100

    #Create KalmanFilter object KF
    #KalmanFilter(dt, u_x, u_y, std_acc, x_std_meas, y_std_meas)

    KF = KalmanFilter(0.1, 1, 1, 1, 0.1,0.1)

    debugMode=1
    try:
        while(True):
            st = time.time()

            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()

            if not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())

            color_colormap_dim = color_image.shape
                
            color_dim = color_image.shape
            
            height = color_image.shape[0]
            width = color_image.shape[1]
            # Read frame
            #ret, frame = VideoCap.read()
            
            # Detect object
            centers = detect(color_image,debugMode)

            # If centroids are detected then track them
            if (len(centers) > 0):

                # Draw the detected circle
                cv2.circle(color_image, (int(centers[0][0]), int(centers[0][1])), 10, (0, 191, 255), 2)

                # Predict
                (x, y) = KF.predict()
                # Draw a rectangle as the predicted object position
                #cv2.rectangle(frame, (x - 15, y - 15), (x + 15, y + 15), (255, 0, 0), 2)
                
                # Update
                (x1, y1) = KF.update(centers[0])

                # Draw a rectangle as the estimated object position
                #cv2.rectangle(frame, (x1 - 15, y1 - 15), (x1 + 15, y1 + 15), (0, 0, 255), 2)

                #cv2.putText(frame, "Estimated Position", (x1 + 15, y1 + 10), 0, 0.5, (0, 0, 255), 2)
                #cv2.putText(frame, "Predicted Position", (x + 15, y), 0, 0.5, (255, 0, 0), 2)
                #cv2.putText(frame, "Measured Position", (centers[0][0] + 15, centers[0][1] - 15), 0, 0.5, (0,191,255), 2)
            cv2.namedWindow('RealSense',cv2.WINDOW_AUTOSIZE)
            cv2.imshow('image', color_image)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
            #if cv2.waitKey(2) & 0xFF == ord('q'):
            #    VideoCap.release()
            #    cv2.destroyAllWindows()
            #    break

            cv2.waitKey(HiSpeed-ControlSpeedVar+1)
    finally:
        pipeline.stop()


if __name__ == "__main__":
    # execute main
    main()