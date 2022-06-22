import pyrealsense2 as rs
import numpy as np
import cv2
import torch
import time
import sys
import os
import threading
#################################################
#Camera lens parameters setup
P_PIXEL_WIDTH = 200      #Pixel of object with distance D_initial #43 for fisheye #94 without #plant 157
D_KNOWN_DISTANCE = 46   #Measured distance from camera to object in cm #52 for fishye #40 without #plant 46
W_KNOWN_WIDTH = 6    #Known width of object in cm 5.7 #plant 6
#################################################

#Functions used for distance calculation based on the focal length and fixed, known object size
def focal_length(pixelWidth, knownDistance, knownWidth):
    focalLength = (pixelWidth * knownDistance) / knownWidth
    return focalLength
#################################################
def distance_to_camera (focalLength, knownWidth, pixelWidth):
    distance = (knownWidth * focalLength) / pixelWidth
    return distance
#################################################

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

#model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
#model = torch.hub.load('ultralytics/yolov5', 'custom', path='path/to/best.pt')  # local model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='path/to/old-best.pt')

time.sleep(5)
print('model created')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
#import utilities
try:
        while True:
            st = time.time()
            
            #Wait for the frames to arrive from the camera and save them
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            
            if not color_frame:
                continue
            
            #Convert image to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())
            
            #Get Camera image dimensions
            height = color_image.shape[0]
            width = color_image.shape[1]
            
            #Use the created model to detect the object in the image, in this case a bottle
            result = model(color_image)
            objs = result.pandas().xyxy[0]
            objs_name = objs.loc[objs['name'] == 'weed'] #bottle #weed
            
            try:
                obj = objs_name.iloc[0]
                middle_x = (obj.xmax-obj.xmin)/2 + obj.xmin
                middle_y = (obj.ymax-obj.ymin)/2 + obj.ymin
                xdist = middle_x-width/2
                W_object_width = obj.xmax-obj.xmin
                ydist = middle_y-height/2
                
                F = focal_length(P_PIXEL_WIDTH, D_KNOWN_DISTANCE, W_KNOWN_WIDTH)
                W_object_width = obj.xmax-obj.xmin
                D = distance_to_camera(F, W_KNOWN_WIDTH, W_object_width)
                #print('xdist: ' + str(round(xdist,2)) + '\t' + 'ydist: ' + str(round(ydist,2)))
            #    anglex, angley = get_angles(int(middle_x), int(middle_y), width, height)
            #    dist = round(depth_image[int(middle_y)][int(middle_x)] * depth_scale,2)
            #    x,y,z = get_cart(anglex, angley, dist)
                print('xdist: ' + str(round(xdist,2)) + '\t' + 'ydist: ' + str(round(ydist,2)) + '\t' ++ 'width[px]: ' + str(round(W_object_width,2)))    
                cv2.putText(color_frame, "{}cm".format(D), (int(middle_x), int(middle_y) - 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)

                cv2.rectangle(color_image, (int(obj.xmin), int(obj.ymin)), (int(obj.xmax), int(obj.ymax)), (0,255,0),2)
                cv2.circle(color_image, (int(middle_x), int(middle_y)), 5, (0, 255, 0), 2)
                cv2.circle(color_image, (int(width/2), int(height/2)), 5, (0, 0, 255), 2)
                cv2.line(color_image, (int(middle_x), int(middle_y)), (int(width/2), int(height/2)), (0,0,255), 2)
            except:
                print('fail')
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', color_image)
            key = cv2.waitKey(1)

            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
finally:      
    #stop streaming
    pipeline.stop()

