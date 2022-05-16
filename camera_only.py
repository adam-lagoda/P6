import pyrealsense2 as rs
import numpy as np
import cv2
import torch
import time
import sys
import os
import threading

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

#model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
#model = torch.hub.load('ultralytics/yolov5', 'custom', path='path/to/best.pt')  # local model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='path/to/best1.pt')

time.sleep(5)
print('model created')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
#import utilities
try:
        while True:
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
            objs_name = objs.loc[objs['name'] == 'weed']
            
            try:
                obj = objs_name.iloc[0]
                middle_x = (obj.xmax-obj.xmin)/2 + obj.xmin
                middle_y = (obj.ymax-obj.ymin)/2 + obj.ymin
                xdist = middle_x-width/2
                ydist = middle_y-height/2
                #print('xdist: ' + str(round(xdist,2)) + '\t' + 'ydist: ' + str(round(ydist,2)))
            #    anglex, angley = get_angles(int(middle_x), int(middle_y), width, height)
            #    dist = round(depth_image[int(middle_y)][int(middle_x)] * depth_scale,2)
            #    x,y,z = get_cart(anglex, angley, dist)
                print('xdist: ' + str(round(xdist,2)) + '\t' + 'ydist: ' + str(round(ydist,2)))    
                
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

