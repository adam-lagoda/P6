import pyrealsense2 as rs
import numpy as np
import math
import cv2
import os
import sys
import time
import torch
import threading

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 60)    
config.enable_stream(rs.stream.depth, 1024, 768, rs.format.z16, 30)
pipeline.start(config)

x = 512
y = 384

align_to = rs.stream.depth
align = rs.align(align_to)
#model = torch.hub.load('ultralytics/yolov5', 'custom', path='path/to/best1.pt')
#time.sleep(5)  
#print('Model has been downloaded and created') 
#sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        
try:
    while True:
        # This call waits until a new coherent set of frames is available on a device
        frames = pipeline.wait_for_frames()
        
        #Aligning color frame to depth frame
        aligned_frames =  align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        aligned_color_frame = aligned_frames.get_color_frame()
        #color_frame = frames.get_color_frame()

        
        if not depth_frame or not aligned_color_frame: continue     

        color_intrin = aligned_color_frame.profile.as_video_stream_profile().intrinsics
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(aligned_color_frame.get_data())
        
        height = color_image.shape[0]
        width = color_image.shape[1]

        
        depth = depth_frame.get_distance(x, y)
        dx ,dy, dz = rs.rs2_deproject_pixel_to_point(color_intrin, [x,y], depth)
        distance = math.sqrt(((dx)**2) + ((dy)**2) + ((dz)**2))
        
    
        print('Distance from camera to pixel:' + '\t' +str(round(distance, 2 )))
        print("Z-depth from camera surface to pixel surface:", depth)
        
        cv2.namedWindow('RealSense Depth', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', depth_image) #depth_image
        cv2.namedWindow('RealSense Color', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense Color', color_image)
        key = cv2.waitKey(1)

        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
except:
    pipeline.stop()