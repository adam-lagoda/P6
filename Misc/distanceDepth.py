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
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30) #1280x720 30
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  #1024x768 30
pipeline.start(config)

x = 320
y = 240

#align_to = rs.stream.depth
#align = rs.align(align_to)

model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
#model = torch.hub.load('ultralytics/yolov5', 'custom', path='path/to/best1.pt')
time.sleep(5)  
print('Model has been downloaded and created') 
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        
try:
    while True:
        # This call waits until a new coherent set of frames is available on a device
        frames = pipeline.wait_for_frames()
        
        #Aligning color frame to depth frame
        #aligned_frames =  align.process(frames)
        #depth_frame = aligned_frames.get_depth_frame()
        #aligned_color_frame = aligned_frames.get_color_frame()
        color_frame = frames.get_color_frame()

        
        if not color_frame: continue     

        #color_intrin = aligned_color_frame.profile.as_video_stream_profile().intrinsics
        
        color_image = np.asanyarray(color_frame.get_data())
        
        height = color_image.shape[0]
        width = color_image.shape[1]

        result = model(color_image)
        objs = result.pandas().xyxy[0]
        objs_name = objs.loc[objs['name'] == 'bottle'] #bottle #weed
    
        obj = objs_name.iloc[0]
        x_middle = obj.xmin + (obj.xmax-obj.xmin)/2
        y_middle = obj.ymin + (obj.ymax-obj.ymin)/2
    
        
        print("1")
        depth_frame = frames.get_depth_frame()
        print("2")
        depth_image = np.asanyarray(depth_frame.get_data())
        print("3")
        depth = depth_image.get_distance(x, y)
        
        #dx ,dy, dz = rs.rs2_deproject_pixel_to_point(depth_frame, [x,y], depth)
        #distance = math.sqrt(((dx)**2) + ((dy)**2) + ((dz)**2))
        
        #print('Distance from camera to pixel:' + '\t' +str(round(distance, 2 )))
        print("Z-depth from camera surface to pixel surface:", depth)
        
        #cv2.rectangle(color_image, (int(obj.xmin), int(obj.ymin)), (int(obj.xmax), int(obj.ymax)), (0,255,0),2)
        #cv2.circle(color_image, (int(x_middle), int(y_middle)), 5, (0, 255, 0), 2)
        #cv2.circle(color_image, (int(width/2), int(height/2)), 5, (0, 0, 255), 2)
        #cv2.line(color_image, (int(x_middle), int(y_middle)), (int(width/2), int(height/2)), (0,0,255), 2)

        
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