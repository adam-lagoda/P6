import pyrealsense2 as rs
import numpy as np

class DepthCamera:
    def __init__(self):
        # Source the pipeline for camera streaming and the configuration prefix
        self.pipeline = rs.pipeline()
        config = rs.config()

        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        
        # Make sure the camera is in "close range" mode
        rs.threshold_filter(0.1, 2.7)

        # Configure the camera streaming properties
        config.enable_stream(rs.stream.depth, 1024, 768, rs.format.z16, 30) #1024x768
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30) #640 480 for both
        
        # Start streaming
        self.pipeline.start(config)
        
        # Define local frame alignment functions
        self.align_to = rs.stream.color
        self.align = rs.align(self.align_to)

    def get_frame(self):
        # Wait for frames to arrive
        frames = self.pipeline.wait_for_frames()
        
        # Get the color frame
        color_frame = frames.get_color_frame()

        # Convert the color frame to an array using Numpy
        color_image = np.asanyarray(color_frame.get_data())
        
        if not color_frame:
            return False, None
        return True, color_image
    
    def get_aligned_frame(self):
        # Wait for frames to arrive
        frames = self.pipeline.wait_for_frames()
        
        # Align the depth and color camera together
        aligned_frames = self.align.process(frames)
        
        # Get aligned depth frame
        aligned_depth_frame = aligned_frames.get_depth_frame()
        
        if not aligned_depth_frame:
            return False, None
        return True, aligned_depth_frame
        
    def release(self):
        # Stop streaming from camera
        self.pipeline.stop()