from __future__ import print_function
import pyrealsense2 as rs
import numpy as np
import cv2
import argparse
import datetime as dt

# Create a pipeline
pipeline = rs.pipeline()

# Create a config and configure the pipeline to stream
#  different resolutions of color and depth streams
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

#---------------------------------------------------------------

# Args Parsing
# to record and playback from command line
# python3 cv_demos.py --record_pb record --file_name 'test.mp4'
# python3 cv_demos.py --record_pb playback --file_name 'test.mp4'

# parser = argparse.ArgumentParser(description='Start/stop recording data.')
# parser.add_argument('--record_pb', type=str, required=True)
# parser.add_argument('--file_name', type=str, required=True)
# args = parser.parse_args()
# if args.record_pb.lower() == 'record':
#     # now = dt.datetime.now()
#     # file_time = now.strftime('%m-%d-%Y_%H:%M:%S')
#     # filename = f'CV_{file_time}.mov'
#     filename = f'{args.file_name}'
#     config.enable_record_to_file(filename)
# if args.record_pb == 'playback':
#     config.enable_device_from_file(args.file_name)

#---------------------------------------------------------------

# Trackbars
alpha_slider_max = 100
trackbar_name = 'Alpha x %d' % alpha_slider_max
title_window = 'Trackbars'

def on_trackbar(val):
    alpha = val / alpha_slider_max
    beta = ( 1.0 - alpha )
    dst = cv2.addWeighted(images, alpha, images, beta, 0.0)
    cv2.imshow(title_window, dst)

cv2.createTrackbar(trackbar_name, title_window, 9, alpha_slider_max, on_trackbar)


#---------------------------------------------------------------


#---------------------------------------------------------------

#---------------------------------------------------------------

#---------------------------------------------------------------


# Start streaming
profile = pipeline.start(config)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)

# We will be removing the background of objects more than
#  clipping_distance_in_meters meters away
clipping_distance_in_meters = 1 #1 meter
clipping_distance = clipping_distance_in_meters / depth_scale

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)


## Streaming loop
try:
    while True:
        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()
        # frames.get_depth_frame() is a 640x360 depth image

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Remove background - Set pixels further than clipping_distance to grey
        grey_color = 153
        depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
        bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)

        # Render images:
        #   depth align to color on left
        #   depth on right
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        images = np.hstack((bg_removed, depth_colormap))

        # showing just color_image
        images = color_image

        # Thresholding
        hsv = cv2.cvtColor(images, cv2.COLOR_BGR2HSV)

        lower = np.array([110, 106, 8], np.uint8)
        upper = np.array([142, 255, 255], np.uint8)

        mask = cv2.inRange(hsv, lower, upper)
        res = cv2.bitwise_and(images, images, mask = mask)

        # Contours
        # imgray = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)
        # ret, thresh = cv2.threshold(imgray, 127, 255, 0)

        ret, thresh = cv2.threshold(mask, 127, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # cv2.drawContours(images, contours, -1, (0,255,0), 3)

        areas = [cv2.contourArea(c) for c in contours]
        max_idx = np.argmax(areas)
        cnt = contours[max_idx]
        cv2.drawContours(images, [cnt], 0, (0,255,0), 3)

        # M = cv2.moments(cnt)
        centroid_x = int(M['m10']/M['m00'])
        centroid_y = int(M['m01']/M['m00'])
        cv2.circle(images, (centroid_x, centroid_y), 10, (255,0,0), -1)

    
        # Display
        cv2.imshow('frame', images)
        cv2.imshow('mask', mask)
        # cv2.imshow('res', res)
    
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break

finally:
    pipeline.stop()