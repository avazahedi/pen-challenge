from __future__ import print_function
import pyrealsense2 as rs
import numpy as np
import cv2
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
import modern_robotics as mr

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


# returns pen coords wrt camera in meters
def vision():
    # Start streaming
    cfg = pipeline.start(config)

    profile = cfg.get_stream(rs.stream.color)
    intr = profile.as_video_stream_profile().get_intrinsics()

    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_sensor = cfg.get_device().first_depth_sensor()
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
    running = True
    while running:
       
        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()
        # frames.get_depth_frame() is a 640x360 depth image

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        # depth frame to get pixel distance later
        dpt_frame = aligned_depth_frame.as_depth_frame()

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

        # Thresholding
        hsv = cv2.cvtColor(images, cv2.COLOR_BGR2HSV)

        lower = np.array([110, 80, 8], np.uint8)
        upper = np.array([142, 180, 255], np.uint8)

        mask = cv2.inRange(hsv, lower, upper)
        res = cv2.bitwise_and(images, images, mask = mask)

        # Contours
        ret, thresh = cv2.threshold(mask, 127, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            areas = [cv2.contourArea(c) for c in contours]
            max_idx = np.argmax(areas)
            cnt = contours[max_idx]
            cv2.drawContours(images, [cnt], 0, (0,255,0), 3)

            M = cv2.moments(cnt)
            try:
                # get centroid coords and display it
                centroid_x = int(M['m10']/M['m00'])
                centroid_y = int(M['m01']/M['m00'])
                cv2.circle(images, (centroid_x, centroid_y), 10, (255,0,0), -1)
                print(f'centroid found: {centroid_x}, {centroid_y}') 

                # get depth of centroid in pixel coordinates
                # pixel_distance_in_meters
                pen_depth = dpt_frame.get_distance(centroid_x, centroid_y)

                # get location of centroid wrt camera in cartesian coordinates
                pen_coords_wrt_camera = rs.rs2_deproject_pixel_to_point(intr, [centroid_x, centroid_y], pen_depth)
                # intrinsics - the intrinsic parameters
                # (px, py) - the pixel coordinates
                # depth - the depth in meters
                # returns the x,y, and z coordinates in meters as a list

                running = False
                pipeline.stop()

            except:
                pass
    
    return pen_coords_wrt_camera



# The robot object is what you use to control the robot
robot = InterbotixManipulatorXS("px100", "arm", "gripper")

# # Calibration
# robot.arm.go_to_home_pose()
# import time
# time.sleep(8)  # 8s to get pen in position for robot to grab
# robot.gripper.grasp(2.0)
# pen_coords_wrt_camera = vision()


# move arm to starting position
robot.arm.go_to_home_pose()

# open grippers
# robot.gripper.release(2.0)

# measure pen location
pen_coords_wrt_camera = vision()
print(f'pen coords wrt camera (m): {pen_coords_wrt_camera}')

P_cx, P_cy, P_cd = pen_coords_wrt_camera[0], pen_coords_wrt_camera[1], pen_coords_wrt_camera[2]

# given the location of the pincherX relative to the camera frame
# can convert camera (x,y,z) coordinates into cylindrical coordinates
# cenetered at the base frame of the pincherX


# find the position of the end effector given the joint states
joints = robot.arm.get_joint_commands()
T = mr.FKinSpace(robot.arm.robot_des.M, robot.arm.robot_des.Slist, joints)
[R, p] = mr.TransToRp(T) # get the rotation matrix and the displacement
print(f'pen coords wrt robot base: {p}')
# pen coords wrt robot base
P_rx, P_ry, P_rz = p[0], p[1], p[2]


O_cy = P_ry + P_cd
O_cx = P_rx + P_cx
O_cz = P_rz + P_cy


# turn at waist until end-effector is facing the pen
robot.arm.set_single_joint_position('waist', -0.1)

# move forward until pen is inside grippers
joints_pos = [0, 0.1, -0.1, 0]  # [waist, shoulder, elbow, wrist_angle] relative posns
robot.arm.set_joint_positions(joints_pos)

# # close grippers
# robot.gripper.grasp(2.0)

# move arm to starting position
robot.arm.go_to_home_pose()

# move arm to sleep position
robot.arm.go_to_sleep_pose()