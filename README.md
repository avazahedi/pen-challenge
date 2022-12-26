# Pen Challenge    

## Author: Ava Zahedi  

## Description
As part of Northwestern's MSR orientation, we did a 2-week hackathon to develop our skills in Python, Linux, and Git.  
One project within this was programming a PincherX 100 desktop robot arm to grasp a purple Northwestern pen and return it to a home position. Controlling the arm was done with Python and the Interbotix ROS packages. For detecting the pen, we used OpenCV and RealSense cameras to detect the pen by color, calculate its location relative to the base of the robot, and send desired end-effector coordinates to the robot for movement.  

More information regarding this pen challenge can be found on the [MSR Hackathon Pen Challenge](https://nu-msr.github.io/hackathon/pen_challenge.html) page.  

## Hardware
1. RealSense camera
2. PincherX 100 arm
3. Purple Northwestern pen

## Setup, Calibration, and Usage
1. Position the camera in a way such that it can see the space around the PincherX 100.
2. In one terminal, run `ros2 launch interbotix_xsarm_control xsarm_control.launch.py robot_model:=px100`.
3. In another terminal, run `python3 interbotix_demo.py`.
4. Follow the instructions in the terminal to open the grippers and put the pen between them. Close the grippers to hold the pen.
5. Exit interbotix_demo.
6. In the robot_pen.py code, uncomment line 175, which prints the pen's coordinates with respect to the camera.
7. Run `python3 robot_pen.py`.
8. Use the printed output for the pen's coordinates with respect to the camera to hard-code the P_cx, P_cy, P_cd values inside the calibration section of robot_pen.py.
9. Comment out line 175 in robot_pen.py.
10. In the robot_pen.py code, uncomment the calibration code.
11. Run `python3 robot_pen.py` once more.
12. Use the printed output for O_cx, O_cy, O_cz to hard-code these values under the calibration section of robot_pen.py.
13. Comment out the calibration code in robot_pen.py. This concludes the calibration.
14. Run `python3 robot_pen.py` to have the robot detect, grasp, and return the pen.  

Note: this code can be modified to detect a different range of colors by editing lines 145 and 146 (under Thresholding in the Streaming Loop) in robot_pen.py with the desired lower and upper HSV values.