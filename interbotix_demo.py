from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
import modern_robotics as mr

# The robot object is what you use to control the robot
robot = InterbotixManipulatorXS("px100", "arm", "gripper")

mode = 'h'

# find the position of the end effector given the joint states
joints = robot.arm.get_joint_commands()
T = mr.FKinSpace(robot.arm.robot_des.M, robot.arm.robot_des.Slist, joints)
[R, p] = mr.TransToRp(T) # get the rotation matrix and the displacement
# R is rotation matrix
# p is x,y,z coordinates
print(R)
print(p)


# print(robot.arm.get_joint_commands)


# Let the user select the position
while mode != 'q':
    mode=input("[h]ome, [s]leep, [q]uit, [o]pen, [c]lose, [f]orward, [b]ackward, [u]p, [d]own, [r]otate, [w]rist")
    if mode == "h":
        robot.arm.go_to_home_pose()
    elif mode == "s":
        robot.arm.go_to_sleep_pose()
    elif mode == 'o':
        robot.gripper.release(2.0)
    elif mode == 'c':
        robot.gripper.grasp(2.0)
    elif mode == 'f':
        joints_pos = [0, 0.1, -0.1, 0]  # [waist, shoulder, elbow, wrist_angle]
        robot.arm.set_joint_positions(joints_pos)
    elif mode == 'b':
        joints_pos = [0, -0.1, 0.1, 0]
        robot.arm.set_joint_positions(joints_pos)

    elif mode == 'u':
        robot.arm.set_single_joint_position('elbow', -0.1)

    elif mode == 'd':
        robot.arm.set_single_joint_position('elbow', 0.1)

    elif mode == 'r':
        robot.arm.set_single_joint_position('waist', 0.1)

    elif mode == 'w':
        robot.arm.set_single_joint_position('wrist_angle', 0.2)