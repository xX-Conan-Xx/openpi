#!/usr/bin/env python
import rospy
import time
from geometry_msgs.msg import Pose, PoseArray, PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
from signal_arm.msg._gripper_position_control import gripper_position_control
from planner_IK import solve_ik_for_urdf
# from planner_IK_speed import solve_ik_xyz_quat

'''
A1机械臂控制:执行一个动作或者一系列动作
Author: dhy
'''

def create_pose(x, y, z, w, ox, oy, oz):
    """
    创建一个Pose消息对象
    """
    pose = Pose()
    pose.position.x = x
    pose.position.y = y
    pose.position.z = z
    pose.orientation.w = w
    pose.orientation.x = ox
    pose.orientation.y = oy
    pose.orientation.z = oz
    return pose

current_gripper_position = None
current_joint_position = None

def gripper_state_callback(msg):
    global current_gripper_position
    if msg.position:
        current_gripper_position = msg.position[0]
        # print(f"当前夹爪位置: {current_gripper_position:.2f}")

def joint_state_callback(msg):
    global current_joint_position
    if msg.position:
        current_joint_position = msg.position
        # print(f"当前夹爪位置: {current_gripper_position:.2f}")  

def interpolate_and_publish_joint_positions(target_position, gripper_target, steps=1, hz=10):
    """
    插值并以指定频率发布 JointState 消息到 /arm_joint_target_position

    :param target_position: 目标关节角度位置，长度为6的列表 [rad]
    :param steps: 插值步数（默认 100）
    :param hz: 发布频率 Hz（默认 10Hz）
    """
    global current_gripper_position
    
    if len(target_position) != 6:
        rospy.logerr("目标关节位置必须是6维列表")
        return
    
    # breakpoint()

    rospy.init_node('joint_state_publisher', anonymous=True)
    pub = rospy.Publisher('/arm_joint_target_position', JointState, queue_size=10)
    gripper_pub = rospy.Publisher('/gripper_position_control_host', gripper_position_control, queue_size=10)
    rospy.Subscriber('/gripper_stroke_host', JointState, gripper_state_callback)
    rospy.Subscriber('/joint_states_host', JointState, joint_state_callback)
    
    rate = rospy.Rate(hz)
    
    # rospy.loginfo("wait for receiving: /gripper_stroke_host ...")
    while current_gripper_position is None and not rospy.is_shutdown():
        rospy.sleep(0.1)
        
    # rospy.loginfo(f"received current gripper stroke: {current_gripper_position:.2f}")
    while current_joint_position is None and not rospy.is_shutdown():
        rospy.sleep(0.1)
    
    joint_state = JointState()
    joint_state.header.frame_id = 'world'
    joint_state.name = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
    joint_state.velocity = []
    joint_state.effort = []
    joint_state.position = list(current_joint_position)  # 使用实际关节位置作为初始位置

    step_increment = [
        (target - current) / steps for target, current in zip(target_position, joint_state.position)
    ]
    # rospy.loginfo(f"Step increment: {step_increment}")

    # rospy.loginfo("wait for connect...")
    while pub.get_num_connections() == 0 and not rospy.is_shutdown():
        rospy.sleep(0.1)

    # rospy.loginfo("connected and publishing: JointState...")
    start_time = time.time()
    for step in range(steps):
        if rospy.is_shutdown():
            break
        joint_state.header.stamp = rospy.Time.now()
        joint_state.position = [
            current + increment for current, increment in zip(joint_state.position, step_increment)
        ]
        pub.publish(joint_state)
        rate.sleep()
    print(f"JointState interpolation time: {(time.time() - start_time) * 1000:.2f} ms")

    # rospy.loginfo("JointState published!")
    
    # ---------------------------------------------------
    # print("!!!!Now Gripper...")
    # breakpoint()
    
    gripper_start = current_gripper_position
    gripper_steps = 5
    gripper_increment = (gripper_target - gripper_start) / gripper_steps
    gripper_msg = gripper_position_control()
    gripper_msg.header = Header(frame_id="")
    start_time = time.time()
    for i in range(gripper_steps + 1):
        if rospy.is_shutdown():
            break
        gripper_msg.header.stamp = rospy.Time.now()
        gripper_msg.gripper_stroke = gripper_start + gripper_increment * i
        gripper_pub.publish(gripper_msg)
        # rospy.loginfo(f"publishing gripper stroke: {gripper_msg.gripper_stroke:.2f}")
        rate.sleep()
    print(f"Gripper interpolation time: {(time.time() - start_time) * 1000:.2f} ms")

    # rospy.loginfo("Gripper interpolation finished!")
    


def execute_eef_IK(action):
    # """
    # 接受单个动作（一个包含7个数值的元组: (x, y, z, w, ox, oy, oz)），
    # 将其封装成 PoseStamped 消息，发布到 ROS 话题 '/a1_ee_target'，
    # 执行完后返回 True。
    # """
    # if not rospy.core.is_initialized():
    #     rospy.init_node('pose_publisher', anonymous=True)
    
    # from geometry_msgs.msg import PoseStamped
    from tf.transformations import quaternion_from_euler
    # pub = rospy.Publisher('/a1_ee_target', PoseStamped, queue_size=10)
    # rospy.sleep(1)  # 等待订阅者连
    
    if len(action) == 7:
        gripper = action[6] # TODO
        gripper = min(100, gripper * 100)
        # Convert roll, pitch, yaw (r, p, y) to quaternion (w, ox, oy, oz)
        r, p, y = action[3], action[4], action[5]
        quaternion = quaternion_from_euler(r, p, y) #x y z w

        # Update action with quaternion values
        action = (action[0], action[1], action[2], quaternion[0], quaternion[1], quaternion[2], quaternion[3]) # debug
        # print("modified action is:",action) 
        # breakpoint()
        
        start_time = time.time()
        joint_solution = solve_ik_for_urdf(
            urdf_file = "/home/luka/A1_SDK/install/share/mobiman/urdf/A1/urdf/A1_URDF_0607_0028.urdf", 
            base_link = "base_link", 
            ee_link = "arm_seg6", 
            target_position = action[0:3], 
            target_orientation = action[3:]
        )
        print(f"🌟 IK 耗时: {(time.time() - start_time) * 1000:.2f} ms")

        
        # breakpoint()
        
        target_joint = joint_solution.js_solution.position.cpu().numpy().flatten()
        
        start_exe_time = time.time()
        interpolate_and_publish_joint_positions(
            target_position = target_joint,
            gripper_target = gripper)
        print(f"🌟 执行 耗时: {(time.time() - start_exe_time) * 1000:.2f} ms")

        
        
    elif len(action) == 8:
        # print("action length is 8")
        joint_solution = solve_ik_for_urdf(
            urdf_file = "/home/luka/A1_SDK/install/share/mobiman/urdf/A1/urdf/A1_URDF_0607_0028.urdf", 
            base_link = "base_link", 
            ee_link = "arm_seg6", 
            target_position = action[0:3], 
            target_orientation = action[3:7]
        )
        
        gripper = action[7] # TODO
        gripper = max(0, min(100, gripper * 100))
        
        target_joint = joint_solution.js_solution.position.cpu().numpy().flatten()
        print("target joint is:",target_joint)
        print("gripper target is:",gripper)
        interpolate_and_publish_joint_positions(
            target_position = target_joint,
            gripper_target = gripper)
        
    # rospy.loginfo("Published PoseStamped message to /a1_ee_target")
    return True
