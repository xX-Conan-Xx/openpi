#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IK Solver for A1 arm with ROS control, wrapped into a class
"""

import rospy
import time
from geometry_msgs.msg import Pose, PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
from signal_arm.msg._gripper_position_control import gripper_position_control
from planner_IK_speed import IKSolver
from tf.transformations import quaternion_from_euler

class A1ArmController:
    def __init__(self, urdf_path, base_link, ee_link):
        """
        初始化机械臂控制器，并设置 ROS 话题
        """
        self.solver = IKSolver(urdf_path, base_link, ee_link)
        self.pub = rospy.Publisher('/arm_joint_target_position', JointState, queue_size=10)
        self.gripper_pub = rospy.Publisher('/gripper_position_control_host', gripper_position_control, queue_size=10)
        rospy.Subscriber('/gripper_stroke_host', JointState, self.gripper_state_callback)
        rospy.Subscriber('/joint_states_host', JointState, self.joint_state_callback)
        self.current_gripper_position = None
        self.current_joint_position = None

    def gripper_state_callback(self, msg):
        if msg.position:
            self.current_gripper_position = msg.position[0]

    def joint_state_callback(self, msg):
        if msg.position:
            self.current_joint_position = msg.position

    def interpolate_and_publish_joint_positions(self, target_position, gripper_target, steps=1, hz=10):
        """ 插值并以指定频率发布 JointState 消息到 /arm_joint_target_position """
        if len(target_position) != 6:
            rospy.logerr("目标关节位置必须是6维列表")
            return

        rospy.init_node('joint_state_publisher', anonymous=True)
        rate = rospy.Rate(hz)

        while self.current_gripper_position is None and not rospy.is_shutdown():
            rospy.sleep(0.1)
        while self.current_joint_position is None and not rospy.is_shutdown():
            rospy.sleep(0.1)

        joint_state = JointState()
        joint_state.header.frame_id = 'world'
        joint_state.name = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        joint_state.velocity = []
        joint_state.effort = []
        joint_state.position = list(self.current_joint_position)

        step_increment = [(target - current) / steps for target, current in zip(target_position, joint_state.position)]

        for step in range(steps):
            if rospy.is_shutdown():
                break
            joint_state.header.stamp = rospy.Time.now()
            joint_state.position = [current + increment for current, increment in zip(joint_state.position, step_increment)]
            self.pub.publish(joint_state)
            rate.sleep()

        gripper_start = self.current_gripper_position
        gripper_steps = 5
        gripper_increment = (gripper_target - gripper_start) / gripper_steps
        gripper_msg = gripper_position_control()
        gripper_msg.header = Header(frame_id="")
        for i in range(gripper_steps + 1):
            if rospy.is_shutdown():
                break
            gripper_msg.header.stamp = rospy.Time.now()
            gripper_msg.gripper_stroke = gripper_start + gripper_increment * i
            self.gripper_pub.publish(gripper_msg)
            rate.sleep()

    def execute_ik_action(self, action):
        if len(action) == 7:
            gripper = min(100, action[6] * 100)
            r, p, y = action[3], action[4], action[5]
            quaternion = quaternion_from_euler(r, p, y)  #x y z w
            joint_solution = self.solver.solve_ik(action[0:3], [quaternion[3], quaternion[0], quaternion[1], quaternion[2]], orientation_mode="Z", max_iter=1000)
            print(f"joint_solution: {joint_solution}")
            self.interpolate_and_publish_joint_positions(joint_solution[1:], gripper)
        elif len(action) == 8:
            gripper = max(0, min(100, action[7] * 100))
            joint_solution = self.solver.solve_ik(action[0:3], action[3:7], orientation_mode="Z", max_iter=1000)
            print(f"joint_solution: {joint_solution}")
            time.sleep(0.1)
            self.interpolate_and_publish_joint_positions(joint_solution[1:], gripper)
