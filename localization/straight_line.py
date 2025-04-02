#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker
from rcl_interfaces.msg import SetParametersResult


class StraightLine(Node):

    def __init__(self):
        super().__init__("straight_line")
        self.declare_parameter("drive_topic", "/drive")
        self.DRIVE_TOPIC = self.get_parameter('drive_topic').get_parameter_value().string_value
        self.simple_publisher = self.create_publisher(AckermannDriveStamped, self.DRIVE_TOPIC, 10)
        timer_period = 1
        self.i =0
        self.angle = 0.0
        self.distance = 0.0
        self.timer = self.create_timer(timer_period, self.drive_straight)

    def drive_straight(self):
        driveCommand = AckermannDriveStamped()
        driveCommand.header.frame_id = "base_link"
        driveCommand.header.stamp = self.get_clock().now().to_msg()
        driveCommand.drive.steering_angle= 0. #self.angle
        driveCommand.drive.speed= 1.0 #self.VELOCITY
        self.simple_publisher.publish(driveCommand)

def main():
    rclpy.init()
    straight_line = StraightLine()
    rclpy.spin(straight_line)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
    