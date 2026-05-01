#!/usr/bin/env python3

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from nav_msgs.msg import OccupancyGrid, MapMetaData
from cv_bridge import CvBridge
from rclpy.qos import qos_profile_sensor_data

class EgoLanesIPMNode(Node):
    def __init__(self):
        super().__init__("egolanes_ipm_node")

        # Declare parameters
        self.declare_parameter("mask_topic", "/perception/lane_detection/mask")
        self.declare_parameter("costmap_topic", "/perception/lane_detection/costmap")
        self.declare_parameter("vis_topic", "/perception/lane_detection/ipm_vis")
        
        # Default source points (Trapezoid in camera image, flat array)
        default_src = [300.0, 250.0, 500.0, 250.0, 750.0, 400.0, 50.0, 400.0]
        # Default destination points (Rectangle in BEV image, flat array)
        default_dst = [100.0, 0.0, 300.0, 0.0, 300.0, 400.0, 100.0, 400.0]
        
        # In ROS2, parameters cannot be nested lists. We use flat lists of floats.
        self.declare_parameter("src_points", default_src)
        self.declare_parameter("dst_points", default_dst)
        
        self.declare_parameter("grid_resolution", 0.05)
        self.declare_parameter("grid_width", 400)
        self.declare_parameter("grid_height", 400)
        self.declare_parameter("publish_vis", True)

        # Get parameters
        mask_topic = self.get_parameter("mask_topic").value
        costmap_topic = self.get_parameter("costmap_topic").value
        vis_topic = self.get_parameter("vis_topic").value
        
        src_pts_param = self.get_parameter("src_points").value
        dst_pts_param = self.get_parameter("dst_points").value
        
        self.grid_resolution = float(self.get_parameter("grid_resolution").value)
        self.grid_width = int(self.get_parameter("grid_width").value)
        self.grid_height = int(self.get_parameter("grid_height").value)
        self.publish_vis = bool(self.get_parameter("publish_vis").value)

        # Process points into numpy arrays
        # We handle flat lists of 8 floats and reshape them to 4x2
        try:
            self.src_pts = np.array(src_pts_param, dtype=np.float32).reshape(4, 2)
            self.dst_pts = np.array(dst_pts_param, dtype=np.float32).reshape(4, 2)
        except Exception as e:
            self.get_logger().error(f"Failed to parse src_points/dst_points: {e}. Using defaults.")
            self.src_pts = np.array(default_src, dtype=np.float32).reshape(4, 2)
            self.dst_pts = np.array(default_dst, dtype=np.float32).reshape(4, 2)

        # Compute Perspective Transform Matrix
        self.M = cv2.getPerspectiveTransform(self.src_pts, self.dst_pts)

        self.bridge = CvBridge()
        
        # Subscribers and Publishers
        self.sub = self.create_subscription(
            Image,
            mask_topic,
            self.mask_callback,
            qos_profile_sensor_data
        )
        self.costmap_pub = self.create_publisher(
            OccupancyGrid,
            costmap_topic,
            qos_profile_sensor_data
        )
        if self.publish_vis:
            self.vis_pub = self.create_publisher(
                Image,
                vis_topic,
                qos_profile_sensor_data
            )

        self.get_logger().info(f"EgoLanesIPMNode ready | Grid: {self.grid_width}x{self.grid_height} at {self.grid_resolution}m/px")

    def mask_callback(self, msg: Image):
        # Convert ROS Image to OpenCV mono8
        mask_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")

        # Apply IPM Transform
        bev_img = cv2.warpPerspective(
            mask_img, 
            self.M, 
            (self.grid_width, self.grid_height), 
            flags=cv2.INTER_NEAREST, 
            borderMode=cv2.BORDER_CONSTANT, 
            borderValue=0
        )

        # Convert to OccupancyGrid
        # Autoware expects 100 for obstacle (lane), 0 for free space
        # Our mask classes: 0=bg, 1=left, 2=right, 3=other
        occupancy_data = np.zeros_like(bev_img, dtype=np.int8)
        
        # Set all lane pixels (1, 2) to 100
        # Optional: You could choose to only map ego lanes (1, 2) or all lanes (1, 2, 3)
        lane_pixels = (bev_img == 1) | (bev_img == 2) | (bev_img == 3)
        occupancy_data[lane_pixels] = 100

        grid_msg = OccupancyGrid()
        grid_msg.header = msg.header
        grid_msg.header.frame_id = "base_link" # Costmaps are usually in base_link
        
        grid_msg.info = MapMetaData()
        grid_msg.info.resolution = self.grid_resolution
        grid_msg.info.width = self.grid_width
        grid_msg.info.height = self.grid_height
        
        # Origin sets the position of the bottom-left corner of the grid in base_link coordinates.
        # Assuming the camera is at (0,0) in base_link, and pointing forward (X is forward, Y is left).
        # We center the grid in Y:
        grid_msg.info.origin.position.x = 0.0
        grid_msg.info.origin.position.y = -(self.grid_width * self.grid_resolution) / 2.0
        grid_msg.info.origin.position.z = 0.0
        
        # Identity orientation
        grid_msg.info.origin.orientation.x = 0.0
        grid_msg.info.origin.orientation.y = 0.0
        grid_msg.info.origin.orientation.z = 0.0
        grid_msg.info.origin.orientation.w = 1.0

        # ROS OccupancyGrid data is a flat list, row-major, starting from bottom-left
        # OpenCV image is row-major, starting from top-left.
        # We need to flip the image vertically to match ROS grid coordinates
        flipped_data = np.flipud(occupancy_data)
        grid_msg.data = flipped_data.flatten().tolist()

        self.costmap_pub.publish(grid_msg)

        # Publish Visualization
        if self.publish_vis:
            # Create a nice RGB image for the BEV
            vis_img = np.zeros((self.grid_height, self.grid_width, 3), dtype=np.uint8)
            vis_img[bev_img == 1] = [255, 0, 0]   # Left (Red)
            vis_img[bev_img == 2] = [0, 255, 0]   # Right (Green)
            vis_img[bev_img == 3] = [0, 0, 255]   # Other (Blue)
            
            vis_msg = self.bridge.cv2_to_imgmsg(vis_img, encoding="rgb8")
            vis_msg.header = msg.header
            self.vis_pub.publish(vis_msg)

def main(args=None):
    rclpy.init(args=args)
    node = EgoLanesIPMNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()
