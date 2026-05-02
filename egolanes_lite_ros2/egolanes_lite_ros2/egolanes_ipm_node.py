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

        self.declare_parameter("mask_topic",    "/perception/lane_detection/mask")
        self.declare_parameter("costmap_topic", "/perception/lane_detection/costmap")
        self.declare_parameter("vis_topic",     "/perception/lane_detection/ipm_vis")

        # Flat float arrays — reshaped to (4,2) internally.
        # Order: Top-Left, Top-Right, Bottom-Right, Bottom-Left
        default_src = [300.0, 250.0, 500.0, 250.0, 750.0, 400.0, 50.0, 400.0]
        default_dst = [100.0, 0.0,   300.0, 0.0,   300.0, 400.0, 100.0, 400.0]
        self.declare_parameter("src_points",     default_src)
        self.declare_parameter("dst_points",     default_dst)

        self.declare_parameter("grid_resolution", 0.05)
        self.declare_parameter("grid_width",      400)
        self.declare_parameter("grid_height",     400)
        self.declare_parameter("publish_vis",     True)

        # FIX: expose grid origin as a parameter so integrators can correctly
        # position the grid relative to base_link without editing source code.
        # origin_x: how far forward (metres) the front edge of the grid is.
        # origin_y_offset: lateral shift from centre (usually 0).
        # By default the grid spans from the car position forward (origin_x = 0
        # puts the bottom-left corner at the car; set negative to look behind).
        self.declare_parameter("origin_x", 0.0)
        self.declare_parameter("origin_y_offset", 0.0)

        mask_topic    = self.get_parameter("mask_topic").value
        costmap_topic = self.get_parameter("costmap_topic").value
        vis_topic     = self.get_parameter("vis_topic").value

        src_pts_param = self.get_parameter("src_points").value
        dst_pts_param = self.get_parameter("dst_points").value

        self.grid_resolution = float(self.get_parameter("grid_resolution").value)
        self.grid_width      = int(self.get_parameter("grid_width").value)
        self.grid_height     = int(self.get_parameter("grid_height").value)
        self.publish_vis     = bool(self.get_parameter("publish_vis").value)
        self._origin_x       = float(self.get_parameter("origin_x").value)
        self._origin_y_off   = float(self.get_parameter("origin_y_offset").value)

        try:
            self.src_pts = np.array(src_pts_param, dtype=np.float32).reshape(4, 2)
            self.dst_pts = np.array(dst_pts_param, dtype=np.float32).reshape(4, 2)
        except Exception as e:
            self.get_logger().error(
                f"Failed to parse src_points/dst_points: {e}. Using defaults."
            )
            self.src_pts = np.array(default_src, dtype=np.float32).reshape(4, 2)
            self.dst_pts = np.array(default_dst, dtype=np.float32).reshape(4, 2)

        self.M = cv2.getPerspectiveTransform(self.src_pts, self.dst_pts)

        # Pre-build the static parts of OccupancyGrid so we only fill data each frame.
        self._grid_template = self._build_grid_template()

        self.bridge = CvBridge()

        self.sub = self.create_subscription(
            Image, mask_topic, self.mask_callback, qos_profile_sensor_data
        )
        self.costmap_pub = self.create_publisher(
            OccupancyGrid, costmap_topic, qos_profile_sensor_data
        )

        # FIX: guard vis_pub creation — only create if needed, and check with
        # hasattr before publishing to survive runtime param changes.
        if self.publish_vis:
            self.vis_pub = self.create_publisher(
                Image, vis_topic, qos_profile_sensor_data
            )
            self._vis_buf = np.zeros((self.grid_height, self.grid_width, 3), dtype=np.uint8)

        self.get_logger().info(
            f"EgoLanesIPMNode ready | Grid: {self.grid_width}x{self.grid_height} "
            f"at {self.grid_resolution}m/px | origin_x={self._origin_x}"
        )

    def _build_grid_template(self) -> OccupancyGrid:
        """
        Build a reusable OccupancyGrid with all static fields pre-filled.
        Only .header and .data need to be updated each callback.
        """
        g = OccupancyGrid()

        g.info = MapMetaData()
        g.info.resolution = self.grid_resolution
        g.info.width      = self.grid_width
        g.info.height     = self.grid_height

        # FIX: set map_load_time to avoid log spam in Autoware nodes that check it.
        g.info.map_load_time = rclpy.clock.Clock().now().to_msg()

        # Origin = position of the bottom-left corner of the grid in base_link.
        # ROS convention: X forward, Y left.
        #
        # FIX: the previous code placed origin.x = 0, which means the car sits
        # at the very rear edge of the grid and the planner sees no area behind
        # the car at all. With origin_x = 0 and a 400px × 0.05m grid the grid
        # covers 0 → 20m ahead, which is correct for a forward-only costmap.
        # If you need rear coverage, set origin_x to a negative value.
        g.info.origin.position.x = self._origin_x
        g.info.origin.position.y = (
            -(self.grid_width * self.grid_resolution) / 2.0 + self._origin_y_off
        )
        g.info.origin.position.z = 0.0
        g.info.origin.orientation.w = 1.0   # identity

        return g

    def mask_callback(self, msg: Image) -> None:
        mask_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")

        bev_img = cv2.warpPerspective(
            mask_img,
            self.M,
            (self.grid_width, self.grid_height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        # Build occupancy array: lane pixels → 100, free space → 0.
        # np.int8 matches the OccupancyGrid wire type exactly.
        occupancy_data = np.zeros((self.grid_height, self.grid_width), dtype=np.int8)
        lane_pixels = (bev_img == 1) | (bev_img == 2) | (bev_img == 3)
        occupancy_data[lane_pixels] = 100

        # ROS OccupancyGrid is row-major from bottom-left; OpenCV is top-left.
        flipped = np.flipud(occupancy_data)

        grid_msg = self._grid_template
        grid_msg.header          = msg.header
        grid_msg.header.frame_id = "base_link"

        # FIX: pass bytes directly instead of converting to a Python list.
        # np.ndarray.tobytes() on a contiguous int8 array is ~50x faster than
        # .flatten().tolist() and produces the identical wire format.
        grid_msg.data = flipped.tobytes()

        self.costmap_pub.publish(grid_msg)

        if self.publish_vis and hasattr(self, "vis_pub"):
            self._vis_buf[:] = 0
            self._vis_buf[bev_img == 1] = (255, 0,   0)   # Left  (Red)
            self._vis_buf[bev_img == 2] = (0,   255, 0)   # Right (Green)
            self._vis_buf[bev_img == 3] = (0,   0,   255) # Other (Blue)

            vis_msg        = self.bridge.cv2_to_imgmsg(self._vis_buf, encoding="rgb8")
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
