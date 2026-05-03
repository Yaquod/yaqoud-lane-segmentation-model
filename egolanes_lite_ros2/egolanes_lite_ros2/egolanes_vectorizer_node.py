#!/usr/bin/env python3
"""
egolanes_vectorizer_node.py

Converts the mono8 lane mask from EgoLanesLiteNode into metric 3D point clusters
published as PointCloud2 + a MarkerArray of polylines for RViz.

Subscriptions
-------------
  /perception/lane_detection/mask             (sensor_msgs/Image, mono8)
  /sensing/camera/traffic_light/camera_info   (sensor_msgs/CameraInfo)

Publications
------------
  /perception/lane_detection/left_cluster     (sensor_msgs/PointCloud2)
  /perception/lane_detection/right_cluster    (sensor_msgs/PointCloud2)
  /perception/lane_detection/lane_markers     (visualization_msgs/MarkerArray)

Extrinsics strategy
-------------------
  Priority 1 — TF tree (uses Autoware's sensor calibration directly):
      Looks up camera_frame → base_link_frame at startup.
      camera_frame is auto-detected from the first camera_info message.

  Priority 2 — manual params (fallback if TF unavailable):
      camera_height_m + camera_pitch_deg synthesise the transform.

Intrinsics strategy
-------------------
  Priority 1 — live camera_info topic (always preferred).
  Priority 2 — manual camera_fx/fy/cx/cy params.

Parameters
----------
  mask_topic          (str)   /perception/lane_detection/mask
  camera_info_topic   (str)   /sensing/camera/traffic_light/camera_info
  left_topic          (str)   /perception/lane_detection/left_cluster
  right_topic         (str)   /perception/lane_detection/right_cluster
  marker_topic        (str)   /perception/lane_detection/lane_markers
  publish_markers     (bool)  true
  base_link_frame     (str)   base_link
  camera_frame        (str)   "" → auto-detected from camera_info
  use_tf_extrinsics   (bool)  true

  Manual fallbacks (used only if TF lookup fails):
    camera_fx / camera_fy / camera_cx / camera_cy
    camera_height_m     (float)  1.2   metres above ground
    camera_pitch_deg    (float)  0.0   positive = nose-down

  Sampling:
    row_step            (int)    4
    min_pixels_per_row  (int)    5
    max_range_m         (float)  30.0
"""

import math

import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Point
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray

try:
    from tf2_ros import Buffer, TransformListener
    _TF2_AVAILABLE = True
except ImportError:
    _TF2_AVAILABLE = False


# ── helpers ───────────────────────────────────────────────────────────────────

def _quat_to_rotation_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    """Quaternion → 3×3 rotation matrix."""
    n = math.sqrt(x*x + y*y + z*z + w*w)
    if n < 1e-10:
        return np.eye(3)
    x, y, z, w = x/n, y/n, z/n, w/n
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def _rotation_from_height_pitch(
    height_m: float, pitch_rad: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Synthesise camera→base_link R and t from manual params.
    Camera convention: z-forward, x-right, y-down (ROS camera_optical).
    base_link convention: x-forward, y-left, z-up.
    """
    cp, sp = math.cos(pitch_rad), math.sin(pitch_rad)
    R = np.array([
        [ 0,  sp,  cp],
        [-1,   0,   0],
        [ 0, -cp,  sp],
    ], dtype=np.float64)
    t = np.array([0.0, 0.0, height_m], dtype=np.float64)
    return R, t


def _make_pointcloud2(header: Header, points: np.ndarray) -> PointCloud2:
    # FIX: np.ascontiguousarray ensures correct byte layout before tobytes().
    # Non-contiguous slices (e.g. from fancy indexing) produce wrong wire data.
    pts = np.ascontiguousarray(points, dtype=np.float32)
    msg = PointCloud2()
    msg.header       = header
    msg.height       = 1
    msg.width        = len(pts)
    msg.is_dense     = True
    msg.is_bigendian = False
    msg.point_step   = 12
    msg.row_step     = msg.point_step * msg.width
    msg.fields       = [
        PointField(name="x", offset=0,  datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4,  datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8,  datatype=PointField.FLOAT32, count=1),
    ]
    msg.data = pts.tobytes()
    return msg


def _make_line_marker(
    header: Header, points: np.ndarray,
    marker_id: int, r: float, g: float, b: float,
    scale: float = 0.05,
) -> Marker:
    m = Marker()
    m.header = header
    m.ns     = "lane_polylines"
    m.id     = marker_id
    m.type   = Marker.LINE_STRIP
    m.action = Marker.ADD
    m.scale.x = scale
    m.color.r, m.color.g, m.color.b, m.color.a = r, g, b, 1.0
    m.pose.orientation.w = 1.0
    for pt in points:
        p = Point()
        p.x, p.y, p.z = float(pt[0]), float(pt[1]), float(pt[2])
        m.points.append(p)
    return m


def _make_delete_marker(marker_id: int, header: Header) -> Marker:
    m = Marker()
    m.header = header
    m.ns     = "lane_polylines"
    m.id     = marker_id
    m.action = Marker.DELETE
    return m


# ── node ─────────────────────────────────────────────────────────────────────

class EgoLanesVectorizerNode(Node):
    CLASS_LEFT  = 1
    CLASS_RIGHT = 2

    def __init__(self):
        super().__init__("egolanes_vectorizer_node")

        self.declare_parameter("mask_topic",        "/perception/lane_detection/mask")
        self.declare_parameter("camera_info_topic", "/sensing/camera/traffic_light/camera_info")
        self.declare_parameter("left_topic",        "/perception/lane_detection/left_cluster")
        self.declare_parameter("right_topic",       "/perception/lane_detection/right_cluster")
        self.declare_parameter("marker_topic",      "/perception/lane_detection/lane_markers")
        self.declare_parameter("publish_markers",   True)
        self.declare_parameter("base_link_frame",   "base_link")
        self.declare_parameter("camera_frame",      "")
        self.declare_parameter("use_tf_extrinsics", True)
        self.declare_parameter("camera_fx",         600.0)
        self.declare_parameter("camera_fy",         600.0)
        self.declare_parameter("camera_cx",         -1.0)
        self.declare_parameter("camera_cy",         -1.0)
        self.declare_parameter("camera_height_m",   1.2)
        self.declare_parameter("camera_pitch_deg",  0.0)
        self.declare_parameter("row_step",           4)
        self.declare_parameter("min_pixels_per_row", 5)
        self.declare_parameter("max_range_m",        30.0)

        mask_topic        = self.get_parameter("mask_topic").value
        camera_info_topic = self.get_parameter("camera_info_topic").value
        left_topic        = self.get_parameter("left_topic").value
        right_topic       = self.get_parameter("right_topic").value
        marker_topic      = self.get_parameter("marker_topic").value
        self.pub_markers  = bool(self.get_parameter("publish_markers").value)
        self.frame_id     = self.get_parameter("base_link_frame").value
        self._cam_frame   = self.get_parameter("camera_frame").value
        self._use_tf      = bool(self.get_parameter("use_tf_extrinsics").value)

        self._fx_param  = float(self.get_parameter("camera_fx").value)
        self._fy_param  = float(self.get_parameter("camera_fy").value)
        self._cx_param  = float(self.get_parameter("camera_cx").value)
        self._cy_param  = float(self.get_parameter("camera_cy").value)
        self._cam_h     = float(self.get_parameter("camera_height_m").value)
        self._pitch_rad = math.radians(self.get_parameter("camera_pitch_deg").value)

        self.row_step  = int(self.get_parameter("row_step").value)
        self.min_pix   = int(self.get_parameter("min_pixels_per_row").value)
        self.max_range = float(self.get_parameter("max_range_m").value)

        # State — populated asynchronously from camera_info / TF
        self._K: np.ndarray | None = None
        self._R: np.ndarray | None = None
        self._t: np.ndarray | None = None

        # TF setup
        self._tf_buffer   = None
        self._tf_listener = None
        if _TF2_AVAILABLE and self._use_tf:
            self._tf_buffer   = Buffer()
            self._tf_listener = TransformListener(self._tf_buffer, self)

        self.bridge = CvBridge()

        self._info_sub = self.create_subscription(
            CameraInfo, camera_info_topic, self._info_cb, qos_profile_sensor_data
        )
        self.sub = self.create_subscription(
            Image, mask_topic, self._mask_cb, qos_profile_sensor_data
        )
        self.pub_left  = self.create_publisher(PointCloud2, left_topic,  qos_profile_sensor_data)
        self.pub_right = self.create_publisher(PointCloud2, right_topic, qos_profile_sensor_data)
        if self.pub_markers:
            self.pub_viz = self.create_publisher(MarkerArray, marker_topic, qos_profile_sensor_data)

        self.get_logger().info(
            f"EgoLanesVectorizerNode ready | frame={self.frame_id} | use_tf={self._use_tf}"
        )

    # ── camera_info ──────────────────────────────────────────────────────────

    def _info_cb(self, msg: CameraInfo) -> None:
        K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        if K[0, 0] < 1.0:
            return
        self._K = K
        self.get_logger().info(
            f"Intrinsics from camera_info: "
            f"fx={K[0,0]:.1f} fy={K[1,1]:.1f} cx={K[0,2]:.1f} cy={K[1,2]:.1f}"
        )
        if not self._cam_frame:
            self._cam_frame = msg.header.frame_id
            self.get_logger().info(f"Camera frame auto-detected: '{self._cam_frame}'")

        self._try_tf_extrinsics()
        self.destroy_subscription(self._info_sub)

    # ── extrinsics ───────────────────────────────────────────────────────────

    def _try_tf_extrinsics(self) -> None:
        if not (_TF2_AVAILABLE and self._use_tf and self._cam_frame):
            self._use_manual_extrinsics("TF not available or not requested")
            return
        try:
            tf_stamped = self._tf_buffer.lookup_transform(
                self.frame_id,
                self._cam_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=3.0),
            )
        except Exception as e:
            self._use_manual_extrinsics(f"TF lookup failed ({e})")
            return

        tr = tf_stamped.transform.translation
        q  = tf_stamped.transform.rotation
        self._t = np.array([tr.x, tr.y, tr.z], dtype=np.float64)
        self._R = _quat_to_rotation_matrix(q.x, q.y, q.z, q.w)
        self.get_logger().info(
            f"Extrinsics from TF ({self._cam_frame} → {self.frame_id}): "
            f"t=[{self._t[0]:.3f}, {self._t[1]:.3f}, {self._t[2]:.3f}] m"
        )

    def _use_manual_extrinsics(self, reason: str) -> None:
        self.get_logger().warn(
            f"{reason}. Using manual params: "
            f"height={self._cam_h}m pitch={math.degrees(self._pitch_rad):.1f}deg"
        )
        self._R, self._t = _rotation_from_height_pitch(self._cam_h, self._pitch_rad)

    # ── mask callback ────────────────────────────────────────────────────────

    def _mask_cb(self, msg: Image) -> None:
        if self._R is None:
            self._use_manual_extrinsics("Extrinsics not yet resolved — using manual fallback")

        mask  = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
        h, w  = mask.shape

        if self._K is not None:
            fx, fy = self._K[0, 0], self._K[1, 1]
            cx, cy = self._K[0, 2], self._K[1, 2]
        else:
            fx = self._fx_param
            fy = self._fy_param
            cx = self._cx_param if self._cx_param > 0 else w / 2.0
            cy = self._cy_param if self._cy_param > 0 else h / 2.0

        left_pts  = self._extract_lane_points(mask, self.CLASS_LEFT,  h, w, fx, fy, cx, cy)
        right_pts = self._extract_lane_points(mask, self.CLASS_RIGHT, h, w, fx, fy, cx, cy)

        header          = msg.header
        header.frame_id = self.frame_id

        def _pub(pub, pts):
            arr = np.array(pts, dtype=np.float32) if pts else np.zeros((0, 3), dtype=np.float32)
            pub.publish(_make_pointcloud2(header, arr))

        _pub(self.pub_left,  left_pts)
        _pub(self.pub_right, right_pts)

        if self.pub_markers:
            ma = MarkerArray()
            ma.markers.append(
                _make_line_marker(header, np.array(left_pts),  0, 1.0, 0.0, 0.0)
                if left_pts else _make_delete_marker(0, header)
            )
            ma.markers.append(
                _make_line_marker(header, np.array(right_pts), 1, 0.0, 1.0, 0.0)
                if right_pts else _make_delete_marker(1, header)
            )
            self.pub_viz.publish(ma)

    # ── geometry ─────────────────────────────────────────────────────────────

    def _extract_lane_points(
        self,
        mask: np.ndarray,
        class_id: int,
        img_h: int, img_w: int,
        fx: float, fy: float, cx: float, cy: float,
    ) -> list:
        lane_mask = (mask == class_id)
        R, t = self._R, self._t
        points = []

        for row in range(img_h - 1, 0, -self.row_step):
            col_indices = np.where(lane_mask[row, :])[0]
            if len(col_indices) < self.min_pix:
                continue

            u = float(col_indices.mean())
            v = float(row)

            # Back-project pixel → normalised ray in camera optical frame
            d_cam = np.array([(u - cx) / fx, (v - cy) / fy, 1.0], dtype=np.float64)

            # Rotate ray into base_link frame
            d_bl = R @ d_cam

            # Intersect ray with ground plane (z = 0 in base_link).
            # Ray: P = t + s * d_bl.  Ground: P.z = 0 → s = -t.z / d_bl.z
            if abs(d_bl[2]) < 1e-6:
                continue
            s = -t[2] / d_bl[2]
            if s < 0:
                continue

            pt = t + s * d_bl
            if math.hypot(pt[0], pt[1]) > self.max_range:
                continue

            points.append((float(pt[0]), float(pt[1]), 0.0))

        return points


# ── entry point ───────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = EgoLanesVectorizerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
