#!/usr/bin/env python3

from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from rclpy.qos import qos_profile_sensor_data

class EgoLanesLiteNode(Node):
    def __init__(self):
        super().__init__("egolanes_lite_node")

        self.declare_parameter("model_path", "EgoLanesLite_best.onnx")
        self.declare_parameter("input_h", 400)
        self.declare_parameter("input_w", 800)
        self.declare_parameter("threshold", 0.0)
        self.declare_parameter(
            "mean",
            [0.485, 0.456, 0.406],
        )
        self.declare_parameter(
            "std",
            [0.229, 0.224, 0.225],
        )
        self.declare_parameter("image_topic", "/sensing/camera/traffic_light/image_raw")
        self.declare_parameter("mask_topic", "/perception/lane_detection/mask")
        self.declare_parameter("mask_vis_topic", "/perception/lane_detection/mask_vis")
        self.declare_parameter("use_cuda", True)

        model_path = self.get_parameter("model_path").value
        self.input_h = int(self.get_parameter("input_h").value)
        self.input_w = int(self.get_parameter("input_w").value)
        self.threshold = float(self.get_parameter("threshold").value)
        self.mean = np.array(self.get_parameter("mean").value, dtype=np.float32)
        self.std = np.array(self.get_parameter("std").value, dtype=np.float32)
        if self.mean.shape != (3,) or self.std.shape != (3,):
            raise ValueError("Parameters 'mean' and 'std' must be length-3 arrays.")
        image_topic = self.get_parameter("image_topic").value
        mask_topic = self.get_parameter("mask_topic").value
        mask_vis_topic = self.get_parameter("mask_vis_topic").value
        use_cuda = bool(self.get_parameter("use_cuda").value)

        providers = []
        available_providers = ort.get_available_providers()
        if use_cuda and "CUDAExecutionProvider" in available_providers:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

        resolved_model_path = self._resolve_model_path(model_path)
        self.session = ort.InferenceSession(
            str(resolved_model_path),
            providers=providers,
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self._align_input_size_with_model()

        self.bridge = CvBridge()
        # self.sub = self.create_subscription(Image, image_topic, self.image_callback, 10)
        self.sub = self.create_subscription(
            Image,
            image_topic,
            self.image_callback,
            qos_profile_sensor_data
        )
        # self.pub = self.create_publisher(Image, mask_topic, 10)
        self.pub = self.create_publisher(
            Image,
            mask_topic,
            qos_profile_sensor_data
        )
        self.vis_pub = self.create_publisher(
            Image,
            mask_vis_topic,
            qos_profile_sensor_data
        )

        self.get_logger().info(
            f"EgoLanesLite ROS2 node ready | model={resolved_model_path} | providers={providers} "
            f"| input={self.input_w}x{self.input_h}"
        )

    @staticmethod
    def _resolve_model_path(model_path: str) -> Path:
        path = Path(model_path).expanduser()
        candidates = [path]
        if not path.is_absolute():
            cwd_candidate = Path.cwd() / path
            candidates.append(cwd_candidate)

            source_file = Path(__file__).resolve()
            candidates.extend(parent / path for parent in source_file.parents)

        checked = []
        for candidate in candidates:
            resolved_candidate = candidate.resolve()
            checked.append(str(resolved_candidate))
            if resolved_candidate.is_file():
                return resolved_candidate

        raise FileNotFoundError(
            f"ONNX model not found: {model_path}. Checked: {checked}"
        )

    @staticmethod
    def _is_positive_int(value) -> bool:
        return isinstance(value, int) and value > 0

    def _align_input_size_with_model(self) -> None:
        shape = self.session.get_inputs()[0].shape
        if len(shape) != 4:
            return

        model_h, model_w = shape[2], shape[3]
        if self._is_positive_int(model_h) and self._is_positive_int(model_w):
            if (self.input_h, self.input_w) != (model_h, model_w):
                self.get_logger().warn(
                    "Configured input size "
                    f"{self.input_w}x{self.input_h} does not match model size "
                    f"{model_w}x{model_h}. Using model size."
                )
                self.input_h = model_h
                self.input_w = model_w

    def _preprocess(self, frame_rgb: np.ndarray) -> np.ndarray:
        image = cv2.resize(
            frame_rgb,
            (self.input_w, self.input_h),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32)
        image = image / 255.0
        image = (image - self.mean) / self.std
        return image.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)

    def _postprocess_to_mono8(
        self, raw_output: np.ndarray, out_h: int, out_w: int
    ) -> np.ndarray:
        if raw_output.ndim == 3:
            channel_first = raw_output.shape[0] <= 8
            logits = raw_output if channel_first else raw_output.transpose(2, 0, 1)

            if logits.shape[0] == 3:
                # EgoLanesLite is a 3-channel lane predictor (left, right, other).
                # Convert to a class-id map: background=0, left=1, right=2, other=3.
                lane_mask = logits > self.threshold
                mono = np.zeros(logits.shape[1:], dtype=np.uint8)
                mono[lane_mask[2]] = 3
                mono[lane_mask[1]] = 2
                mono[lane_mask[0]] = 1
            elif logits.shape[0] == 1:
                mono = (logits[0] > self.threshold).astype(np.uint8) * 255
            else:
                mono = np.argmax(logits, axis=0).astype(np.uint8)
        elif raw_output.ndim == 2:
            mono = (raw_output > self.threshold).astype(np.uint8) * 255
        else:
            raise ValueError(f"Unsupported model output shape: {raw_output.shape}")

        if mono.shape != (out_h, out_w):
            mono = cv2.resize(mono, (out_w, out_h), interpolation=cv2.INTER_NEAREST)

        return mono

    def image_callback(self, msg: Image):
        frame_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        tensor = self._preprocess(frame_rgb)

        outputs = self.session.run([self.output_name], {self.input_name: tensor})
        raw_output = outputs[0][0]
        mask_mono8 = self._postprocess_to_mono8(raw_output, msg.height, msg.width)

        out_msg = self.bridge.cv2_to_imgmsg(mask_mono8, encoding="mono8")
        out_msg.header = msg.header
        self.pub.publish(out_msg)

        # Create an RGB visualization image for RViz
        vis_img = np.zeros((msg.height, msg.width, 3), dtype=np.uint8)
        vis_img[mask_mono8 == 1] = [255, 0, 0]   # Ego-left: Red
        vis_img[mask_mono8 == 2] = [0, 255, 0]   # Ego-right: Green
        vis_img[mask_mono8 == 3] = [0, 0, 255]   # Other lanes: Blue

        vis_msg = self.bridge.cv2_to_imgmsg(vis_img, encoding="rgb8")
        vis_msg.header = msg.header
        self.vis_pub.publish(vis_msg)


def main(args=None):
    rclpy.init(args=args)
    node = EgoLanesLiteNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
