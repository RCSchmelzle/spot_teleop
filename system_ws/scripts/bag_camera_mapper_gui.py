#!/usr/bin/env python3
"""
GUI for mapping camera topics from a ROS bag file.
Extracts a middle frame and lets user assign names to each camera.
"""

import sys
import yaml
from pathlib import Path
import subprocess
import cv2
import numpy as np
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

try:
    from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                                  QHBoxLayout, QLabel, QLineEdit, QPushButton, QScrollArea)
    from PyQt5.QtGui import QImage, QPixmap
    from PyQt5.QtCore import Qt
except ImportError:
    print("ERROR: PyQt5 not installed. Install with: pip install PyQt5")
    sys.exit(1)


class CameraFrame:
    """Holds a camera topic and its frame"""
    def __init__(self, topic, frame):
        self.topic = topic
        self.frame = frame
        # Generate default name from topic
        parts = topic.split('/')
        if 'cam' in topic.lower():
            for part in parts:
                if 'cam' in part.lower():
                    self.default_name = part
                    break
        else:
            self.default_name = f"cam{topic.replace('/', '_')}"


class BagCameraMapperGUI(QMainWindow):
    """GUI for assigning names to cameras from bag file"""

    def __init__(self, bag_path, output_dir):
        super().__init__()
        self.bag_path = Path(bag_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.camera_frames = []
        self.name_inputs = []

        self.setWindowTitle("ORB-SLAM3 Camera Mapper - Bag Mode")
        self.setGeometry(100, 100, 1200, 800)

        self.load_middle_frames()
        self.init_ui()

    def load_middle_frames(self):
        """Extract middle frame from each camera topic in bag"""
        print(f"Loading frames from: {self.bag_path}")

        # Get bag info to find image topics
        result = subprocess.run(
            ['ros2', 'bag', 'info', str(self.bag_path)],
            capture_output=True, text=True
        )

        # Find RGB image topics
        rgb_topics = []
        for line in result.stdout.split('\n'):
            # Look for lines with "Topic:" prefix
            if 'Topic:' in line and 'Type:' in line:
                # Parse format: "Topic: /camA/rgb/image_raw | Type: sensor_msgs/msg/Image | ..."
                parts = [p.strip() for p in line.split('|')]
                if len(parts) >= 2:
                    # Extract topic name (after "Topic: ")
                    topic_part = parts[0]
                    if 'Topic:' in topic_part:
                        topic = topic_part.split('Topic:')[1].strip()
                    else:
                        continue

                    # Extract message type (after "Type: ")
                    type_part = parts[1]
                    if 'Type:' in type_part:
                        msg_type = type_part.split('Type:')[1].strip()
                    else:
                        continue

                    # Check if it's an RGB image topic
                    if 'sensor_msgs/msg/Image' in msg_type and '/rgb/' in topic:
                        if topic not in rgb_topics:
                            rgb_topics.append(topic)

        print(f"Found {len(rgb_topics)} RGB camera topics")
        if rgb_topics:
            print(f"Topics: {', '.join(rgb_topics)}")

        # Read middle frame from each topic
        for topic in rgb_topics:
            frame = self.extract_middle_frame(topic)
            if frame is not None:
                self.camera_frames.append(CameraFrame(topic, frame))

        if not self.camera_frames:
            print("ERROR: No camera frames found in bag!")
            sys.exit(1)

    def extract_middle_frame(self, topic):
        """Extract middle frame from a specific topic using ros2 bag API"""
        try:
            from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions

            # Setup reader (use empty storage_id to auto-detect)
            storage_options = StorageOptions(uri=str(self.bag_path), storage_id='')
            converter_options = ConverterOptions('', '')

            reader = SequentialReader()
            reader.open(storage_options, converter_options)

            # Get topic metadata
            topic_types = reader.get_all_topics_and_types()
            type_map = {t.name: t.type for t in topic_types}

            if topic not in type_map:
                print(f"Topic {topic} not found in bag")
                return None

            # Read all messages from this topic
            messages = []
            while reader.has_next():
                (topic_name, data, timestamp) = reader.read_next()
                if topic_name == topic:
                    messages.append(data)

            if not messages:
                print(f"No messages found for topic {topic}")
                return None

            # Get middle message
            middle_idx = len(messages) // 2
            msg_data = messages[middle_idx]

            # Deserialize message
            msg_type = get_message('sensor_msgs/msg/Image')
            msg = deserialize_message(msg_data, msg_type)

            # Convert to CV image
            if msg.encoding == 'rgb8':
                frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            elif msg.encoding == 'bgr8':
                frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
            else:
                print(f"Warning: Unknown encoding {msg.encoding}")
                return None

            print(f"  ✓ Extracted frame from {topic} ({len(messages)} messages total)")
            return frame

        except Exception as e:
            print(f"Error extracting frame from {topic}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def init_ui(self):
        """Initialize UI with camera frames and name inputs"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        # Title
        title = QLabel("Assign Names to Cameras")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        main_layout.addWidget(title)

        # Scroll area for cameras
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout()
        scroll_widget.setLayout(scroll_layout)
        scroll.setWidget(scroll_widget)

        # Add each camera
        for i, cam_frame in enumerate(self.camera_frames):
            cam_widget = self.create_camera_widget(i, cam_frame)
            scroll_layout.addWidget(cam_widget)

        main_layout.addWidget(scroll)

        # Save button
        save_btn = QPushButton("Save Configuration")
        save_btn.clicked.connect(self.save_configuration)
        save_btn.setStyleSheet("font-size: 14px; padding: 10px;")
        main_layout.addWidget(save_btn)

    def create_camera_widget(self, index, cam_frame):
        """Create widget for one camera"""
        widget = QWidget()
        widget.setStyleSheet("border: 1px solid #ccc; margin: 5px; padding: 5px;")
        layout = QHBoxLayout()
        widget.setLayout(layout)

        # Display frame
        height, width = cam_frame.frame.shape[:2]
        display_width = 320
        display_height = int(height * display_width / width)
        frame_resized = cv2.resize(cam_frame.frame, (display_width, display_height))

        # Convert to QPixmap
        rgb_image = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)

        img_label = QLabel()
        img_label.setPixmap(pixmap)
        layout.addWidget(img_label)

        # Info and name input
        info_layout = QVBoxLayout()

        topic_label = QLabel(f"Topic: {cam_frame.topic}")
        topic_label.setWordWrap(True)
        info_layout.addWidget(topic_label)

        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Camera Name:"))

        name_input = QLineEdit(cam_frame.default_name)
        self.name_inputs.append(name_input)
        name_layout.addWidget(name_input)

        info_layout.addLayout(name_layout)
        info_layout.addStretch()

        layout.addLayout(info_layout)
        layout.setStretch(0, 1)
        layout.setStretch(1, 2)

        return widget

    def save_configuration(self):
        """Save camera mapping to YAML file"""
        # Collect camera mappings
        cameras = []
        for i, cam_frame in enumerate(self.camera_frames):
            name = self.name_inputs[i].text().strip()
            if not name:
                name = cam_frame.default_name

            # Derive depth topic from RGB topic
            # First try depth_raw (common in Xtion bags), fallback to depth
            depth_topic_raw = cam_frame.topic.replace('/rgb/', '/depth_raw/')
            depth_topic = cam_frame.topic.replace('/rgb/', '/depth/')

            # Use depth_raw if it exists in the bag, otherwise use depth
            # For now, default to depth_raw/image since that's what our bags use
            depth_topic_final = depth_topic_raw.replace('image_raw', 'image')

            cameras.append({
                'name': name,
                'topics': {
                    'rgb': cam_frame.topic,
                    'depth': depth_topic_final
                }
            })

        # Save to YAML
        mapping = {'cameras': cameras}
        output_file = self.output_dir / 'camera_mapping.yaml'

        with open(output_file, 'w') as f:
            yaml.dump(mapping, f, default_flow_style=False)

        print(f"Saved camera mapping to: {output_file}")
        print(f"Configured {len(cameras)} cameras:")
        for cam in cameras:
            print(f"  - {cam['name']}: {cam['topics']['rgb']}")

        self.close()


def main():
    if len(sys.argv) != 3:
        print("Usage: bag_camera_mapper_gui.py <bag_path> <output_dir>")
        sys.exit(1)

    bag_path = sys.argv[1]
    output_dir = sys.argv[2]

    app = QApplication(sys.argv)
    window = BagCameraMapperGUI(bag_path, output_dir)
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()