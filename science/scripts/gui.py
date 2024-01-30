#!/usr/bin/env python3
import sys
import os
import cv2
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer
import rospy
from std_msgs.msg import Int64
import numpy as np
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge

class GUI(QWidget):
    def __init__(self):
        super(GUI, self).__init__()

        self.bridge = CvBridge()
        self.camera_number = 1
        self.count_site1 = 0
        self.count_site2 = 0
        self.stitch_count = 0
        self.count1=0

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('GUI')
        self.setGeometry(100, 100, 800, 600)

        self.label = QLabel(self)

        self.button_screenshot_site1 = QPushButton('Take Screenshot Site1', self)
        self.button_screenshot_site1.clicked.connect(lambda: self.screenshot(site_number=1))

        self.button_screenshot_site2 = QPushButton('Take Screenshot Site2', self)
        self.button_screenshot_site2.clicked.connect(lambda: self.screenshot(site_number=2))

        self.button_stitch_site1 = QPushButton('Stitch Images Site1', self)
        self.button_stitch_site1.clicked.connect(lambda: self.stitch_images(site_number=1))

        self.button_stitch_site2 = QPushButton('Stitch Images Site2', self)
        self.button_stitch_site2.clicked.connect(lambda: self.stitch_images(site_number=2))

        self.button_launch1 = QPushButton('Switch to Microscope', self)
        self.button_launch1.clicked.connect(lambda: self.switch_camera(1))

        self.button_launch2 = QPushButton('Switch to Camera', self)
        self.button_launch2.clicked.connect(lambda: self.switch_camera(2))

        layout = QVBoxLayout(self)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.button_screenshot_site1)
        button_layout.addWidget(self.button_screenshot_site2)
        button_layout.addWidget(self.button_stitch_site1)
        button_layout.addWidget(self.button_stitch_site2)

        layout.addWidget(self.label)
        layout.addLayout(button_layout)
        layout.addWidget(self.button_launch1)
        layout.addWidget(self.button_launch2)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_image)
        self.timer.start(100)  

    def update_image(self):
        try:
            camera_image = rospy.wait_for_message('/usb_cam/image_raw/compressed', CompressedImage, timeout=1)
            np_arr = np.frombuffer(camera_image.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            cv_image_bgr = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)  # Convert to BGR format

            height, width, channel = cv_image_bgr.shape
            bytes_per_line = 3 * width
            q_image = QImage(cv_image_bgr.data, width, height, bytes_per_line, QImage.Format_RGB888)

            self.label.setPixmap(QPixmap.fromImage(q_image))

        except rospy.exceptions.ROSException as e:
            print(f"Error: {e}")

    def screenshot(self, site_number):
        if site_number == 1:
            self.save_screenshot(self.label, site_number=self.count_site1, folder='panorama_site1')
            self.count_site1 += 1
        elif site_number == 2:
            self.save_screenshot(self.label, site_number=self.count_site2, folder='panorama_site2')
            self.count_site2 += 1

    def save_screenshot(self, label, site_number, folder):
        pixmap = label.pixmap()
        if pixmap:
            screenshot_path = os.path.join(
                f"/home/nikhilesh/GUII/src/science/scripts/{folder}/screenshot_{site_number}.png"
            )
            pixmap.save(screenshot_path)
            print(f"Saved screenshot for Site {folder}: {screenshot_path}")

    def stitch_images(self, site_number):
        if site_number == 1:
            folder_path = '/home/nikhilesh/GUII/src/science/scripts/panorama_site1'
            count1 = self.count1
        elif site_number == 2:
            folder_path = '/home/nikhilesh/GUII/src/science/scripts/panorama_site2'
            count1 = self.count1
        else:
            print("Invalid site number.")
            return

        images = load_images_from_folder(folder_path)

        stitcher = cv2.Stitcher_create()
        for i in range(len(images)-3):
            for j in range(i+1, len(images)-2):
                status, result = stitcher.stitch([images[i], images[j], images[j+1], images[j+2]])

                if status == 0:
                    stitched_image = result
                    h, w = stitched_image.shape[:2]

                    if 3 * h > w:
                        cropped_image = stitched_image[int((h - w / 3) / 2):int((h - w / 3) / 2 + (w / 3)), :]
                    elif 3 * h < w:
                        cropped_image = stitched_image[:, int((w - 3 * h) / 2): int(((w - 3 * h) / 2 + w))]
                    else:
                        cropped_image = stitched_image

                    cropped_filename = f"{folder_path}/final_{count1}.jpg"
                    cv2.imwrite(cropped_filename, cropped_image)
                    print(f"Cropped image saved as {cropped_filename} for {i} and {j}")

                    count1 += 1
                else:
                    print("Image stitching failed.")

    def switch_camera(self, camera_number):
        self.camera_number = camera_number
        rospy.loginfo(f'Switching to Camera {self.camera_number}')
        camera_switch_publisher.publish(self.camera_number)

def load_images_from_folder(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.gif')):
            image_path = os.path.join(folder_path, filename)
            img = cv2.imread(image_path)
            images.append(img)
    return images

if __name__ == '__main__':
    rospy.init_node('ros_image_subscriber', anonymous=True)
    camera_switch_publisher = rospy.Publisher('camera_switch_topic', Int64, queue_size=1)

    app = QApplication(sys.argv)
    camera_controller = GUI()
    camera_controller.show()
    sys.exit(app.exec_())