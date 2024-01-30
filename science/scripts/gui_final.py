#!/usr/bin/env python3
import sys
import os
import cv2
from pyrtcm import RTCMReader
from serial import Serial
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer
import rospy
from PyQt5.QtWidgets import QInputDialog
import socket
import numpy as np
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import threading

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
port = 12345
s.connect(('10.0.0.6', port))

class GUI(QWidget):
    def __init__(self):
        super(GUI, self).__init__()

        self.bridge = CvBridge()
        self.camera_number = 1
        self.count_site1 = 0
        self.count_site2 = 0
        self.count_site3 = 0
        self.count_site4 = 0

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('GUI')
        self.setGeometry(100, 100, 800, 600)

        self.label = QLabel(self)
        self.gps_label = QLabel(self)

        self.button_screenshot_site1 = QPushButton('Take Screenshot Site1', self)
        self.button_screenshot_site1.clicked.connect(lambda: self.screenshot(site_number=1))

        self.button_screenshot_site2 = QPushButton('Take Screenshot Site2', self)
        self.button_screenshot_site2.clicked.connect(lambda: self.screenshot(site_number=2))

        self.button_screenshot_site3 = QPushButton('Take Screenshot Site3', self)
        self.button_screenshot_site3.clicked.connect(lambda: self.screenshot(site_number=3))

        self.button_screenshot_site4 = QPushButton('Take Screenshot Site4', self)
        self.button_screenshot_site4.clicked.connect(lambda: self.screenshot(site_number=4))

        self.button_stitch_site1 = QPushButton('Stitch Images Site1', self)
        self.button_stitch_site1.clicked.connect(lambda: self.stitch_images(site_number=1))

        self.button_stitch_site2 = QPushButton('Stitch Images Site2', self)
        self.button_stitch_site2.clicked.connect(lambda: self.stitch_images(site_number=2))

        self.button_stitch_site3 = QPushButton('Stitch Images Site3', self)
        self.button_stitch_site3.clicked.connect(lambda: self.stitch_images(site_number=3))

        self.button_stitch_site4 = QPushButton('Stitch Images Site4', self)
        self.button_stitch_site4.clicked.connect(lambda: self.stitch_images(site_number=4))

        layout = QVBoxLayout(self)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.button_screenshot_site1)
        button_layout.addWidget(self.button_screenshot_site2)
        button_layout.addWidget(self.button_screenshot_site3)
        button_layout.addWidget(self.button_screenshot_site4)
        button_layout.addWidget(self.button_stitch_site1)
        button_layout.addWidget(self.button_stitch_site2)
        button_layout.addWidget(self.button_stitch_site3)
        button_layout.addWidget(self.button_stitch_site4)

        layout.addWidget(self.label)
        layout.addWidget(self.gps_label)
        layout.addLayout(button_layout)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_image)
        self.timer.start(100)

    def update_image(self):
        try:
            camera_image = rospy.wait_for_message('/test/image_raw/compressed', CompressedImage, timeout=1)
            np_arr = np.frombuffer(camera_image.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            cv_image_bgr = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)

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
        elif site_number == 3:
            self.save_screenshot(self.label, site_number=self.count_site3, folder='panorama_site3')
            self.count_site3 += 1
        elif site_number == 4:
            self.save_screenshot(self.label, site_number=self.count_site4, folder='panorama_site4')
            self.count_site4 += 1

    def save_screenshot(self, label, site_number, folder):
        pixmap = label.pixmap()
        if pixmap:
            screenshot_path = os.path.join(
                f"/home/mustafa/GUI/src/science/scripts/{folder}/screenshot_{site_number}.png"
            )
            pixmap.save(screenshot_path)
            print(f"Saved screenshot for Site {folder}: {screenshot_path}")

    def update_gps_label(self, lat, lon, alt):
        gps_text = f'GPS: Latitude: {lat}, Longitude: {lon}, Altitude: {alt}'
        self.gps_label.setText(gps_text)

    def get_imu_value(self):
        text, ok = QInputDialog.getText(self, 'IMU Input', 'Enter IMU value:')
        if ok:
            imu_value = float(text.strip())
            return imu_value
        else:
            return None

    def stitch_images(self, site_number):
        if site_number == 1:
            folder_path = '/home/mustafa/GUI/src/science/scripts/panorama_site1'
            count = self.count_site1
        elif site_number == 2:
            folder_path = '/home/mustafa/GUI/src/science/scripts/panorama_site2'
            count = self.count_site2
        elif site_number == 3:
            folder_path = '/home/mustafa/GUI/src/science/scripts/panorama_site3'
            count = self.count_site3
        elif site_number == 4:
            folder_path = '/home/mustafa/GUI/src/science/scripts/panorama_site4'
            count = self.count_site4
        else:
            print("Invalid site number.")
            return

        images = load_images_from_folder(folder_path)

        stitcher = cv2.Stitcher_create()
        imu_value = self.get_imu_value()

        for i in range(len(images) - 3):
            for j in range(i + 1, len(images) - 2):
                status, result = stitcher.stitch([images[i], images[j], images[j + 1], images[j + 2]])

                if status == 0:
                    stitched_image = result
                    h, w = stitched_image.shape[:2]

                    if 3 * h > w:
                        cropped_image = stitched_image[int((h - w / 3) / 2):int((h - w / 3) / 2 + (w / 3)), :]
                    elif 3 * h < w:
                        cropped_image = stitched_image[:, int((w - 3 * h) / 2): int(((w - 3 * h) / 2 + w))]
                    else:
                        cropped_image = stitched_image

                    gps = s.recv(1024).decode('utf-8').split("G")[:-1]
                    
                    lat=None
                    lon=None
                    alt=None
                    if len(gps)==1:
                        # print(gps)
                        lat, lon, alt = gps[0].split(',')
                        # print(lat,lon)
                        cropped_image = add_overlay(cropped_image, imu_value, lat, lon, alt)

                        cropped_filename = f"{folder_path}/final_{count}.jpg"
                        cv2.imwrite(cropped_filename, cropped_image)
                        print(f"Cropped image saved as {cropped_filename} for {i} and {j}")

                        count += 1
                else:
                    print("Image stitching failed.")

def load_images_from_folder(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.gif')):
            image_path = os.path.join(folder_path, filename)
            img = cv2.imread(image_path)
            images.append(img)
    return images

def add_overlay(image, angle, lat, long, elevation):
    color = (0, 0, 255)
    h, w, = image.shape[:2]
    image = cv2.putText(image, 'N', (70, 30), cv2.FONT_HERSHEY_TRIPLEX,
                        0.5, color, 1, cv2.LINE_AA)
    image = cv2.putText(image, 'W', (20, 75), cv2.FONT_HERSHEY_TRIPLEX,
                        0.5, color, 1, cv2.LINE_AA)
    image = cv2.putText(image, 'E', (120, 75), cv2.FONT_HERSHEY_TRIPLEX,
                        0.5, color, 1, cv2.LINE_AA)
    image = cv2.putText(image, 'S', (70, 120), cv2.FONT_HERSHEY_TRIPLEX,
                        0.5, color, 1, cv2.LINE_AA)

    image = cv2.putText(image, f'GPS: Latitude: {lat}', (200, 95), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 1, cv2.LINE_AA)
    image = cv2.putText(image, f'GPS: Longitude: {long}', (200, 75), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 1, cv2.LINE_AA)
    image = cv2.putText(image, f'Elevation: {elevation}', (200, 55), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 1, cv2.LINE_AA)
    image = cv2.putText(image, f'Accuracy: 1.5 meters', (200, 35), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 1, cv2.LINE_AA)
    image = cv2.putText(image, f'Scale: 1:40', (200, 115), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 1, cv2.LINE_AA)
    

    overlay_image = cv2.imread('/home/mustafa/GUI/src/science/scripts/needle2.png', cv2.IMREAD_UNCHANGED)
    overlay_image = cv2.resize(overlay_image, (70, 70))
    row, col, _ = overlay_image.shape
    center = tuple(np.array([row, col]) / 2)
    rot_mat = cv2.getRotationMatrix2D(center, (-1.0) * angle, 1.0)
    overlay_image = cv2.warpAffine(overlay_image, rot_mat, (col, row))

    x_pos = 40
    y_pos = 38

    needle_image = image.copy()

    for y in range(overlay_image.shape[0]):
        for x in range(overlay_image.shape[1]):
            if overlay_image[y, x, 3] > 0:
                needle_image[y + y_pos, x + x_pos, 0] = overlay_image[y, x, 0]
                needle_image[y + y_pos, x + x_pos, 1] = overlay_image[y, x, 1]
                needle_image[y + y_pos, x + x_pos, 2] = overlay_image[y, x, 2]

    overlay_image1 = cv2.imread('/home/mustafa/GUI/src/science/scripts/MRM_logo.png', cv2.IMREAD_UNCHANGED)
    overlay_image1 = cv2.resize(overlay_image1, (50, 50))

    x_pos = (image.shape[1]) - overlay_image1.shape[1] - 20
    y_pos = 10

    mrm_image = needle_image.copy()

    for y in range(overlay_image1.shape[0]):
        for x in range(overlay_image1.shape[1]):
            if overlay_image1[y, x, 3] > 0:
                mrm_image[y + y_pos, x + x_pos, 0] = overlay_image1[y, x, 0]
                mrm_image[y + y_pos, x + x_pos, 1] = overlay_image1[y, x, 1]
                mrm_image[y + y_pos, x + x_pos, 2] = overlay_image1[y, x, 2]

    return mrm_image

if __name__ == '__main__':
    rospy.init_node('ros_image_subscriber', anonymous=True)
    app = QApplication(sys.argv)
    camera_controller = GUI()

    camera_controller.show()
    sys.exit(app.exec_())
