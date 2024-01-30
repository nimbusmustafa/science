import cv2
import numpy as np
import socket
import rospy
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
port = 12345
s.connect(('10.0.0.6', port))

bridge = CvBridge()

def add_overlay(image, lat, long, elevation,angle):
    color = (0, 0, 255)
    h,w,=image.shape[:2]
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

def imgCallback(img):
    input_image = bridge.compressed_imgmsg_to_cv2(img, desired_encoding="passthrough")
    gps_data=s.recv(1024).decode('utf-8').split("G")[:-1]
    if len(gps_data)==1:
        lat, lon, alt = gps_data[0].split(',')
   
        h,w=input_image.shape[:2]
        if 3 * h > w:
            cropped_image = input_image[int((h - w / 3) / 2):int((h - w / 3) / 2 + (w / 3)), :]
        elif 3 * h < w:
            cropped_image = input_image[:, int((w - 3 * h) / 2): int(((w - 3 * h) / 2 + w))]
        else:
            cropped_image = input_image

        output_image = add_overlay(cropped_image, lat, lon, alt, angle=160)

        cv2.imshow('Output Image', output_image)
        cv2.imwrite('/home/mustafa/GUI/src/science/scripts/site1_img1.png', output_image)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__=='__main__':
    # input_image = cv2.imread('/home/mustafa/GUI/src/science/scripts/panorama_site2/screenshot_1.png')  
    rospy.init_node('overlay',anonymous=True)
    rospy.Subscriber("/test/image_raw/compressed",CompressedImage, imgCallback)
    rospy.spin()


