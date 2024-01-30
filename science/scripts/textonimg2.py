import cv2
import numpy as np

def add_overlay(image, angle, lat, long, elevation):
    color = (0, 0, 255)

    image = cv2.putText(image, 'N', (102, 45), cv2.FONT_HERSHEY_TRIPLEX,
                        1.5, color, 1, cv2.LINE_AA)
    image = cv2.putText(image, 'W', (20, 125), cv2.FONT_HERSHEY_TRIPLEX,
                        1.5, color, 1, cv2.LINE_AA)
    image = cv2.putText(image, 'E', (180, 125), cv2.FONT_HERSHEY_TRIPLEX,
                        1.5, color, 1, cv2.LINE_AA)
    image = cv2.putText(image, 'S', (102, 205), cv2.FONT_HERSHEY_TRIPLEX,
                        1.5, color, 1, cv2.LINE_AA)

    image = cv2.putText(image, f'GPS: Latitude -{lat}', (230, 135), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2, cv2.LINE_AA)
    image = cv2.putText(image, f'GPS: Longitude -{long}', (230, 95), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2, cv2.LINE_AA)
    image = cv2.putText(image, f'Elevation: -{elevation}', (230, 55), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2, cv2.LINE_AA)
    image = cv2.putText(image, f'Accuracy:- 1.5 meters', (230, 35), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2, cv2.LINE_AA)  

    overlay_image = cv2.imread('/home/nikhilesh/camera_ws/src/science/scripts/needle2.png', cv2.IMREAD_UNCHANGED)
    overlay_image = cv2.resize(overlay_image, (130, 130))
    row, col, _ = overlay_image.shape
    center = tuple(np.array([row, col]) / 2)
    rot_mat = cv2.getRotationMatrix2D(center, (-1.0) * angle, 1.0)
    overlay_image = cv2.warpAffine(overlay_image, rot_mat, (col, row))

    x_pos = 52
    y_pos = 43

    needle_image = image.copy()

    for y in range(overlay_image.shape[0]):
        for x in range(overlay_image.shape[1]):
            if overlay_image[y, x, 3] > 0:
                needle_image[y + y_pos, x + x_pos, 0] = overlay_image[y, x, 0]
                needle_image[y + y_pos, x + x_pos, 1] = overlay_image[y, x, 1]
                needle_image[y + y_pos, x + x_pos, 2] = overlay_image[y, x, 2]

    overlay_image1 = cv2.imread('/home/nikhilesh/camera_ws/src/science/scripts/MRM_logo.png', cv2.IMREAD_UNCHANGED)
    overlay_image1 = cv2.resize(overlay_image1, (100, 100))

    x_pos = (image.shape[1]) - overlay_image1.shape[1] - 20
    y_pos = 30

    mrm_image = needle_image.copy()

    for y in range(overlay_image1.shape[0]):
        for x in range(overlay_image1.shape[1]):
            if overlay_image1[y, x, 3] > 0:
                mrm_image[y + y_pos, x + x_pos, 0] = overlay_image1[y, x, 0]
                mrm_image[y + y_pos, x + x_pos, 1] = overlay_image1[y, x, 1]
                mrm_image[y + y_pos, x + x_pos, 2] = overlay_image1[y, x, 2]

    return mrm_image

input_image = cv2.imread('/home/nikhilesh/camera_ws/src/science/scripts/a.png')  
output_image = add_overlay(input_image, angle=45, lat=37.7749, long=-122.4194, elevation=50)

cv2.imshow('Output Image', output_image)
cv2.imwrite('site2_img.png', output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
