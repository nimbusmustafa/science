import os
# from PIL import Image
import cv2
# cap = cv2.VideoCapture(2)
key_pressed = False
image = None
images = []
count1 = 0
def load_images_from_folder(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.gif')):  
            image_path = os.path.join(folder_path, filename)
            img = cv2.imread(image_path)
            images.append(img)
    return images

folder_path = '/home/mustafa/science/site1'
images = load_images_from_folder(folder_path)


stitcher = cv2.Stitcher_create()
for i in range(len(images)-3):
         for j in range(i+1, len(images)-2):
            status, result = stitcher.stitch([images[i], images[j], images[j+1], images[j+2]])

            if status == 0:
                stitched_image = result
                h, w = stitched_image.shape[:2]

                # print(f"Stitched image dimensions: {w} x {h}")

                if 3 * h > w:
                    cropped_image = stitched_image[int((h - w / 3) / 2):int((h - w / 3) / 2 + (w / 3)), :]
                elif 3 * h < w:
                    cropped_image = stitched_image[:, int((w - 3 * h) / 2): int(((w - 3 * h) / 2 + w))]
                else:
                    cropped_image = stitched_image

                cropped_filename = f"{folder_path}/final_{count1}.jpg"  
                # print(f"Cropped image dimensions: {cropped_image.shape[0]} x {cropped_image.shape[1]}")
                cv2.imwrite(cropped_filename, cropped_image)
                print(f"Cropped image saved as {cropped_filename} for {i} and {j}")

                count1 += 1

            else:
                print("Image stitching failed.")

