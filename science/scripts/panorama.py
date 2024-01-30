import cv2
cap = cv2.VideoCapture(2)
key_pressed = False
image = None
images = []
count = 0
count1 = 0

while True:
    ret, frame = cap.read()

    cv2.imshow("Video Feed", frame)

    key = cv2.waitKey(1)

    if key == ord('c'):
        key_pressed = True
        image = frame.copy()
        count = count + 1
        print(f"Image {count} captured.")
        print(f"Image {count} dimensions: {image.shape[0]} x {image.shape[1]}")

        image_filename = f"pic{count}.jpg"
        cv2.imwrite(image_filename, image)
        print(f"Image saved as {image_filename}")

    if key == ord('s') and len(images) >= 4:
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

                cropped_filename = f"final_{count1}.jpg"
                # print(f"Cropped image dimensions: {cropped_image.shape[0]} x {cropped_image.shape[1]}")
                cv2.imwrite(cropped_filename, cropped_image)
                print(f"Cropped image saved as {cropped_filename} for {i} and {j}")

                count1 += 1

            else:
                print("Image stitching failed.")


        key_pressed = False

    if key_pressed:
        images.append(image)
        key_pressed = False

    if key == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break
