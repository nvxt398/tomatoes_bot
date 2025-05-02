import os
import cv2
import numpy as np


def leaf_segmentation(image_path):
    img = cv2.imread(image_path)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_green = np.array([25, 40, 20])
    upper_green = np.array([90, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)

    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return cv2.bitwise_and(img, img, mask=mask)

def segment_dataset(input_base, output_base):
    for subset in ["train", "val"]:
        subset_input = os.path.join(input_base, subset)
        subset_output = os.path.join(output_base, subset)
        os.makedirs(subset_output, exist_ok=True)

        for class_name in os.listdir(subset_input):
            class_input_path = os.path.join(subset_input, class_name)
            if not os.path.isdir(class_input_path):
                continue

            class_output_path = os.path.join(subset_output, class_name)
            os.makedirs(class_output_path, exist_ok=True)

            for image_name in os.listdir(class_input_path):
                if not image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue

                image_input_path = os.path.join(class_input_path, image_name)
                segmented = leaf_segmentation(image_input_path)
                if segmented is not None:
                    save_path = os.path.join(class_output_path, image_name)
                    cv2.imwrite(save_path, segmented)
