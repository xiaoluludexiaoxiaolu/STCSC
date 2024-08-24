import os
import torch
import cv2
from PIL import Image
from torchvision import transforms
import numpy as np


def main():
    save_path = "other/trainval_pyramid4threshold7"
    # save_path = "other/test_pyramid4threshold7"
    img_path = "train_val_img"
    # img_path = "test_img"
    img_list = sorted(os.listdir(img_path))
    # print(img_list)
    for i in img_list:
        image_path = os.path.join(img_path, i)
        print(image_path)
        image = cv2.imread(image_path)
        newImage = cv2.pyrDown(image)
        newImage = cv2.pyrDown(newImage)
        # newImage = cv2.pyrDown(newImage)
        save_image_path = os.path.join(save_path, i)
        print(save_image_path)
        cv2.imwrite(save_image_path, newImage)




if __name__ == '__main__':
    main()