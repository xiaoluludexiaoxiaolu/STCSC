import os
import torch
import cv2
from PIL import Image
from torchvision import transforms
import numpy as np

class ConvNetwork(torch.nn.Module):
    def __init__(self):
        super(ConvNetwork, self).__init__()
        self.max_pooling = torch.nn.MaxPool2d(kernel_size=(2, 2))

    def forward(self, x):
        x = self.max_pooling(x)
        return x

net = ConvNetwork()

def main():
    save_path = "other/trainval_maxpool4threshold7"
    # save_path = "other/test_maxpool8threshold7"
    img_path = "train_val_img"
    # img_path = "test_img"
    img_list = sorted(os.listdir(img_path))
    # print(img_list)
    for i in img_list:
        image_path = os.path.join(img_path, i)
        print(image_path)
        image = cv2.imread(image_path)
        image = image[:, :, ::-1].transpose(2, 0, 1)
        image = np.ascontiguousarray(image).astype(np.float32)
        image = torch.from_numpy(image)
        if image.ndimension() == 3:
            image = image.unsqueeze(0)
        newImage = net(image)
        newImage = net(newImage)
        # newImage = net(newImage)
        newImage = np.array(newImage.squeeze()).transpose(1, 2, 0)
        save_image_path = os.path.join(save_path, i)
        print(save_image_path)
        cv2.imwrite(save_image_path, newImage)




if __name__ == '__main__':
    main()