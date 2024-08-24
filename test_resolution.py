import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt


from model_V_predict import swin_tiny_patch4_window7_224 as create_model
from visualization import visualize_single_img, get_transform

import numpy as np
import cv2
from torchvision import utils as vutils


def trans(alpha, vis_dict):
    attention1 = vis_dict[0].tolist()
    attention_matrix1 = []

    for i in range(7):
        temp = []
        for j in range(8):
            temp.append(attention1[j][i * 7])
            temp.append(attention1[j][i * 7 + 1])
            temp.append(attention1[j][i * 7 + 2])
            temp.append(attention1[j][i * 7 + 3])
            temp.append(attention1[j][i * 7 + 4])
            temp.append(attention1[j][i * 7 + 5])
            temp.append(attention1[j][i * 7 + 6])
        attention_matrix1.append(temp)

    for i in range(7, 14):
        temp = []
        for j in range(8, 16):
            temp.append(attention1[j][(i - 7) * 7])
            temp.append(attention1[j][(i - 7) * 7 + 1])
            temp.append(attention1[j][(i - 7) * 7 + 2])
            temp.append(attention1[j][(i - 7) * 7 + 3])
            temp.append(attention1[j][(i - 7) * 7 + 4])
            temp.append(attention1[j][(i - 7) * 7 + 5])
            temp.append(attention1[j][(i - 7) * 7 + 6])
        attention_matrix1.append(temp)

    for i in range(14, 21):
        temp = []
        for j in range(16, 24):
            temp.append(attention1[j][(i - 14) * 7])
            temp.append(attention1[j][(i - 14) * 7 + 1])
            temp.append(attention1[j][(i - 14) * 7 + 2])
            temp.append(attention1[j][(i - 14) * 7 + 3])
            temp.append(attention1[j][(i - 14) * 7 + 4])
            temp.append(attention1[j][(i - 14) * 7 + 5])
            temp.append(attention1[j][(i - 14) * 7 + 6])
        attention_matrix1.append(temp)

    for i in range(21, 28):
        temp = []
        for j in range(24, 32):
            temp.append(attention1[j][(i - 21) * 7])
            temp.append(attention1[j][(i - 21) * 7 + 1])
            temp.append(attention1[j][(i - 21) * 7 + 2])
            temp.append(attention1[j][(i - 21) * 7 + 3])
            temp.append(attention1[j][(i - 21) * 7 + 4])
            temp.append(attention1[j][(i - 21) * 7 + 5])
            temp.append(attention1[j][(i - 21) * 7 + 6])
        attention_matrix1.append(temp)

    for i in range(28, 35):
        temp = []
        for j in range(32, 40):
            temp.append(attention1[j][(i - 28) * 7])
            temp.append(attention1[j][(i - 28) * 7 + 1])
            temp.append(attention1[j][(i - 28) * 7 + 2])
            temp.append(attention1[j][(i - 28) * 7 + 3])
            temp.append(attention1[j][(i - 28) * 7 + 4])
            temp.append(attention1[j][(i - 28) * 7 + 5])
            temp.append(attention1[j][(i - 28) * 7 + 6])
        attention_matrix1.append(temp)

    for i in range(35, 42):
        temp = []
        for j in range(40, 48):
            temp.append(attention1[j][(i - 35) * 7])
            temp.append(attention1[j][(i - 35) * 7 + 1])
            temp.append(attention1[j][(i - 35) * 7 + 2])
            temp.append(attention1[j][(i - 35) * 7 + 3])
            temp.append(attention1[j][(i - 35) * 7 + 4])
            temp.append(attention1[j][(i - 35) * 7 + 5])
            temp.append(attention1[j][(i - 35) * 7 + 6])
        attention_matrix1.append(temp)

    for i in range(42, 49):
        temp = []
        for j in range(48, 56):
            temp.append(attention1[j][(i - 42) * 7])
            temp.append(attention1[j][(i - 42) * 7 + 1])
            temp.append(attention1[j][(i - 42) * 7 + 2])
            temp.append(attention1[j][(i - 42) * 7 + 3])
            temp.append(attention1[j][(i - 42) * 7 + 4])
            temp.append(attention1[j][(i - 42) * 7 + 5])
            temp.append(attention1[j][(i - 42) * 7 + 6])
        attention_matrix1.append(temp)

    for i in range(49, 56):
        temp = []
        for j in range(56, 64):
            temp.append(attention1[j][(i - 49) * 7])
            temp.append(attention1[j][(i - 49) * 7 + 1])
            temp.append(attention1[j][(i - 49) * 7 + 2])
            temp.append(attention1[j][(i - 49) * 7 + 3])
            temp.append(attention1[j][(i - 49) * 7 + 4])
            temp.append(attention1[j][(i - 49) * 7 + 5])
            temp.append(attention1[j][(i - 49) * 7 + 6])
        attention_matrix1.append(temp)
        i = i + 1

    attention_matrix_array1 = np.array(attention_matrix1)

    attention2 = vis_dict[1].tolist()
    attention_matrix2 = []

    for i in range(7):
        temp = []
        for j in range(4):
            temp.append(attention2[j][i * 7])
            temp.append(attention2[j][i * 7 + 1])
            temp.append(attention2[j][i * 7 + 2])
            temp.append(attention2[j][i * 7 + 3])
            temp.append(attention2[j][i * 7 + 4])
            temp.append(attention2[j][i * 7 + 5])
            temp.append(attention2[j][i * 7 + 6])
        attention_matrix2.append(temp)

    for i in range(7, 14):
        temp = []
        for j in range(4, 8):
            temp.append(attention2[j][(i - 7) * 7])
            temp.append(attention2[j][(i - 7) * 7 + 1])
            temp.append(attention2[j][(i - 7) * 7 + 2])
            temp.append(attention2[j][(i - 7) * 7 + 3])
            temp.append(attention2[j][(i - 7) * 7 + 4])
            temp.append(attention2[j][(i - 7) * 7 + 5])
            temp.append(attention2[j][(i - 7) * 7 + 6])
        attention_matrix2.append(temp)

    for i in range(14, 21):
        temp = []
        for j in range(8, 12):
            temp.append(attention2[j][(i - 14) * 7])
            temp.append(attention2[j][(i - 14) * 7 + 1])
            temp.append(attention2[j][(i - 14) * 7 + 2])
            temp.append(attention2[j][(i - 14) * 7 + 3])
            temp.append(attention2[j][(i - 14) * 7 + 4])
            temp.append(attention2[j][(i - 14) * 7 + 5])
            temp.append(attention2[j][(i - 14) * 7 + 6])
        attention_matrix2.append(temp)

    i = 21
    j = 12
    for i in range(21, 28):
        temp = []
        for j in range(12, 16):
            temp.append(attention2[j][(i - 21) * 7])
            temp.append(attention2[j][(i - 21) * 7 + 1])
            temp.append(attention2[j][(i - 21) * 7 + 2])
            temp.append(attention2[j][(i - 21) * 7 + 3])
            temp.append(attention2[j][(i - 21) * 7 + 4])
            temp.append(attention2[j][(i - 21) * 7 + 5])
            temp.append(attention2[j][(i - 21) * 7 + 6])
        attention_matrix2.append(temp)

    attention_matrix_array2 = np.array(attention_matrix2)

    attention_matrix1_weight = []  # 58 * 58 -> 28 * 28

    for i in range(28):
        temp = []
        for j in range(28):
            value = (attention_matrix1[2 * i][2 * j] + attention_matrix1[2 * i][2 * j + 1] +
                     attention_matrix1[2 * i + 1][2 * j] + attention_matrix1[2 * i + 1][2 * j + 1]) / 4
            temp.append(value)
        attention_matrix1_weight.append(temp)

    attention_matrix1_weight_array = np.array(attention_matrix1_weight)

    attention_matrix_array3 = alpha * attention_matrix1_weight_array + (1 - alpha) * attention_matrix_array2
    attention_matrix_array3 = attention_matrix_array3.flatten()
    return torch.from_numpy(attention_matrix_array3)




def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    visualization_trans = get_transform(256)

    img_size = 224
    data_transform = transforms.Compose(
        [transforms.Resize(int(img_size * 1.14)),
         transforms.CenterCrop(img_size),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    save_path = "test_resolution73"

    image_path_base = "test_img_partiton_token/resolution73"
    rate_list = [14, 28, 60, 89, 131, 165,
                 206, 237, 272, 299, 324, 348,
                 368, 386, 399, 410, 417, 420]
    index = 0
    token_list = ["254", "259", "270", "281", "296", "308",
              "323", "334", "347", "356", "365", "374",
              "381", "388", "392", "396", "399", "400"]
    token_width = 8
    token_height = 8
    for token in token_list:
        num = int(token)
        # print("token num: ", num)
        if rate_list[index]<100:
            resolution0 = 150
            resolution1 = 80
            resolution2 = 25
            if resolution0+resolution1+resolution2>num:
                resolution2 = num - resolution0 - resolution1
                resolution3 = 0
            else:
                resolution3 = num - resolution0 - resolution1 - resolution2
        elif rate_list[index]>=100 and rate_list[index]<200:
            resolution0 = 160
            resolution1 = 90
            resolution2 = 35
            if resolution0 + resolution1 + resolution2 > num:
                resolution2 = num - resolution0 - resolution1
                resolution3 = 0
            else:
                resolution3 = num - resolution0 - resolution1 - resolution2
        elif rate_list[index]>=200 and rate_list[index]<300:
            resolution0 = 170
            resolution1 = 100
            resolution2 = 45
            if resolution0 + resolution1 + resolution2 > num:
                resolution2 = num - resolution0 - resolution1
                resolution3 = 0
            else:
                resolution3 = num - resolution0 - resolution1 - resolution2
        elif rate_list[index]>=300 and rate_list[index]<400:
            resolution0 = 180
            resolution1 = 110
            resolution2 = 65
            if resolution0 + resolution1 + resolution2 > num:
                resolution2 = num - resolution0 - resolution1
                resolution3 = 0
            else:
                resolution3 = num - resolution0 - resolution1 - resolution2
        else:
            resolution0 = 190
            resolution1 = 120
            resolution2 = 75
            if resolution0 + resolution1 + resolution2 > num:
                resolution2 = num - resolution0 - resolution1
                resolution3 = 0
            else:
                resolution3 = num - resolution0 - resolution1 - resolution2
        # print("resolution: ", resolution0, resolution1, resolution2, resolution3)
        index = index + 1

        image_path = os.path.join(image_path_base, token)
        img_list = sorted(os.listdir(image_path))
        for i in img_list:
            img_path = os.path.join(image_path, i)
            print(img_path)
            assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
            img = Image.open(img_path)
            original_img = img
            # [N, C, H, W]
            original_img = data_transform(original_img)  # [3,224,224]
            # expand batch dimension
            original_img = torch.unsqueeze(original_img, dim=0)  # [1,3,224,224]

            # [N, C, H, W]
            img = data_transform(img)  # [3,224,224]
            img = img.permute(1, 2, 0)
            img = np.array(img)
            img -= np.min(img)
            img /= np.max(img)
            img = img * 255
            img = Image.fromarray(img.astype('uint8')).convert('RGB')

            sample_image = np.zeros((224, 224, 3), np.uint8)

            # read class_indict
            json_path = 'class_indices.json'
            assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

            with open(json_path, "r") as f:
                class_indict = json.load(f)

                # create model
                model = create_model(num_classes=45).to(device)
                # load model weights
                model_weight_path = "weights/model-98.pth"
                model.load_state_dict(torch.load(model_weight_path, map_location=device))
                model.eval()
                with torch.no_grad():
                    # predict class
                    output, attentions = model(original_img.to(device))
                    alpha = 0.4
                    attn_dict = trans(alpha, attentions)
                    # print(attn_dict)
                    sorted_index = torch.argsort(attn_dict, descending=True).tolist()
                    # print(sorted_index)
                    resolution0_index = sorted_index[0: resolution0]
                    # print(resolution0_index, len(resolution0_index))
                    for index0 in resolution0_index:
                        index_i = int(index0 / 28)
                        index_j = index0 - 28 * index_i
                        x = index_j * 8
                        y = index_i * 8
                        img_block = img.crop((x, y, x + 8, y + 8))
                        x1 = x
                        x2 = x + 8
                        y1 = y
                        y2 = y + 8
                        sample_image[y1:y2, x1:x2] = img_block
                    resolution1_index = sorted_index[resolution0: resolution0+resolution1]
                    # print(resolution1_index, len(resolution1_index))
                    for index1 in resolution1_index:
                        index_i = int(index1 / 28)
                        index_j = index1 - 28 * index_i
                        x = index_j * 8
                        y = index_i * 8
                        img_block = img.crop((x, y, x + 8, y + 8))
                        dowm_block = img_block.resize((4, 4))
                        up_block = dowm_block.resize((token_width, token_height))
                        x1 = x
                        x2 = x + 8
                        y1 = y
                        y2 = y + 8
                        sample_image[y1:y2, x1:x2] = up_block
                    resolution2_index = sorted_index[resolution0+resolution1: resolution0 + resolution1 + resolution2]
                    # print(resolution2_index, len(resolution2_index))
                    for index2 in resolution2_index:
                        index_i = int(index2 / 28)
                        index_j = index2 - 28 * index_i
                        x = index_j * 8
                        y = index_i * 8
                        img_block = img.crop((x, y, x + 8, y + 8))
                        dowm_block = img_block.resize((2, 2))
                        up_block = dowm_block.resize((token_width, token_height))
                        x1 = x
                        x2 = x + 8
                        y1 = y
                        y2 = y + 8
                        sample_image[y1:y2, x1:x2] = up_block
                    if resolution3 > 0:
                        resolution3_index = sorted_index[
                                            resolution0 + resolution1 + resolution2: resolution0 + resolution1 + resolution2 + resolution3]
                        # print(resolution3_index, len(resolution3_index))
                        for index3 in resolution3_index:
                            index_i = int(index3 / 28)
                            index_j = index3 - 28 * index_i
                            x = index_j * 8
                            y = index_i * 8
                            img_block = img.crop((x, y, x + 8, y + 8))
                            dowm_block = img_block.resize((1, 1))
                            up_block = dowm_block.resize((token_width, token_height))
                            x1 = x
                            x2 = x + 8
                            y1 = y
                            y2 = y + 8
                            sample_image[y1:y2, x1:x2] = up_block

                    save_img_path = os.path.join(save_path, i[4:])
                    print(save_img_path)
                    cv2.imwrite(save_img_path, sample_image)
                    # break
        # break

                    # visualation
                    # important_tokens = num
                    # visualize_single_img(attentions, important_tokens, original_img, device, visualization_trans, i[4:])


if __name__ == '__main__':
    main()