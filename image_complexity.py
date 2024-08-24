import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from model_V_predict import swin_tiny_patch4_window7_224 as create_model
from visualization import visualize_single_img, get_transform

import numpy as np
from torchvision import utils as vutils

def gen_masked_tokens(tokens, indices, alpha=0.3):
    indices = [i for i in range(tokens.shape[0]) if i not in indices]
    tokens = tokens.copy()
    # tokens[indices] = alpha * tokens[indices] + (1 - alpha) * 255
    tokens[indices] = 0
    return tokens

def recover_image(tokens):
    # image: (C, 196, 16, 16)
    image = tokens.reshape(28, 28, 8, 8, 3).swapaxes(1, 2).reshape(224, 224, 3)
    return image

def gen_visualization(image, keep_indices):
    image_tokens = image.reshape(28, 8, 28, 8, 3).swapaxes(1, 2).reshape(784, 8, 8, 3)

    viz = recover_image(gen_masked_tokens(image_tokens, keep_indices))
    return viz

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

def save_image_tensor(input_tensor: torch.Tensor, class_names, filename):
    """
    将tensor保存为图片
    :param input_tensor: 要保存的tensor
    :param filename: 保存的文件名
    """
    assert ((len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1) or len(input_tensor.shape) == 3)
    # 复制一份
    input_tensor = input_tensor.clone().detach()
    # 到cpu
    input_tensor = input_tensor.to(torch.device('cpu'))
    # 反归一化
    save_name = './visualization'
    class_name = os.path.join(save_name, class_names)
    filename = os.path.join(class_name, filename)
    print(filename)
    vutils.save_image(input_tensor, filename)


def visualize_single_img(attentions, important_tokens, img_input, device, transform, save_name, class_names):
    alpha = 0.1

    # img: 1, 3, H, W
    image_raw = transform(img_input)
    images = image_raw.unsqueeze(0)
    images = images.to(device, non_blocking=True)
    print(images.shape)

    attn_dict = trans(alpha, attentions)

    image_raw = image_raw * 255
    image_raw = image_raw.squeeze(0).permute(1, 2, 0).cpu().numpy()

    sorted_index = torch.argsort(attn_dict, descending=True)[:important_tokens]

    viz = gen_visualization(image_raw, sorted_index)
    viz = torch.from_numpy(viz).permute(2, 0, 1)

    viz = viz / 255

    save_name = save_name.split(".")[0]
    save_image_tensor(viz, class_names, '{}_{}.jpg'.format(save_name, important_tokens))
    print("Visualization finished")

def getComplication(attentions, img_input, device, transform, save_name):
    mass_percentage = 0.6
    sum_image_token = 16
    mass_threshold = mass_percentage * sum_image_token
    alpha = 1.0
    attn_dict = trans(alpha, attentions)
    # print(attn_dict)
    # sum = torch.sum(attn_dict)
    # print("sum: ", sum)
    attn_dict_sort = torch.sort(attn_dict, descending=True)[0]
    # print(attn_dict_sort)
    sum_token = 0
    for i in range(784):
        sum_token = sum_token + attn_dict_sort[i]
        if sum_token >= mass_threshold:
            # print("token number: ", i)
            # break
            return i



def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    visualization_trans = get_transform(256)

    img_size = 224
    data_transform = transforms.Compose(
        [transforms.Resize(int(img_size * 1.14)),
         transforms.CenterCrop(img_size),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


    # load image
    class_path = "train_val_image"
    # class_path = "img"
    class_names = os.listdir(class_path)
    class_names = sorted(class_names)
    token_num_list = []
    token_num_keys = []
    token_num_values = []

    all_img_path = "train_val_img_token_attn/train_val_img_token_attn_1.0"
    # all_img_path = "test_img_token_attn/test_img_token_attn_1.0"

    for i in class_names:
        dir_origin_path = os.path.join(class_path, i)
        # print(dir_origin_path)
        img_names = os.listdir(dir_origin_path)
        img_names = sorted(img_names)

        for idx, image in enumerate(img_names):
            # print(image)
            token_num_keys.append(image)
            # img_id = image[:-4]
            # print(img_id)
            img_path = os.path.join(dir_origin_path, image)
            print(img_path)
            assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)

            img = Image.open(img_path)
            original_img = img
            plt.imshow(img)
            # [N, C, H, W]
            img = data_transform(img)  # [3,224,224]
            # expand batch dimension
            img = torch.unsqueeze(img, dim=0)  # [1,3,224,224]

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
                output, attentions = model(img.to(device))

                # visualation
                complication = getComplication(attentions, original_img, device, visualization_trans, img_path)
                print("token number: ", complication)
                token_num_list.append(complication)
                token_num_values.append(complication)
                if complication<10:
                    save_img_path = str("00") + str(complication) + str("_") + image
                elif complication<100 and complication>=10:
                    save_img_path = str("0") + str(complication) + str("_") + image
                else:
                    save_img_path = str(complication) + str("_") + image
                save_img_path = os.path.join(all_img_path, save_img_path)
                print(save_img_path)
                original_img.save(save_img_path)



if __name__ == '__main__':
    main()
