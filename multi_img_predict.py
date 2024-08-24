import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import swin_tiny_patch4_window7_224 as create_model

sample_num = 6300

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    img_size = 224
    data_transform = transforms.Compose(
        [transforms.Resize(int(img_size * 1.14)),
         transforms.CenterCrop(img_size),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    total_num = 0
    accu_num = 0
    # load image file
    class_path = "test_data/random/random_threshold8"
    class_names = os.listdir(class_path)
    class_names = sorted(class_names)
    print(class_names)

    # read class_indict
    json_path = 'class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # create model
    model = create_model(num_classes=45).to(device)
    # load model weights
    # model_weight_path = "weights/model-97.pth"
    model_weight_path = "total_weight/pretrain/trainval_original-97.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()

    for i in class_names:
        img_path = os.path.join(class_path, i)
        print(img_path)
        # print(i[4: -12])
        assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
        img = Image.open(img_path)
        img = data_transform(img)  # [3,224,224]
        img = torch.unsqueeze(img, dim=0)  # [1,3,224,224]

        with torch.no_grad():
            # predict class
            output = torch.squeeze(model(img.to(device))).cpu()
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()

        print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                     predict[predict_cla].numpy())
        print(print_res)
        if class_indict[str(predict_cla)] == i[0: -8]:
            print("预测正确")
            accu_num = accu_num + 1
            total_num = total_num + 1
            print("present acc: {:.3}".format(accu_num / total_num))
        else:
            print("预测错误")
            total_num = total_num + 1
            print("present acc: {:.3}".format(accu_num / total_num))

    print("acc: {:.3}".format(accu_num / sample_num))


if __name__ == '__main__':
    main()
