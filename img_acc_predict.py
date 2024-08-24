import os
import json

import torch
from PIL import Image
from torchvision import transforms


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

    accu_num = 0
    # load image file
    class_path = "img"
    class_names = os.listdir(class_path)
    class_names = sorted(class_names)

    for i in class_names:
        dir_origin_path = os.path.join(class_path, i)
        # print(dir_origin_path)
        img_names = os.listdir(dir_origin_path)
        img_names = sorted(img_names)

        for idx, image in enumerate(img_names):
            # print(image)
            img_id = image[:-4]
            img_path = os.path.join(dir_origin_path, image)
            print(img_path)
            assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
            img = Image.open(img_path)
            img = data_transform(img)  # [3,224,224]
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

            with torch.no_grad():
                # predict class
                output = torch.squeeze(model(img.to(device))).cpu()
                predict = torch.softmax(output, dim=0)
                predict_cla = torch.argmax(predict).numpy()

            print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                         predict[predict_cla].numpy())
            print(print_res)
            if class_indict[str(predict_cla)] == i:
                print("预测正确")
                accu_num = accu_num + 1
            else:
                print("预测错误")

    print("acc: {:.3}".format(accu_num / sample_num))


if __name__ == '__main__':
    main()
