import os
import cv2

img_path = "test_img"
img = sorted(os.listdir(img_path))

img_path_554 = "test_img_partiton_token/unthreshold7/254"
img_path_559 = "test_img_partiton_token/unthreshold7/259"
img_path_570 = "test_img_partiton_token/unthreshold7/270"
img_path_581 = "test_img_partiton_token/unthreshold7/281"
img_path_596 = "test_img_partiton_token/unthreshold7/296"
img_path_608 = "test_img_partiton_token/unthreshold7/308"
img_path_623 = "test_img_partiton_token/unthreshold7/323"
img_path_634 = "test_img_partiton_token/unthreshold7/334"
img_path_647 = "test_img_partiton_token/unthreshold7/347"
img_path_656 = "test_img_partiton_token/unthreshold7/356"
img_path_665 = "test_img_partiton_token/unthreshold7/365"
img_path_674 = "test_img_partiton_token/unthreshold7/374"
img_path_681 = "test_img_partiton_token/unthreshold7/381"
img_path_688 = "test_img_partiton_token/unthreshold7/388"
img_path_692 = "test_img_partiton_token/unthreshold7/392"
img_path_696 = "test_img_partiton_token/unthreshold7/396"
img_path_699 = "test_img_partiton_token/unthreshold7/399"
img_path_700 = "test_img_partiton_token/unthreshold7/400"

# len_list1 = [32, 96, 224, 304, 416, 464, 504, 525, 540,
#             540, 525, 510, 465, 405, 330, 255, 135, 30]

# len_list2 = [30, 105, 225, 315, 389, 448, 490, 518, 546,
#             546, 532, 504, 476, 406, 336, 252, 140, 42]

# len_list3 = [26, 104, 208, 312, 377, 455, 494, 520, 546,
#             546, 533, 520, 468, 416, 351, 244, 144, 36]

# len_list4 = [36, 96, 216, 312, 396, 456, 492, 528, 552,
#             552, 540, 516, 480, 391, 330, 231, 132, 44]

# len_list5 = [33, 110, 220, 319, 396, 462, 506, 539, 550,
#             550, 550, 475, 440, 400, 320, 250, 140, 40]

# len_list6 = [30, 120, 220, 330, 420, 460, 520, 540, 560,
#             517, 495, 477, 441, 396, 333, 252, 144, 75]

len_list = [36, 117, 243, 342, 423, 477, 522, 508, 504,
            504, 496, 480, 448, 400, 344, 256, 152, 48]

# len_list = [40, 128, 256, 360, 440, 442, 469, 490, 504,
#             504, 497, 476, 455, 406, 350, 266, 168, 49]

list_554 = img[0: sum(len_list[:1])]
list_570 = img[sum(len_list[:1]): sum(len_list[:2])]
list_596 = img[sum(len_list[:2]): sum(len_list[:3])]
list_623 = img[sum(len_list[:3]): sum(len_list[:4])]
list_647 = img[sum(len_list[:4]): sum(len_list[:5])]
list_665 = img[sum(len_list[:5]): sum(len_list[:6])]
list_681 = img[sum(len_list[:6]): sum(len_list[:7])]
list_692 = img[sum(len_list[:7]): sum(len_list[:8])]
list_699 = img[sum(len_list[:8]): sum(len_list[:9])]
list_700 = img[sum(len_list[:9]): sum(len_list[:10])]
list_696 = img[sum(len_list[:10]): sum(len_list[:11])]
list_688 = img[sum(len_list[:11]): sum(len_list[:12])]
list_674 = img[sum(len_list[:12]): sum(len_list[:13])]
list_656 = img[sum(len_list[:13]): sum(len_list[:14])]
list_634 = img[sum(len_list[:14]): sum(len_list[:15])]
list_608 = img[sum(len_list[:15]): sum(len_list[:16])]
list_581 = img[sum(len_list[:16]): sum(len_list[:17])]
list_559 = img[sum(len_list[:17]): sum(len_list[:18])]

for i in list_554:
    img_token_path = os.path.join(img_path, i)
    # print(img_token_path)
    image = cv2.imread(img_token_path)
    save_img_path = os.path.join(img_path_554, i)
    # print(save_img_path)
    cv2.imwrite(save_img_path, image)

for i in list_559:
    img_token_path = os.path.join(img_path, i)
    # print(img_token_path)
    image = cv2.imread(img_token_path)
    save_img_path = os.path.join(img_path_559, i)
    # print(save_img_path)
    cv2.imwrite(save_img_path, image)

for i in list_570:
    img_token_path = os.path.join(img_path, i)
    # print(img_token_path)
    image = cv2.imread(img_token_path)
    save_img_path = os.path.join(img_path_570, i)
    # print(save_img_path)
    cv2.imwrite(save_img_path, image)

for i in list_581:
    img_token_path = os.path.join(img_path, i)
    # print(img_token_path)
    image = cv2.imread(img_token_path)
    save_img_path = os.path.join(img_path_581, i)
    # print(save_img_path)
    cv2.imwrite(save_img_path, image)

for i in list_596:
    img_token_path = os.path.join(img_path, i)
    # print(img_token_path)
    image = cv2.imread(img_token_path)
    save_img_path = os.path.join(img_path_596, i)
    # print(save_img_path)
    cv2.imwrite(save_img_path, image)

for i in list_608:
    img_token_path = os.path.join(img_path, i)
    # print(img_token_path)
    image = cv2.imread(img_token_path)
    save_img_path = os.path.join(img_path_608, i)
    # print(save_img_path)
    cv2.imwrite(save_img_path, image)

for i in list_623:
    img_token_path = os.path.join(img_path, i)
    # print(img_token_path)
    image = cv2.imread(img_token_path)
    save_img_path = os.path.join(img_path_623, i)
    # print(save_img_path)
    cv2.imwrite(save_img_path, image)

for i in list_634:
    img_token_path = os.path.join(img_path, i)
    # print(img_token_path)
    image = cv2.imread(img_token_path)
    save_img_path = os.path.join(img_path_634, i)
    # print(save_img_path)
    cv2.imwrite(save_img_path, image)

for i in list_647:
    img_token_path = os.path.join(img_path, i)
    # print(img_token_path)
    image = cv2.imread(img_token_path)
    save_img_path = os.path.join(img_path_647, i)
    # print(save_img_path)
    cv2.imwrite(save_img_path, image)

for i in list_656:
    img_token_path = os.path.join(img_path, i)
    # print(img_token_path)
    image = cv2.imread(img_token_path)
    save_img_path = os.path.join(img_path_656, i)
    # print(save_img_path)
    cv2.imwrite(save_img_path, image)

for i in list_665:
    img_token_path = os.path.join(img_path, i)
    # print(img_token_path)
    image = cv2.imread(img_token_path)
    save_img_path = os.path.join(img_path_665, i)
    # print(save_img_path)
    cv2.imwrite(save_img_path, image)

for i in list_674:
    img_token_path = os.path.join(img_path, i)
    # print(img_token_path)
    image = cv2.imread(img_token_path)
    save_img_path = os.path.join(img_path_674, i)
    # print(save_img_path)
    cv2.imwrite(save_img_path, image)

for i in list_681:
    img_token_path = os.path.join(img_path, i)
    # print(img_token_path)
    image = cv2.imread(img_token_path)
    save_img_path = os.path.join(img_path_681, i)
    # print(save_img_path)
    cv2.imwrite(save_img_path, image)

for i in list_688:
    img_token_path = os.path.join(img_path, i)
    # print(img_token_path)
    image = cv2.imread(img_token_path)
    save_img_path = os.path.join(img_path_688, i)
    # print(save_img_path)
    cv2.imwrite(save_img_path, image)

for i in list_692:
    img_token_path = os.path.join(img_path, i)
    # print(img_token_path)
    image = cv2.imread(img_token_path)
    save_img_path = os.path.join(img_path_692, i)
    # print(save_img_path)
    cv2.imwrite(save_img_path, image)

for i in list_696:
    img_token_path = os.path.join(img_path, i)
    # print(img_token_path)
    image = cv2.imread(img_token_path)
    save_img_path = os.path.join(img_path_696, i)
    # print(save_img_path)
    cv2.imwrite(save_img_path, image)

for i in list_699:
    img_token_path = os.path.join(img_path, i)
    # print(img_token_path)
    image = cv2.imread(img_token_path)
    save_img_path = os.path.join(img_path_699, i)
    # print(save_img_path)
    cv2.imwrite(save_img_path, image)

for i in list_700:
    img_token_path = os.path.join(img_path, i)
    # print(img_token_path)
    image = cv2.imread(img_token_path)
    save_img_path = os.path.join(img_path_700, i)
    # print(save_img_path)
    cv2.imwrite(save_img_path, image)