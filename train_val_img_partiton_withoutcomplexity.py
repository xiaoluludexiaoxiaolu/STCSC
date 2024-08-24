import os
import cv2

img_path = "train_val_img"
img = sorted(os.listdir(img_path))

img_path_554 = "train_val_img_partiton_token/unthreshold7/254"
img_path_559 = "train_val_img_partiton_token/unthreshold7/259"
img_path_570 = "train_val_img_partiton_token/unthreshold7/270"
img_path_581 = "train_val_img_partiton_token/unthreshold7/281"
img_path_596 = "train_val_img_partiton_token/unthreshold7/296"
img_path_608 = "train_val_img_partiton_token/unthreshold7/308"
img_path_623 = "train_val_img_partiton_token/unthreshold7/323"
img_path_634 = "train_val_img_partiton_token/unthreshold7/334"
img_path_647 = "train_val_img_partiton_token/unthreshold7/347"
img_path_656 = "train_val_img_partiton_token/unthreshold7/356"
img_path_665 = "train_val_img_partiton_token/unthreshold7/365"
img_path_674 = "train_val_img_partiton_token/unthreshold7/374"
img_path_681 = "train_val_img_partiton_token/unthreshold7/381"
img_path_688 = "train_val_img_partiton_token/unthreshold7/388"
img_path_692 = "train_val_img_partiton_token/unthreshold7/392"
img_path_696 = "train_val_img_partiton_token/unthreshold7/396"
img_path_699 = "train_val_img_partiton_token/unthreshold7/399"
img_path_700 = "train_val_img_partiton_token/unthreshold7/400"

# len_list1 = [124, 372, 868, 1178, 1612, 1770, 1952, 2135, 2196,
#             2196, 2135, 2074, 1891, 1647, 1342, 1037, 549, 122]

# len_list2 = [114, 399, 855, 1197, 1539, 1824, 1995, 2109, 2223,
#             2193, 2128, 2016, 1904, 1624, 1344, 1008, 560, 168]

# len_list3 = [104, 416, 832, 1248, 1508, 1820, 1976, 2080, 2184,
#             2184, 2132, 2080, 1872, 1653, 1377, 969, 612, 153]

# len_list4 = [144, 384, 864, 1248, 1584, 1824, 1950, 2068, 2162,
#             2162, 2115, 2021, 1880, 1645, 1410, 987, 564, 188]

# len_list5 = [129, 430, 860, 1247, 1548, 1806, 1978, 2107, 2150,
#             2150, 2143, 1974, 1848, 1680, 1344, 1050, 588, 168]

# len_list6 = [117, 459, 836, 1254, 1596, 1748, 1976, 2052, 2128,
#             2128, 2090, 2014, 1862, 1672, 1406, 1064, 608, 190]

len_list = [136, 442, 918, 1292, 1598, 1802, 1972, 2058, 2079,
            2079, 2046, 1980, 1848, 1650, 1419, 1056, 627, 198]

# len_list8 = [145, 464, 928, 1305, 1595, 1798, 1943, 2030, 2088,
#             2088, 2059, 1972, 1829, 1624, 1400, 1064, 672, 196]

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