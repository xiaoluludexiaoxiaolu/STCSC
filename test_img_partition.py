import os
import cv2

img_path = "test_img_token_complexity/test_img_token_0.6"
img = sorted(os.listdir(img_path))
'''
img_path_254 = "test_img_partiton_token/unthreshold7/254"
img_path_259 = "test_img_partiton_token/unthreshold7/259"
img_path_270 = "test_img_partiton_token/unthreshold7/270"
img_path_281 = "test_img_partiton_token/unthreshold7/281"
img_path_296 = "test_img_partiton_token/unthreshold7/296"
img_path_308 = "test_img_partiton_token/unthreshold7/308"
img_path_323 = "test_img_partiton_token/unthreshold7/323"
img_path_334 = "test_img_partiton_token/unthreshold7/334"
img_path_347 = "test_img_partiton_token/unthreshold7/347"
img_path_356 = "test_img_partiton_token/unthreshold7/356"
img_path_365 = "test_img_partiton_token/unthreshold7/365"
img_path_374 = "test_img_partiton_token/unthreshold7/374"
img_path_381 = "test_img_partiton_token/unthreshold7/381"
img_path_388 = "test_img_partiton_token/unthreshold7/388"
img_path_392 = "test_img_partiton_token/unthreshold7/392"
img_path_396 = "test_img_partiton_token/unthreshold7/396"
img_path_399 = "test_img_partiton_token/unthreshold7/399"
img_path_400 = "test_img_partiton_token/unthreshold7/400"

list_254_259 = img[0: 84]
list_254 = list_254_259[0::2][0:36]
print(len(list_254))
for i in list_254:
    list_254_259.remove(i)
list_259 = list_254_259
print(len(list_259))

for i in list_254:
    img_token_path = os.path.join(img_path, i)
    # print(img_token_path)
    image = cv2.imread(img_token_path)
    save_img_path = os.path.join(img_path_254, i)
    # print(save_img_path)
    cv2.imwrite(save_img_path, image)

for i in list_259:
    img_token_path = os.path.join(img_path, i)
    # print(img_token_path)
    image = cv2.imread(img_token_path)
    save_img_path = os.path.join(img_path_259, i)
    # print(save_img_path)
    cv2.imwrite(save_img_path, image)


list_270_281 = img[84: 353]
list_270 = list_270_281[0::2][0:117]
print(len(list_270))
for i in list_270:
    list_270_281.remove(i)
list_281 = list_270_281
print(len(list_281))

for i in list_270:
    img_token_path = os.path.join(img_path, i)
    # print(img_token_path)
    image = cv2.imread(img_token_path)
    save_img_path = os.path.join(img_path_270, i)
    # print(save_img_path)
    cv2.imwrite(save_img_path, image)

for i in list_281:
    img_token_path = os.path.join(img_path, i)
    # print(img_token_path)
    image = cv2.imread(img_token_path)
    save_img_path = os.path.join(img_path_281, i)
    # print(save_img_path)
    cv2.imwrite(save_img_path, image)


list_296_308 = img[353: 852]
list_296 = list_296_308[0::2][0:243]
print(len(list_296))
for i in list_296:
    list_296_308.remove(i)
list_308 = list_296_308
print(len(list_308))

for i in list_296:
    img_token_path = os.path.join(img_path, i)
    # print(img_token_path)
    image = cv2.imread(img_token_path)
    save_img_path = os.path.join(img_path_296, i)
    # print(save_img_path)
    cv2.imwrite(save_img_path, image)

for i in list_308:
    img_token_path = os.path.join(img_path, i)
    # print(img_token_path)
    image = cv2.imread(img_token_path)
    save_img_path = os.path.join(img_path_308, i)
    # print(save_img_path)
    cv2.imwrite(save_img_path, image)


list_323_334 = img[852: 1538]
list_323 = list_323_334[0::2][0:342]
print(len(list_323))
for i in list_323:
    list_323_334.remove(i)
list_334 = list_323_334
print(len(list_334))

for i in list_323:
    img_token_path = os.path.join(img_path, i)
    # print(img_token_path)
    image = cv2.imread(img_token_path)
    save_img_path = os.path.join(img_path_323, i)
    # print(save_img_path)
    cv2.imwrite(save_img_path, image)

for i in list_334:
    img_token_path = os.path.join(img_path, i)
    # print(img_token_path)
    image = cv2.imread(img_token_path)
    save_img_path = os.path.join(img_path_334, i)
    # print(save_img_path)
    cv2.imwrite(save_img_path, image)


list_347_356 = img[1538: 2361]
list_356 = list_347_356[1::2][0:400]
print(len(list_356))
for i in list_356:
    list_347_356.remove(i)
list_347 = list_347_356
print(len(list_347))

for i in list_347:
    img_token_path = os.path.join(img_path, i)
    # print(img_token_path)
    image = cv2.imread(img_token_path)
    save_img_path = os.path.join(img_path_347, i)
    # print(save_img_path)
    cv2.imwrite(save_img_path, image)

for i in list_356:
    img_token_path = os.path.join(img_path, i)
    # print(img_token_path)
    image = cv2.imread(img_token_path)
    save_img_path = os.path.join(img_path_356, i)
    # print(save_img_path)
    cv2.imwrite(save_img_path, image)


list_365_374 = img[2361: 3286]
list_374 = list_365_374[1::2][0:448]
print(len(list_374))
for i in list_374:
    list_365_374.remove(i)
list_365 = list_365_374
print(len(list_365))

for i in list_365:
    img_token_path = os.path.join(img_path, i)
    # print(img_token_path)
    image = cv2.imread(img_token_path)
    save_img_path = os.path.join(img_path_365, i)
    # print(save_img_path)
    cv2.imwrite(save_img_path, image)

for i in list_374:
    img_token_path = os.path.join(img_path, i)
    # print(img_token_path)
    image = cv2.imread(img_token_path)
    save_img_path = os.path.join(img_path_374, i)
    # print(save_img_path)
    cv2.imwrite(save_img_path, image)


list_381_388 = img[3286: 4288]
list_388 = list_381_388[1::2][0:480]
print(len(list_388))
for i in list_388:
    list_381_388.remove(i)
list_381 = list_381_388
print(len(list_381))

for i in list_381:
    img_token_path = os.path.join(img_path, i)
    # print(img_token_path)
    image = cv2.imread(img_token_path)
    save_img_path = os.path.join(img_path_381, i)
    # print(save_img_path)
    cv2.imwrite(save_img_path, image)

for i in list_388:
    img_token_path = os.path.join(img_path, i)
    # print(img_token_path)
    image = cv2.imread(img_token_path)
    save_img_path = os.path.join(img_path_388, i)
    # print(save_img_path)
    cv2.imwrite(save_img_path, image)


list_392_396 = img[4288: 5292]
list_396 = list_392_396[1::2][0:496]
print(len(list_396))
for i in list_396:
    list_392_396.remove(i)
list_392 = list_392_396
print(len(list_392))

for i in list_392:
    img_token_path = os.path.join(img_path, i)
    # print(img_token_path)
    image = cv2.imread(img_token_path)
    save_img_path = os.path.join(img_path_392, i)
    # print(save_img_path)
    cv2.imwrite(save_img_path, image)

for i in list_396:
    img_token_path = os.path.join(img_path, i)
    # print(img_token_path)
    image = cv2.imread(img_token_path)
    save_img_path = os.path.join(img_path_396, i)
    # print(save_img_path)
    cv2.imwrite(save_img_path, image)


list_399_400 = img[5292: 6300]
list_399 = list_399_400[0::2][0:504]
print(len(list_399))
for i in list_399:
    list_399_400.remove(i)
list_400 = list_399_400
print(len(list_400))

for i in list_399:
    img_token_path = os.path.join(img_path, i)
    # print(img_token_path)
    image = cv2.imread(img_token_path)
    save_img_path = os.path.join(img_path_399, i)
    # print(save_img_path)
    cv2.imwrite(save_img_path, image)

for i in list_400:
    img_token_path = os.path.join(img_path, i)
    # print(img_token_path)
    image = cv2.imread(img_token_path)
    save_img_path = os.path.join(img_path_400, i)
    # print(save_img_path)
    cv2.imwrite(save_img_path, image)
'''
img_path_254 = "test_img_partiton_token/unresolution73/254"
img_path_259 = "test_img_partiton_token/unresolution73/259"
img_path_270 = "test_img_partiton_token/unresolution73/270"
img_path_281 = "test_img_partiton_token/unresolution73/281"
img_path_296 = "test_img_partiton_token/unresolution73/296"
img_path_308 = "test_img_partiton_token/unresolution73/308"
img_path_323 = "test_img_partiton_token/unresolution73/323"
img_path_334 = "test_img_partiton_token/unresolution73/334"
img_path_347 = "test_img_partiton_token/unresolution73/347"
img_path_356 = "test_img_partiton_token/unresolution73/356"
img_path_365 = "test_img_partiton_token/unresolution73/365"
img_path_374 = "test_img_partiton_token/unresolution73/374"
img_path_381 = "test_img_partiton_token/unresolution73/381"
img_path_388 = "test_img_partiton_token/unresolution73/388"
img_path_392 = "test_img_partiton_token/unresolution73/392"
img_path_396 = "test_img_partiton_token/unresolution73/396"
img_path_399 = "test_img_partiton_token/unresolution73/399"
img_path_400 = "test_img_partiton_token/unresolution73/400"

list_254_259 = img[0: 75]
list_254 = list_254_259[0::2][0:30]
print(len(list_254))
for i in list_254:
    list_254_259.remove(i)
list_259 = list_254_259
print(len(list_259))

for i in list_254:
    img_token_path = os.path.join(img_path, i)
    # print(img_token_path)
    image = cv2.imread(img_token_path)
    save_img_path = os.path.join(img_path_254, i)
    # print(save_img_path)
    cv2.imwrite(save_img_path, image)

for i in list_259:
    img_token_path = os.path.join(img_path, i)
    # print(img_token_path)
    image = cv2.imread(img_token_path)
    save_img_path = os.path.join(img_path_259, i)
    # print(save_img_path)
    cv2.imwrite(save_img_path, image)


list_270_281 = img[75: 313]
list_270 = list_270_281[0::2][0:108]
print(len(list_270))
for i in list_270:
    list_270_281.remove(i)
list_281 = list_270_281
print(len(list_281))

for i in list_270:
    img_token_path = os.path.join(img_path, i)
    # print(img_token_path)
    image = cv2.imread(img_token_path)
    save_img_path = os.path.join(img_path_270, i)
    # print(save_img_path)
    cv2.imwrite(save_img_path, image)

for i in list_281:
    img_token_path = os.path.join(img_path, i)
    # print(img_token_path)
    image = cv2.imread(img_token_path)
    save_img_path = os.path.join(img_path_281, i)
    # print(save_img_path)
    cv2.imwrite(save_img_path, image)


list_296_308 = img[313: 765]
list_296 = list_296_308[0::2][0:222]
print(len(list_296))
for i in list_296:
    list_296_308.remove(i)
list_308 = list_296_308
print(len(list_308))

for i in list_296:
    img_token_path = os.path.join(img_path, i)
    # print(img_token_path)
    image = cv2.imread(img_token_path)
    save_img_path = os.path.join(img_path_296, i)
    # print(save_img_path)
    cv2.imwrite(save_img_path, image)

for i in list_308:
    img_token_path = os.path.join(img_path, i)
    # print(img_token_path)
    image = cv2.imread(img_token_path)
    save_img_path = os.path.join(img_path_308, i)
    # print(save_img_path)
    cv2.imwrite(save_img_path, image)


list_323_334 = img[765: 1388]
list_334 = list_323_334[1::2][0:305]
print(len(list_334))
for i in list_334:
    list_323_334.remove(i)
list_323 = list_323_334
print(len(list_323))

for i in list_323:
    img_token_path = os.path.join(img_path, i)
    # print(img_token_path)
    image = cv2.imread(img_token_path)
    save_img_path = os.path.join(img_path_323, i)
    # print(save_img_path)
    cv2.imwrite(save_img_path, image)

for i in list_334:
    img_token_path = os.path.join(img_path, i)
    # print(img_token_path)
    image = cv2.imread(img_token_path)
    save_img_path = os.path.join(img_path_334, i)
    # print(save_img_path)
    cv2.imwrite(save_img_path, image)


list_347_356 = img[1388: 2182]
list_356 = list_347_356[1::2][0:380]
print(len(list_356))
for i in list_356:
    list_347_356.remove(i)
list_347 = list_347_356
print(len(list_347))

for i in list_347:
    img_token_path = os.path.join(img_path, i)
    # print(img_token_path)
    image = cv2.imread(img_token_path)
    save_img_path = os.path.join(img_path_347, i)
    # print(save_img_path)
    cv2.imwrite(save_img_path, image)

for i in list_356:
    img_token_path = os.path.join(img_path, i)
    # print(img_token_path)
    image = cv2.imread(img_token_path)
    save_img_path = os.path.join(img_path_356, i)
    # print(save_img_path)
    cv2.imwrite(save_img_path, image)


list_365_374 = img[2182: 3054]
list_374 = list_365_374[1::2][0:410]
print(len(list_374))
for i in list_374:
    list_365_374.remove(i)
list_365 = list_365_374
print(len(list_365))

for i in list_365:
    img_token_path = os.path.join(img_path, i)
    # print(img_token_path)
    image = cv2.imread(img_token_path)
    save_img_path = os.path.join(img_path_365, i)
    # print(save_img_path)
    cv2.imwrite(save_img_path, image)

for i in list_374:
    img_token_path = os.path.join(img_path, i)
    # print(img_token_path)
    image = cv2.imread(img_token_path)
    save_img_path = os.path.join(img_path_374, i)
    # print(save_img_path)
    cv2.imwrite(save_img_path, image)


list_381_388 = img[3054: 4080]
list_388 = list_381_388[1::2][0:504]
print(len(list_388))
for i in list_388:
    list_381_388.remove(i)
list_381 = list_381_388
print(len(list_381))

for i in list_381:
    img_token_path = os.path.join(img_path, i)
    # print(img_token_path)
    image = cv2.imread(img_token_path)
    save_img_path = os.path.join(img_path_381, i)
    # print(save_img_path)
    cv2.imwrite(save_img_path, image)

for i in list_388:
    img_token_path = os.path.join(img_path, i)
    # print(img_token_path)
    image = cv2.imread(img_token_path)
    save_img_path = os.path.join(img_path_388, i)
    # print(save_img_path)
    cv2.imwrite(save_img_path, image)


list_392_396 = img[4080: 5190]
list_396 = list_392_396[1::2][0:546]
print(len(list_396))
for i in list_396:
    list_392_396.remove(i)
list_392 = list_392_396
print(len(list_392))

for i in list_392:
    img_token_path = os.path.join(img_path, i)
    # print(img_token_path)
    image = cv2.imread(img_token_path)
    save_img_path = os.path.join(img_path_392, i)
    # print(save_img_path)
    cv2.imwrite(save_img_path, image)

for i in list_396:
    img_token_path = os.path.join(img_path, i)
    # print(img_token_path)
    image = cv2.imread(img_token_path)
    save_img_path = os.path.join(img_path_396, i)
    # print(save_img_path)
    cv2.imwrite(save_img_path, image)


list_399_400 = img[5190: 6300]
list_399 = list_399_400[0::2][0:552]
print(len(list_399))
for i in list_399:
    list_399_400.remove(i)
list_400 = list_399_400
print(len(list_400))

for i in list_399:
    img_token_path = os.path.join(img_path, i)
    # print(img_token_path)
    image = cv2.imread(img_token_path)
    save_img_path = os.path.join(img_path_399, i)
    # print(save_img_path)
    cv2.imwrite(save_img_path, image)

for i in list_400:
    img_token_path = os.path.join(img_path, i)
    # print(img_token_path)
    image = cv2.imread(img_token_path)
    save_img_path = os.path.join(img_path_400, i)
    # print(save_img_path)
    cv2.imwrite(save_img_path, image)
