import os
import cv2

img_path = "train_val_img_token_complexity/train_val_img_token_0.6"
img = sorted(os.listdir(img_path))
'''
img_path_254 = "train_val_img_partiton_token/unthreshold7/254"
img_path_259 = "train_val_img_partiton_token/unthreshold7/259"
img_path_270 = "train_val_img_partiton_token/unthreshold7/270"
img_path_281 = "train_val_img_partiton_token/unthreshold7/281"
img_path_296 = "train_val_img_partiton_token/unthreshold7/296"
img_path_308 = "train_val_img_partiton_token/unthreshold7/308"
img_path_323 = "train_val_img_partiton_token/unthreshold7/323"
img_path_334 = "train_val_img_partiton_token/unthreshold7/334"
img_path_347 = "train_val_img_partiton_token/unthreshold7/347"
img_path_356 = "train_val_img_partiton_token/unthreshold7/356"
img_path_365 = "train_val_img_partiton_token/unthreshold7/365"
img_path_374 = "train_val_img_partiton_token/unthreshold7/374"
img_path_381 = "train_val_img_partiton_token/unthreshold7/381"
img_path_388 = "train_val_img_partiton_token/unthreshold7/388"
img_path_392 = "train_val_img_partiton_token/unthreshold7/392"
img_path_396 = "train_val_img_partiton_token/unthreshold7/396"
img_path_399 = "train_val_img_partiton_token/unthreshold7/399"
img_path_400 = "train_val_img_partiton_token/unthreshold7/400"

list_254_259 = img[0: 334]
list_254 = list_254_259[0::2][0:136]
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


list_270_281 = img[334: 1403]
list_270 = list_270_281[0::2][0:442]
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


list_296_308 = img[1403: 3377]
list_296 = list_296_308[0::2][0:918]
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


list_323_334 = img[3377: 6088]
list_323 = list_323_334[0::2][0:1292]
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


list_347_356 = img[6088: 9336]
list_347 = list_347_356[0::2][0:1598]
print(len(list_347))
for i in list_347:
    list_347_356.remove(i)
list_356 = list_347_356
print(len(list_356))

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


list_365_374 = img[9336: 12986]
list_365 = list_365_374[0::2][0:1802]
print(len(list_365))
for i in list_365:
    list_365_374.remove(i)
list_374 = list_365_374
print(len(list_374))

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


list_381_388 = img[12986: 16938]
list_381 = list_381_388[0::2][0:1972]
print(len(list_381))
for i in list_381:
    list_381_388.remove(i)
list_388 = list_381_388
print(len(list_388))

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


list_392_396 = img[16938: 21042]
list_396 = list_392_396[1::2][0:2046]
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


list_399_400 = img[21042: 25200]
list_399 = list_399_400[0::2][0:2079]
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
img_path_254 = "train_val_img_partiton_token/unresolution73/254"
img_path_259 = "train_val_img_partiton_token/unresolution73/259"
img_path_270 = "train_val_img_partiton_token/unresolution73/270"
img_path_281 = "train_val_img_partiton_token/unresolution73/281"
img_path_296 = "train_val_img_partiton_token/unresolution73/296"
img_path_308 = "train_val_img_partiton_token/unresolution73/308"
img_path_323 = "train_val_img_partiton_token/unresolution73/323"
img_path_334 = "train_val_img_partiton_token/unresolution73/334"
img_path_347 = "train_val_img_partiton_token/unresolution73/347"
img_path_356 = "train_val_img_partiton_token/unresolution73/356"
img_path_365 = "train_val_img_partiton_token/unresolution73/365"
img_path_374 = "train_val_img_partiton_token/unresolution73/374"
img_path_381 = "train_val_img_partiton_token/unresolution73/381"
img_path_388 = "train_val_img_partiton_token/unresolution73/388"
img_path_392 = "train_val_img_partiton_token/unresolution73/392"
img_path_396 = "train_val_img_partiton_token/unresolution73/396"
img_path_399 = "train_val_img_partiton_token/unresolution73/399"
img_path_400 = "train_val_img_partiton_token/unresolution73/400"

list_254_259 = img[0: 313]
list_254 = list_254_259[0::2][0:115]
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


list_270_281 = img[313: 1299]
list_270 = list_270_281[0::2][0:414]
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


list_296_308 = img[1299: 3162]
list_296 = list_296_308[0::2][0:851]
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


list_323_334 = img[3162: 5723]
list_323 = list_323_334[0::2][0:1219]
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


list_347_356 = img[5723: 8982]
list_347 = list_347_356[0::2][0:1587]
print(len(list_347))
for i in list_347:
    list_347_356.remove(i)
list_356 = list_347_356
print(len(list_356))

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


list_365_374 = img[8982: 12596]
list_365 = list_365_374[0::2][0:1771]
print(len(list_365))
for i in list_365:
    list_365_374.remove(i)
list_374 = list_365_374
print(len(list_374))

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


list_381_388 = img[12596: 16690]
list_381 = list_381_388[0::2][0:2001]
print(len(list_381))
for i in list_381:
    list_381_388.remove(i)
list_388 = list_381_388
print(len(list_388))

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


list_392_396 = img[16690: 20945]
list_396 = list_392_396[1::2][0:2093]
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


list_399_400 = img[20945: 25200]
list_399 = list_399_400[0::2][0:2116]
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
