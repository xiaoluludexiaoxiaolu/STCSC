rate_list = [14, 60, 131, 206, 272, 324, 368, 399, 417, 420,
             410, 386, 348, 299, 237, 165, 89, 28, 3]
token_list6 = [304, 320, 346, 373, 397, 415, 431, 442, 449, 450,
              446, 438, 424, 406, 384, 358, 331, 309, 300]
token_list = [x-50 for x in token_list6]
# print(token_list)

thre1_list = [150, 70, 25]
# print(sum(thre1_list))  # 305
thre2_list = [155, 80, 35]
# print(sum(thre2_list))  # 335
thre3_list = [170, 90, 45]
# print(sum(thre3_list))  # 365
thre4_list = [180, 100, 65]
# print(sum(thre4_list))  # 395
thre5_list = [185, 110, 75]
# print(sum(thre5_list))  # 425

image_list = []
for i in range(19):
    # print(i)
    if rate_list[i]<100:
        if sum(thre1_list)>=token_list[i]:
            image_size = 512 * thre1_list[0] + 256 * thre1_list[1] + 128 * (
                    token_list[i]-thre1_list[0]-thre1_list[1])
            image_size = image_size/1024.0
        else:
            image_size = 512 * thre1_list[0] + 256 * thre1_list[1] + 128 * thre1_list[2] + 64 * (
                    token_list[i]-thre1_list[0]-thre1_list[1]-thre1_list[2])
            image_size = image_size/1024.0
    elif rate_list[i]>=100 and rate_list[i]<200:
        if sum(thre2_list)>=token_list[i]:
            image_size = 512 * thre2_list[0] + 256 * thre2_list[1] + 128 * (
                    token_list[i]-thre2_list[0]-thre2_list[1])
            image_size = image_size/1024.0
        else:
            image_size = 512 * thre2_list[0] + 256 * thre2_list[1] + 128 * thre2_list[2] + 64 * (
                    token_list[i] - thre2_list[0] - thre2_list[1] - thre2_list[2])
            image_size = image_size / 1024.0
    elif rate_list[i]>=200 and rate_list[i]<300:
        if sum(thre3_list)>=token_list[i]:
            image_size = 512 * thre3_list[0] + 256 * thre3_list[1] + 128 * (
                    token_list[i]-thre3_list[0]-thre3_list[1])
            image_size = image_size/1024.0
        else:
            image_size = 512 * thre3_list[0] + 256 * thre3_list[1] + 128 * thre3_list[2] + 64 * (
                    token_list[i] - thre3_list[0] - thre3_list[1] - thre3_list[2])
            image_size = image_size / 1024.0
    elif rate_list[i]>=300 and rate_list[i]<400:
        if sum(thre4_list)>=token_list[i]:
            image_size = 512 * thre4_list[0] + 256 * thre4_list[1] + 128 * (
                    token_list[i]-thre4_list[0]-thre4_list[1])
            image_size = image_size/1024.0
        else:
            image_size = 512 * thre4_list[0] + 256 * thre4_list[1] + 128 * thre4_list[2] + 64 * (
                    token_list[i] - thre4_list[0] - thre4_list[1] - thre4_list[2])
            image_size = image_size / 1024.0
    else:
        if sum(thre5_list)>=token_list[i]:
            image_size = 512 * thre5_list[0] + 256 * thre5_list[1] + 128 * (
                    token_list[i]-thre5_list[0]-thre5_list[1])
            image_size = image_size/1024.0
        else:
            image_size = 512 * thre5_list[0] + 256 * thre5_list[1] + 128 * thre5_list[2] + 64 * (
                token_list[i] - thre5_list[0] - thre5_list[1] - thre5_list[2])
            image_size = image_size / 1024.0
    # print(image_size)
    image_list.append(image_size)

print(image_list)
print(len(image_list))