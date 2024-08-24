import math

rate_list = [14, 60, 131, 206, 272, 324, 368, 399, 417, 420,
             410, 386, 348, 299, 237, 165, 89, 28, 3]
capacity_list = [420, 1800, 3930, 6180, 8160, 9720, 11040, 11970, 12510, 12600,
                 12300, 11580, 10440, 8970, 7110, 4950, 2670, 840, 90]

imagesize_list1 = [277, 285, 298, 311.5, 323.5, 332.5, 340.5, 346, 349.5, 350, 348, 344, 337, 328, 317, 304, 290.5, 279.5, 275]
imagesize_list2 = [x-25 for x in imagesize_list1]
imagesize_list3 = [x-25 for x in imagesize_list2]
imagesize_list4 = [x-25 for x in imagesize_list3]
imagesize_list5 = [x-25 for x in imagesize_list4]
imagesize_list6 = [x-25 for x in imagesize_list5]
imagesize_list7 = [x-25 for x in imagesize_list6]
# imagesize_list = [x-25 for x in imagesize_list7]
# print(imagesize_list7)
imagesize_list = [x/2 for x in imagesize_list7]
print(imagesize_list)
# imagesize_list = [111.75, 112.8125, 121.3125, 129.875, 131.375, 139.375, 140.375, 141.0625, 148.375, 148.4375,
#                   148.1875, 140.8125, 139.9375, 131.9375, 130.5625, 122.0625, 113.5, 112.125, 111.25]
# imagesize_list = [113.0, 114.6875, 123.1875, 130.8125, 132.3125, 142.1875, 143.1875, 143.875, 151.1875, 151.25, 151.0, 143.625, 142.75, 132.875, 131.5, 123.9375, 115.375, 113.625, 112.5]
# imagesize_list = [108.0, 109.0625, 117.5625, 126.125, 127.625, 135.625, 136.625, 137.3125, 144.625, 144.6875, 144.4375, 137.0625, 136.1875, 128.1875, 126.8125, 118.3125, 109.75, 108.375, 107.5]
# imagesize_list = [101.75, 103.4375, 111.9375, 120.375, 122.0, 130.625, 131.625, 132.3125, 139.625, 139.6875, 139.4375, 132.0625, 131.1875, 122.5625, 121.1875, 112.6875, 104.125, 102.375, 101.25]
# imagesize_list = [92.4375, 93.4375, 101.9375, 110.5, 112.0, 120.625, 121.625, 122.3125, 129.625, 129.6875, 129.4375, 122.0625, 121.1875, 112.5625, 111.1875, 102.6875, 94.125, 92.75, 92.1875]
# imagesize_list = [98.0, 99.0625, 107.5625, 116.125, 117.625, 126.25, 127.25, 127.9375, 135.25, 135.3125, 135.0625, 127.6875, 126.8125, 118.1875, 116.8125, 108.3125, 99.75, 98.375, 97.5]
# imagesize_list = [96.1875, 97.1875, 103.5, 114.25, 115.75, 124.375, 125.375, 126.0625, 131.1875, 131.25, 131.0, 125.8125, 124.9375, 116.3125, 114.9375, 104.25, 97.875, 96.5, 95.9375]


num_total = 0
time_total = 0

surplus = 0
for i in range(18):
    print("index: ", i)
    num1 = math.floor((capacity_list[i]-surplus)/imagesize_list[i])
    num_total = num_total + num1
    remain = capacity_list[i] - num1 * imagesize_list[i] - surplus
    # print("remain: ", remain)
    time1 = round(imagesize_list[i]/rate_list[i], 3)
    num1time1 = round(num1 * time1, 3)
    print("num1: ", num1, ", time1: ", time1)
    if remain>0:
        surplus = imagesize_list[i] - remain
        if i==17:
            if surplus<=90:
                num2 = 1
                num_total = num_total + num2
                # print("surplus: ", surplus)
                time2 = round((round(remain / rate_list[i], 3) + round(surplus / rate_list[i + 1], 3)), 3)
                num2time2 = round(num2*time2,3)
                print("num2: ", num2, ", time2: ", time2)
                time12 = round((num1time1+num2time2),3)
                print("time1+time2: ", time12)
                time_total = time_total + time12
            else:
                print("time1+time2: ", num1time1)
                time_total = time_total + num1time1
        else:
            num2 = 1
            num_total = num_total + num2
            # print("surplus: ", surplus)
            time2 = round((round(remain / rate_list[i], 3) + round(surplus / rate_list[i + 1], 3)), 3)
            num2time2 = round(num2 * time2, 3)
            print("num2: ", num2, ", time2: ", time2)
            time12 = round((num1time1 + num2time2), 3)
            print("time1+time2: ", time12)
            time_total = time_total + time12

print("num total: ", num_total)
print("time total: ", time_total)