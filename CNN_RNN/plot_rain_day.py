import csv #匯入csv模組
import datetime, re
import matplotlib.pyplot as plt
import numpy as np

day_flag = True
week_flag = True

with open('precipitation.csv', errors='replace') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',') #以模組csv裡的函數reader來讀取csvfile變數，區隔符號為逗號(,)，讀取後存到readCSV變數裡
        coverage = []
        time = [0, 0, 0, 0]
        range1 = []
        for i in range(4):
            coverage.append([])
        i = 0
        for row in readCSV: 	#就readCSV裡的所有資料(以列為單位)
            i = int(row[1])
            coverage[i - 1].append(row[3])
            if int(row[3]) > 0:
                range1.append(int(row[3]))
                time[i - 1] += 1
for i in range(4):
    print('rainy time in region {}: {}'.format(i, time[i - 1]))
avg = sum(range1) / len(range1)
print('avg coverage value: {}'.format(avg))

small= 0
big = 0
for ele in range1:
    if ele > avg:
        small += 1
    else:
        big += 1
print('num that bigger than avg num:{}'.format(big))
print('num that smaller than avg num:{}'.format(small))

plt_list = []
smallest = 100
for count in range(1, 100):
    num = range1.count(count)
    #print('count of {}: {}'.format(count, num))
    plt_list.append(num)
    if num < smallest and num != 0:
        smallest = num
print('smallest: {}'.format(smallest))

#n = 61			#柱子數量
#X = np.arange(n)
Y = coverage

#plt.figure(num = "rainy day")		#畫新聞數-日期的圖
#plt.bar(X, Y, facecolor='#9999ff', edgecolor='white')
plt.figure()
plt.plot(Y[0])

plt.figure()
plt.plot(Y[1])

plt.figure()
plt.plot(Y[2])

plt.figure()
plt.plot(Y[3])
plt.show()

plt.plot(plt_list)
plt.show()

'''
for x, y in zip(X, Y):
    # ha: horizontal alignment
    # va: vertical alignment
        plt.text(x + 0.4, y + 0.05, '%.0f' % y, ha='center', va='bottom')
plt.xticks([0.4, 29.3, 60.35], ['2013-11-01', '2013-11-30', '2013-12-31'])
plt.yticks(())
'''
'''
plt.show()
j = 0

for i in range(1, 6):		#顯示5張新聞數-小時的圖，2013/11/01由開始
        n = 24			#柱子數量
        X = np.arange(n)
        Y = time[j:j+24]

        plt.figure(num = i)		#畫新聞數-小時的圖
        plt.bar(X, Y, facecolor='#9999ff', edgecolor='white')
        plt.xlabel("time(hour)")
        plt.ylabel("amount of news(piece)")

        for x, y in zip(X, Y):
            # ha: horizontal alignment
            # va: vertical alignment
                plt.text(x + 0.4, y + 0.05, '%.0f' % y, ha='center', va='bottom')
        plt.xticks([0.4, 8.45, 12.4, 17.45, 23.45], ['0', '8', '12', '17', '23'])
        plt.yticks(())
        j = j + 24
plt.show()
'''