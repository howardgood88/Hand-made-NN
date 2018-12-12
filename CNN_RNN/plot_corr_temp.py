import numpy as np
import matplotlib.pyplot as plt
from multi_task_data_CDR import Prepare_Task_Data_2
import scipy.stats as stats
import csv

day_flag = True
week_flag = True

path = '/home/qiuhui/Desktop/howard/downloaded_data/Milano_WeatherPhenomena/mi_meteo_5911_full.csv'
wind_speed = []
with open(path, newline='') as csvfile:
	rows = csv.reader(csvfile)
	for row in rows:
		wind_speed.append(float(row[2]))
#X_array = wind_speed
	
TK2 = Prepare_Task_Data_2('/home/qiuhui/Desktop/bo/predict_telecom_traffic/CNN_RNN/npy/final/') ##fusion
_X_array, _Y_array = TK2.Task_avg(generate_data=False) ##fusion
_X_array = _Y_array
_X_array = _X_array[:1464, :, :, :, :]

_X_array = _X_array.reshape((-1, 15, 15, 3)) # 3 is timestamp, grid id, internet traffic

if day_flag == True:
	count = 0
	hour_sum = 0
	X_array_day = []
	CDR_day = [[[] for i in range(15)] for j in range(15)]
	for ele in wind_speed:
		count += 1
		if count == 24:
			X_array_day.append((hour_sum + ele) / 24)
			hour_sum = 0
			count = 0
		else:
			hour_sum += ele
	count = 0
	hour_sum = 0
	for i in range(15):
		for j in range(15):
			for k in range(_X_array.shape[0]):
				ele = _X_array[k, i, j, -1]
				count += 1
				if count == 24:
					CDR_day[i][j].append((hour_sum + ele) / 24)
					hour_sum = 0
					count = 0
				else:
					hour_sum += ele
X_array_day = np.array(X_array_day)
CDR_day = np.array(CDR_day)
corr = np.zeros((15, 15))
for i in range(15):
	for j in range(15):
		# CDR = _X_array[:, i, j, -1]
		corr[i][j], _ = stats.pearsonr(X_array_day, CDR_day[i, j, :])
plt.figure()
plt.imshow(corr, cmap = plt.cm.gray, vmin = 0, vmax = 1)
plt.colorbar()
plt.title('Correlation coefficient of CDR and temporature(each day)')

if week_flag == True:
	count = 0
	hour_sum = 0
	X_array = []
	CDR = [[[] for i in range(15)] for j in range(15)]
	for ele in wind_speed:
		count += 1
		if count == 168:
			X_array.append((hour_sum + ele) / 168)
			hour_sum = 0
			count = 0
		else:
			hour_sum += ele
	count = 0
	hour_sum = 0
	for i in range(15):
		for j in range(15):
			for k in range(_X_array.shape[0]):
				ele = _X_array[k, i, j, -1]
				count += 1
				if count == 168:
					CDR[i][j].append((hour_sum + ele) / 168)
					hour_sum = 0
					count = 0
				else:
					hour_sum += ele
			count = 0

X_array = np.array(X_array)
#CDR = np.array(CDR)
#print(X_array.shape, CDR.shape)
corr = np.zeros((15, 15))
for i in range(15):
	for j in range(15):
		# CDR = _X_array[:, i, j, -1]
		corr[i][j], _ = stats.pearsonr(X_array, CDR[i][j][:])
plt.figure()
plt.imshow(corr, cmap = plt.cm.gray, vmin = 0, vmax = 1)
plt.colorbar()
plt.title('Correlation coefficient of CDR and temporature(each week)')

maximum = [-2, -1, -1]
for i in range(15):
	for j in range(15):
		if corr[i][j] > maximum[0]:
			maximum = [corr[i][j], i, j]
plt.figure()
x = np.arange(0, 504, 24)
plt.xticks(x)
plt_wind = []
for ele in wind_speed:
	plt_wind.append((ele - np.min(wind_speed)) / (np.max(wind_speed) - np.min(wind_speed)))
plt_CDR = []
CDR_X = _X_array[:504, maximum[1], maximum[2], -1]
for ele in CDR_X:
	plt_CDR.append((ele - np.min(CDR_X)) / (np.max(CDR_X) - np.min(CDR_X)))
plt.plot(plt_wind[:504], '-o')
plt.plot(plt_CDR, '--o')
plt.title('the data of CDR and temporature(hour)')

plt.figure()
x = np.arange(0, 8, 1)
plt.xticks(x)
plt_wind = []
for ele in X_array_day:
	plt_wind.append((ele - np.min(X_array_day)) / (np.max(X_array_day) - np.min(X_array_day)))
plt_CDR = []
CDR_X = CDR_day[maximum[1]][maximum[2]][:]
for ele in CDR_X:
	plt_CDR.append((ele - np.min(CDR_X)) / (np.max(CDR_X) - np.min(CDR_X)))
plt.plot(plt_wind, '-o')
plt.plot(plt_CDR, '--o')
plt.title('the data of CDR and temporature(day)')

plt.figure()
x = np.arange(0, 8, 1)
plt.xticks(x)
plt_wind = []
for ele in X_array:
	plt_wind.append((ele - np.min(X_array)) / (np.max(X_array) - np.min(X_array)))
plt_CDR = []
CDR_X = CDR[maximum[1]][maximum[2]][:]
for ele in CDR_X:
	plt_CDR.append((ele - np.min(CDR_X)) / (np.max(CDR_X) - np.min(CDR_X)))
plt.plot(plt_wind, '-o')
plt.plot(plt_CDR, '--o')
plt.title('the data of CDR and temporature(week)')

corr = np.zeros((15, 15))
for i in range(15):
	for j in range(15):
		CDR = _X_array[:, i, j, -1]
		corr[i][j], _ = stats.pearsonr(wind_speed, CDR)
plt.figure()
plt.imshow(corr, cmap = plt.cm.gray, vmin = 0, vmax = 1)
plt.colorbar()
plt.title('Correlation coefficient of CDR and temporature(each hour)')
plt.show()