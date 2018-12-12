import numpy as np
import matplotlib.pyplot as plt
from multi_task_data_CDR import Prepare_Task_Data_2
import scipy.stats as stats
import csv

day_flag = True
week_flag = True

path = '/home/qiuhui/Desktop/howard/downloaded_data/Milano_WeatherPhenomena/mi_meteo_6129_full.csv'
wind_speed = []
with open(path, newline='') as csvfile:
	rows = csv.reader(csvfile)
	for row in rows:
		wind_speed.append(float(row[2]))
X_array = wind_speed

TK2 = Prepare_Task_Data_2('/home/qiuhui/Desktop/bo/predict_telecom_traffic/CNN_RNN/npy/final/') ##fusion
_X_array, _Y_array = TK2.Task_avg(generate_data=False) ##fusion
_X_array = _Y_array
_X_array = _X_array[:1464, :, :, :, :]

_X_array = _X_array.reshape((-1, 15, 15, 3))

if day_flag == True:
	count = 0
	hour_sum = 0
	X_array = []
	CDR = [[[] for i in range(15)] for j in range(15)]
	for ele in wind_speed:
		count += 1
		if count == 24:
			X_array.append((hour_sum + ele) / 24)
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
					CDR[i][j].append((hour_sum + ele) / 24)
					hour_sum = 0
					count = 0
				else:
					hour_sum += ele
			count = 0

corr = np.zeros((15, 15))
for i in range(15):
	for j in range(15):
		#CDR = _X_array[:, i, j, -1]
		corr[i][j], _ = stats.pearsonr(X_array, CDR[i][j][:])
plt.figure()
plt.imshow(corr, cmap = plt.cm.gray, vmin = -1, vmax = 0.2)
plt.colorbar()
plt.title('Correlation coefficient of CDR and Wind Speed(each day)')

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

corr = np.zeros((15, 15))
for i in range(15):
	for j in range(15):
		#CDR = _X_array[:, i, j, -1]
		corr[i][j], _ = stats.pearsonr(X_array, CDR[i][j][:])
plt.figure()
plt.imshow(corr, cmap = plt.cm.gray, vmin = -1, vmax = 0.2)
plt.colorbar()
plt.title('Correlation coefficient of CDR and Wind Speed(each week)')

corr = np.zeros((15, 15))
for i in range(15):
	for j in range(15):
		CDR = _X_array[:, i, j, -1]
		corr[i][j], _ = stats.pearsonr(wind_speed, CDR)
plt.figure()
plt.imshow(corr, cmap = plt.cm.gray, vmin = -1, vmax = 0.2)
plt.colorbar()
plt.title('Correlation coefficient of CDR and Wind Speed(each hour)')
plt.show()