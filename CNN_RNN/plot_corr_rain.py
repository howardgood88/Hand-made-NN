import numpy as np
import matplotlib.pyplot as plt
from multi_task_data import Prepare_Task_Data
from multi_task_data_CDR import Prepare_Task_Data_2
import scipy.stats as stats

day_flag = True
week_flag = True

TK = Prepare_Task_Data('./CNN_RNN/npy/final')
TK2 = Prepare_Task_Data_2('/home/qiuhui/Desktop/bo/predict_telecom_traffic/CNN_RNN/npy/final/') ##fusion
X_array, Y_array = TK.Task_avg(generate_data=False)
wind_speed = Y_array
_X_array, _Y_array = TK2.Task_avg(generate_data=False) ##fusion
_X_array = _Y_array
_X_array = _X_array[:1464, :, :, :, :]

wind_speed = wind_speed.reshape((-1, 15, 15, 3))
_X_array = _X_array.reshape((-1, 15, 15, 3))
print(X_array.shape)

if day_flag == True:
	count = 0
	hour_sum = 0
	X_array = [[[] for i in range(15)] for j in range(15)]
	CDR = [[[] for i in range(15)] for j in range(15)]
	for i in range(15):
		for j in range(15):
			for k in range(wind_speed.shape[0]):
				ele = wind_speed[k, i, j, -1]
				count += 1
				if count == 24:
					X_array[i][j].append((hour_sum + ele) / 24)
					hour_sum = 0
					count = 0
				else:
					hour_sum += ele
			count = 0
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
		corr[i][j], _ = stats.pearsonr(X_array[i][j][:], CDR[i][j][:])
plt.figure()
plt.imshow(corr, cmap = plt.cm.gray)
plt.colorbar()
plt.title('Correlation coefficient of CDR and rain coverage(each day)')

if week_flag == True:
	count = 0
	hour_sum = 0
	X_array = [[[] for i in range(15)] for j in range(15)]
	CDR = [[[] for i in range(15)] for j in range(15)]
	for i in range(15):
		for j in range(15):
			for k in range(wind_speed.shape[0]):
				ele = wind_speed[k, i, j, -1]
				count += 1
				if count == 168:
					X_array[i][j].append((hour_sum + ele) / 168)
					hour_sum = 0
					count = 0
				else:
					hour_sum += ele
			count = 0
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
		corr[i][j], _ = stats.pearsonr(X_array[i][j][:], CDR[i][j][:])
plt.figure()
plt.imshow(corr, cmap = plt.cm.gray, vmin = -1, vmax = 0.2)
plt.colorbar()
plt.title('Correlation coefficient of CDR and rain coverage(each week)')

corr = np.zeros((15, 15))
for i in range(15):
	for j in range(15):
		rain = wind_speed[:, i, j, -1]
		CDR = _X_array[:, i, j, -1]
		corr[i][j], _ = stats.pearsonr(rain, CDR)
plt.figure()
plt.imshow(corr, cmap = plt.cm.gray)
plt.colorbar()
plt.title('Correlation coefficient of CDR and rain coverage(each hour)')
plt.show()

'''
for t in range(X_array.shape[0]):
	for i in range(15):
		for j in range(15):
			if X_array[t, i, j, -1] > 0:
				X_array[t, i, j, -1] = 1

corr = np.zeros((15, 15))
for i in range(15):
	for j in range(15):
		rain = X_array[:, i, j, -1]
		CDR = _X_array[:, i, j, -1]
		corr[i][j], _ = stats.pearsonr(rain, CDR)
plt.figure()
plt.imshow(corr, cmap = plt.cm.gray)
plt.colorbar()
plt.title('Correlation coefficient of CDR and rain \nin two months(61days)(rain or not)')
'''