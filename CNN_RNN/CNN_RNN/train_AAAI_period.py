import numpy as np
from utility import feature_scaling, root_dir, set_time_zone, date_time_covert_to_str
import matplotlib.pyplot as plt
import sys
import CNN_RNN_config
from CNN_RNN_AAAI_period import CNN_RNN_2_test
import os
sys.path.append(root_dir)
import data_utility as du
from multi_task_data import Prepare_Task_Data
from multi_task_data_CDR import Prepare_Task_Data_2
import time
import tensorflow as tf
import tensorlayer as tl
import csv

# root_dir = '/home/mldp/ML_with_bigdata'

def print_Y_array(Y_array):
	print('Y array shape:{}'.format(Y_array.shape))
	plot_y_list = []
	for i in range(148):
		for j in range(Y_array.shape[1]):
			print(Y_array[i, j, 5, 10])
			plot_y_list.append(Y_array[i, j, 5, 10])
	plt.figure()
	plt.plot(plot_y_list, marker='.')
	plt.show()

def time_feature(X_array_date):
	for i in range(X_array_date.shape[0]):
		for j in range(X_array_date.shape[1]):
			for row in range(X_array_date.shape[2]):
				for col in range(X_array_date.shape[3]):
					date = set_time_zone(X_array_date[i, j, row, col])
					#print("---------------------------{}-------------------------".format(date))
					#print("---------------------------{}-------------------------".format(date_time_covert_to_str(date)[6:]))
					X_array_date[i, j, row, col, 0] = date_time_covert_to_str(date)[6:]
	return X_array_date

def one_hot(array):
	bit_num = array.shape[0]
	one_hot_list = np.zeros((array.shape[1], 2**bit_num))
	for i in range(array.shape[1]):
		type_num = int(array[0][i] * 2**(bit_num - 1))
		one_hot_list[i][type_num] = 1
	return one_hot_list

def make_period_data(X_array, p = 24, lp = 6):
	print('X_array shape:{}'.format(X_array.shape))
	period = np.zeros((X_array.shape[0], 6, 15, 15, 1))
	for i in range(24 * lp): # 24 * 6, do this cuz the former 143 hours has no 6 days ago data,
		for j in range(lp): # so I make them fill in the fixed bigining 6 days data as the same
			period[i, j, :, :, :] = X_array[(i % p) + j * p, -1, :, :, :]
	for i in range(X_array.shape[0] - 24 * lp): # 1338 - 24 * 6
		for j in range(lp):
			period[i + 24 * lp, j, :, :, :] = X_array[i + j * p, -1, :, :, :]
	print('period shape:{}'.format(period.shape))
	return period

def make_trend_data(X_array, q = 168, lq = 6):
	trend = np.zeros((X_array.shape[0], 6, 15, 15, 1))
	for i in range(24 * 7 * lq): # 24 * 7 * 6
		for j in range(lq):
			trend[i, j, :, :, :] = X_array[(i % q) + j * q, -1, :, :, :]
	for i in range(X_array.shape[0] - 24 * 7 * lq): # 1338 - (24 * 7 * 6)
		for j in range(lq):
			trend[i + 24 * 7 * lq, j, :, :, :] = X_array[i + j * q, -1, :, :, :]
	print('trend shape:{}'.format(trend.shape))
	return trend

def read_data(path):
	_list = []
	with open(path, newline='') as csvfile:
		rows = csv.reader(csvfile)
		for row in rows:
			_list.append(float(row[2]))
	return _list

def train():
	# X_array, Y_array = get_X_and_Y_array(task_num=5)
	path = '/home/qiuhui/Desktop/howard/downloaded_data/Milano_WeatherPhenomena/mi_meteo_5911_full.csv'
	temporature = read_data(path)
	path = '/home/qiuhui/Desktop/howard/downloaded_data/Milano_WeatherPhenomena/mi_meteo_6129_full.csv'
	wind_speed = read_data(path)
	path = '/home/qiuhui/Desktop/howard/downloaded_data/pollution-mi/mi_pollution_5551.csv'
	pollution = read_data(path)
	TK = Prepare_Task_Data('./npy/final')
	TK2 = Prepare_Task_Data_2('/home/qiuhui/Desktop/bo/predict_telecom_traffic/CNN_RNN/npy/final/') ##fusion
	X_array, Y_array = TK.Task_max_min_avg(generate_data=False)
	_X_array, _Y_array = TK2.Task_max_min_avg(generate_data=False) ##fusion
	
	data_len = _X_array.shape[0]
	X_array_date = X_array[:9 * data_len // 10, :, :, :, 1, np.newaxis]
	X_array_date = time_feature(X_array_date)
	_X_array_date = _X_array[:9 * data_len // 10, :, :, :, 1, np.newaxis]
	_X_array_date = time_feature(_X_array_date)
	_X_array_date, _ = feature_scaling(_X_array_date, feature_range = (0.1, 255))

	temporature = np.array(temporature)
	wind_speed, pollution = np.array(wind_speed), np.array(pollution)

	# process the three external factor into one-hot by average
	# temporature
	temp_list = []
	temp_avg = sum(temporature) / len(temporature)
	temp_list, _ = feature_scaling(temporature, feature_range = (0, 1))

	# wind speed
	wind_speed_list = []
	wind_speed_avg = sum(wind_speed) / len(wind_speed)
	wind_speed_list, _ = feature_scaling(wind_speed, feature_range = (0, 1))

	# pollution ()
	pollution_list = []
	pollution_AQI_threshold = 100
	pollution_list, _ = feature_scaling(pollution, feature_range = (0, 1))
	
	# rain
	rain_avg = sum(map(sum, map(sum, map(sum, X_array[:, :, :, :, -1])))) / len(X_array)
	print('rain_avg = {} {}'.format(rain_avg, len(X_array)))
	rain_list = np.zeros((X_array.shape[0]))
	rain_map = np.zeros((6, 3, 3))
	X_array = X_array[: 9 * data_len // 10, :, 10:13, 10:13, -1]
	for i in range(X_array.shape[0]):
		for j in range(X_array.shape[1]):
			array = np.reshape(X_array[i, j, :, :], (9))
			k = 0
			l = 0
			for ele in array:
				if int(ele) > int(rain_avg):
					rain_map[j, k, l] = 1
					if l + 1 != 3:
						l += 1
					else:
						k += 1
						l = 0
		#print((rain_map == 1).sum())
		if (rain_map == 1).sum() >= 6 * 3 * 3 / 2:
			print((rain_map == 1).sum())
			rain_list[i] = 1
	
	#temp_list = np.array(temp_list)
	#wind_speed_list = np.array(wind_speed_list)

	#X_array = [rain_list, temp_list, wind_speed_list, pollution_list]
	X_array = rain_list
	X_array = np.array(X_array)
	X_array = X_array[np.newaxis, :]
	temp_list = temp_list[:, np.newaxis]
	wind_speed_list = wind_speed_list[:, np.newaxis]
	pollution_list = pollution_list[:, np.newaxis]
	print(X_array.shape)
	X_array = one_hot(X_array)
	Y_array = np.concatenate((X_array, temp_list, wind_speed_list, pollution_list), axis = 1)
	print('input X data shape:{} Y shape:{}'.format(_X_array.shape, Y_array.shape))
	_X_array = _X_array[: 9 * data_len // 10, :, :, :, -1, np.newaxis]
	_Y_array = _Y_array[: 9 * data_len // 10, :, :, :, 2:]
	# X_array = X_array[:, :, 10:15, 10:15, :]
	#X_array, scaler = feature_scaling(X_array, feature_range=(0.1, 255))
	_Y_array = _Y_array[:, :, 10:13, 10:13, :]
	_X_array, _scaler = feature_scaling(_X_array, feature_range = (0.1, 255))

	period = make_period_data(_X_array, p = 24, lp = 4) # p: 1 day ago, lp: take 6 days data
	period, _ = feature_scaling(period, feature_range = (0.1, 255))
	trend = make_trend_data(_X_array, q = 168, lq = 1) # q: 1 week ago, lq: take 6 weeks data
	trend, _ = feature_scaling(trend, feature_range = (0.1, 255))

	_Y_array, _ = feature_scaling(_Y_array, _scaler)
	#_X_array = np.concatenate((_X_array_date, _X_array), -1)
	print('_X_array shape:{}'.format(_X_array.shape))

	# X_array_2, Y_array_2 = get_X_and_Y_array(task_num=6)
	# Y_array_2 = Y_array_2[:, :, 10:13, 10:13, :]
	# parameter
	input_data_shape = [Y_array.shape[1]]
	output_data_shape = [Y_array.shape[1]]

	input_data_shape_2 = [_X_array.shape[1], _X_array.shape[2], _X_array.shape[3], _X_array.shape[4]]
	output_data_shape_2 = [_Y_array.shape[1], _Y_array.shape[2], _Y_array.shape[3], 1]

	input_data_shape_period = [period.shape[1], period.shape[2], period.shape[3], period.shape[4]]
	output_data_shape_period = [period.shape[1], period.shape[2], period.shape[3], 1]

	input_data_shape_trend = [trend.shape[1], trend.shape[2], trend.shape[3], trend.shape[4]]
	output_data_shape_trend = [trend.shape[1], trend.shape[2], trend.shape[3], 1]

	result_path = './result/temp/'
	# result_path = os.path.join(result_path,'report')
	model_path = {
		'reload_path': './output_model/CNN_RNN_test.ckpt',
		'save_path': './output_model/CNN_RNN_test.ckpt',
		'result_path': result_path
	}
	hyper_config = CNN_RNN_config.HyperParameterConfig()
	#print('-------------')
	#print(hyper_config.iter_epoch)
	#hyper_config2 = CNN_RNN_config.HyperParameterConfig()
	# hyper_config.read_config(file_path=os.path.join(root_dir, 'CNN_RNN/result/random_search_0609/_85/config.json'))
	hyper_config.CNN_RNN()
	#hyper_config.CNN_RNN_two()
	
	#neural = CNN_RNN(input_data_shape, output_data_shape, hyper_config)
	#neural2 = CNN_RNN_2_test(input_data_shape, output_data_shape, hyper_config)

	neural = CNN_RNN_2_test(input_data_shape_2, output_data_shape_2, input_data_shape_period,
		output_data_shape_period, input_data_shape_trend, output_data_shape_trend,
		input_data_shape, output_data_shape, hyper_config)

	neural.create_MTL_task(Y_array, Y_array, 'rain_traffic')
	neural.create_MTL_task_period(period, period, 'period')
	neural.create_MTL_task_trend(trend, trend, 'trend')
	neural.create_MTL_task_2(_X_array, _Y_array[:, :, :, :, 0, np.newaxis], 'CDR_min_traffic')
	neural.create_MTL_task_2(_X_array, _Y_array[:, :, :, :, 1, np.newaxis], 'CDR_avg_traffic')
	neural.create_MTL_task_2(_X_array, _Y_array[:, :, :, :, 2, np.newaxis], 'CDR_max_traffic')
	del X_array, Y_array, _X_array, _Y_array
	# neural.create_MTL_task(X_array_2, Y_array_2[:, :, :, :, 0, np.newaxis], '10_mins', 'cross_entropy')
	# del X_array_2, Y_array_2

	# neural.start_train(model_path, reload=False)
	neural.start_MTL_train(model_path, reload=False)

	if hyper_config.STPP == True:	print('STPP is on!')
	else:	print('STPP is off!')
	if hyper_config.trainable_w_fn == True:	print('trainable_w_fn fusion is on!')
	else:	print('trainable_w_fn fusion is off!')
	#neural2.start_MTL_train(model_path, reload = True)


if __name__ == '__main__':
	train()

