import numpy as np
from utility import feature_scaling, root_dir, set_time_zone, date_time_covert_to_str
import matplotlib.pyplot as plt
import sys
import CNN_RNN_config
from CNN_RNN import CNN_RNN_2_test
import os
sys.path.append(root_dir)
import data_utility as du
from multi_task_data import Prepare_Task_Data
from multi_task_data_CDR import Prepare_Task_Data_2
import time
import tensorflow as tf
import tensorlayer as tl

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

def train():
	# X_array, Y_array = get_X_and_Y_array(task_num=5)
	TK = Prepare_Task_Data('./npy/final')
	TK2 = Prepare_Task_Data_2('/home/qiuhui/Desktop/bo/predict_telecom_traffic/CNN_RNN/npy/final/') ##fusion
	X_array, Y_array = TK.Task_max_min_avg(generate_data=False)
	_X_array, _Y_array = TK2.Task_max_min_avg(generate_data=False) ##fusion

	data_len = X_array.shape[0]
	X_array_date = X_array[: 9 * data_len // 10, :, :, :, 1, np.newaxis]
	X_array_date = time_feature(X_array_date)
	_X_array_date = _X_array[: 9 * data_len // 10, :, :, :, 1, np.newaxis]
	_X_array_date = time_feature(_X_array_date)

	X_array_date, _ = feature_scaling(X_array_date, feature_range=(0.1, 255))
	_X_array_date, _ = feature_scaling(_X_array_date, feature_range = (0.1, 255))

	X_array = X_array[: 9 * data_len // 10, :, :, :, -1, np.newaxis]
	Y_array = Y_array[: 9 * data_len // 10, :, :, :, 2:]
	_X_array = _X_array[: 9 * data_len // 10, :, :, :, -1, np.newaxis]
	_Y_array = _Y_array[: 9 * data_len // 10, :, :, :, 2:]
	# X_array = X_array[:, :, 10:15, 10:15, :]
	Y_array = Y_array[:, :, 10:13, 10:13, :]
	X_array, scaler = feature_scaling(X_array, feature_range=(0.1, 255))
	_Y_array = _Y_array[:, :, 10:13, 10:13, :]
	_X_array, _scaler = feature_scaling(_X_array, feature_range = (0.1, 255))

	Y_array, _ = feature_scaling(Y_array, scaler)
	X_array = np.concatenate((X_array_date, X_array), -1)
	_Y_array, _ = feature_scaling(_Y_array, _scaler)
	_X_array = np.concatenate((_X_array_date, _X_array), -1)

	# X_array_2, Y_array_2 = get_X_and_Y_array(task_num=6)
	# Y_array_2 = Y_array_2[:, :, 10:13, 10:13, :]
	# parameter
	input_data_shape = [X_array.shape[1], X_array.shape[2], X_array.shape[3], X_array.shape[4]]
	output_data_shape = [Y_array.shape[1], Y_array.shape[2], Y_array.shape[3], 1]

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

	neural = CNN_RNN_2_test(input_data_shape, output_data_shape, hyper_config)

	#with tf.variable_scope('big_CNN_RNN'):
	#	neural = build_fusion_layer(neural.return_output(), neural2.return_output())

	neural.create_MTL_task(X_array, Y_array[:, :, :, :, 0, np.newaxis], 'SP_min_traffic')
	neural.create_MTL_task(X_array, Y_array[:, :, :, :, 1, np.newaxis], 'SP_avg_traffic')
	neural.create_MTL_task(X_array, Y_array[:, :, :, :, 2, np.newaxis], 'SP_max_traffic')
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
	if hyper_config.concat_fn == True:	print('Concat fusion is on!')
	else:	print('Concat fusion is off!')
	print('fusion model:{}'.format(str(hyper_config.combine_fn)))
	#neural2.start_MTL_train(model_path, reload = True)


if __name__ == '__main__':
	train()

