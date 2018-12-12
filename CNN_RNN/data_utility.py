import numpy as np
import os




def list_all_input_file(input_dir):
	onlyfile = [f for f in os.listdir(input_dir) if (os.path.isfile(
		os.path.join(input_dir, f)) and os.path.splitext(f)[1] == ".npy")]
	return onlyfile


def save_array(x_array, out_file):
	print('saving file to {}...'.format(out_file))
	np.save(out_file, x_array, allow_pickle=True)


def load_array(input_file):
	print('loading file from {}...'.format(input_file))
	X = np.load(input_file)
	return X


def get_one_hour_min(input_array):
	output_shape = [
		input_array.shape[0],
		1,
		input_array.shape[2],
		input_array.shape[3]]
	output_array = np.zeros(output_shape, dtype=np.float32)
	for i in range(input_array.shape[0]):
		for row in range(input_array.shape[2]):
			for col in range(input_array.shape[3]):
				min_value = np.amin(input_array[i, :, row, col])
				# print(input_array[i, :, row, col], ' min:', min_value)
				output_array[i, 0, row, col] = min_value

	print('output min array shape {}'.format(output_array.shape))
	return output_array


def get_one_hour_max(input_array):
	# print('input array shape {}'.format(input_array.shape))
	output_shape = [
		input_array.shape[0],
		1,
		input_array.shape[2],
		input_array.shape[3]]
	output_array = np.zeros(output_shape, dtype=np.float32)
	for i in range(input_array.shape[0]):
		for row in range(input_array.shape[2]):
			for col in range(input_array.shape[3]):
				max_value = np.amax(input_array[i, :, row, col])
				# print('record {} row {} col {} max {}'.format(i, row, col, max_value))
				output_array[i, 0, row, col] = max_value

	print('output max array shape {}'.format(output_array.shape))
	return output_array


def get_one_hour_average(input_array):
	print('input array shape {}'.format(input_array.shape))
	output_shape = [
		input_array.shape[0],
		1,
		input_array.shape[2],
		input_array.shape[3]]
	output_array = np.zeros(output_shape, dtype=np.float32)
	average_value_np = np.mean(input_array, axis=1)
	for i in range(input_array.shape[0]):
		for row in range(input_array.shape[2]):
			for col in range(input_array.shape[3]):
				average_value = np.mean(input_array[i, :, row, col])
				#print(average_value)
				output_array[i, 0, row, col] = average_value
	print('output avg array shape {}'.format(output_array.shape))
	return output_array


def load_data_hour_average(input_dir, filelist):
	x_target_path = './npy/hour_avg/one_hour/'
	y_target_path = './npy/hour_avg/one_hour_avg/'
	if not os.path.exists(x_target_path):
		os.makedirs(x_target_path)
	if not os.path.exists(y_target_path):
		os.makedirs(y_target_path)

	def load_array(input_file):
		print('loading file from {}...'.format(input_file))
		X = np.load(input_file)
		return X.astype(np.float64)

	def split_data(data_array):
		split_block_size = 6
		data_array_depth = data_array.shape[0]
		remainder = data_array_depth % split_block_size
		#print(remainder)
		split_block_num = int(data_array_depth / split_block_size)
		split_data_list = np.split(
			data_array[:data_array_depth - remainder], split_block_num)

		new_data_array = np.stack(split_data_list, axis=0)
		print('new_data_array shape {}'.format(new_data_array.shape))
		return new_data_array

	def one_hour_avg(input_array):
		quadrant_id = input_array[:, 0:1, :, :, 0]
		timestamp = input_array[:, 0:1, :, :, 1]
		intensity = get_one_hour_average(input_array[:, :, :, :, 2])
		coverage = get_one_hour_average(input_array[:, :, :, :, 3])

		new_array_list = [quadrant_id, timestamp, intensity, coverage] #shape(4, 264, 1, 100, 100)
		new_array = np.stack(new_array_list, axis=-1)  #put the first element to the last. ex:shape(6, 5, 4, 3, 2, 1) -> shape(5, 4, 3, 2, 1, 6)

		print('new avg array shape {}'.format(new_array.shape))
		return new_array

	data_array_list = []
	data_group_para = 10
	for i, file_name in enumerate(filelist):
		#print(int(load_array(input_dir + file_name)[1, 0, 0, 1]))
		data_array_list.append(load_array(input_dir + file_name))
		if i == 59:
			data_array_list.append(load_array(input_dir + "output_precipitation_2013-12-31.npy"))
		if i % data_group_para == 9:
			#file_dir, filename = os.path.split(input_dir + file_name)
			month = file_name.split('-')[-2]
			index = i // data_group_para
			data_array = np.concatenate(data_array_list, axis=0)
			data_array_list = []
			X = split_data(data_array)
			Y = one_hour_avg(X)
			#print(X)
			save_array(X, x_target_path + month + '_one_hour_' + str(index))
			save_array(Y, y_target_path + month + '_one_hour_avg_' + str(index))


def load_data_hour_min(input_dir, filelist):
	x_target_path = './npy/hour_min/one_hour/'
	y_target_path = './npy/hour_min/one_hour_min/'
	if not os.path.exists(x_target_path):
		os.makedirs(x_target_path)
	if not os.path.exists(y_target_path):
		os.makedirs(y_target_path)

	def load_array(input_file):
		print('loading file from {}...'.format(input_file))
		X = np.load(input_file)
		return X.astype(np.float64)

	def split_data(data_array):
		split_block_size = 6
		data_array_depth = data_array.shape[0]
		remainder = data_array_depth % split_block_size
		split_block_num = int(data_array_depth / split_block_size)
		split_data_list = np.split(
			data_array[:data_array_depth - remainder], split_block_num)

		new_data_array = np.stack(split_data_list, axis=0)
		print('new_data_array shape {}'.format(new_data_array.shape))
		return new_data_array

	def one_hour_min(input_array):
		quadrant_id = input_array[:, 0:1, :, :, 0]
		timestamp = input_array[:, 0:1, :, :, 1]
		intensity = get_one_hour_average(input_array[:, :, :, :, 2])
		coverage = get_one_hour_average(input_array[:, :, :, :, 3])

		new_array_list = [quadrant_id, timestamp, intensity, coverage] #shape(4, 264, 1, 100, 100)
		new_array = np.stack(new_array_list, axis=-1)  #put the first element to the last. ex:shape(6, 5, 4, 3, 2, 1) -> shape(5, 4, 3, 2, 1, 6)

		print('new min array shape {}'.format(new_array.shape))
		return new_array

	data_array_list = []
	data_group_para = 10
	for i, file_name in enumerate(filelist):
		data_array_list.append(load_array(input_dir + file_name))
		if i == 59:
			data_array_list.append(load_array(input_dir + "output_precipitation_2013-12-31.npy"))
		if i % data_group_para == 9:
			#file_dir, filename = os.path.split(input_dir)
			month = file_name.split('-')[-2]
			index = i // data_group_para
			data_array = np.concatenate(data_array_list, axis=0)
			data_array_list = []
			X = split_data(data_array)
			Y = one_hour_min(X)
			save_array(X, x_target_path + month + '_one_hour_' + str(index))
			save_array(Y, y_target_path + month + '_one_hour_min_' + str(index))


def load_data_hour_max(input_dir, filelist):
	x_target_path = './npy/hour_max/one_hour/'
	y_target_path = './npy/hour_max/one_hour_max/'
	if not os.path.exists(x_target_path):
		os.makedirs(x_target_path)
	if not os.path.exists(y_target_path):
		os.makedirs(y_target_path)

	def load_array(input_file):
		print('loading file from {}...'.format(input_file))
		X = np.load(input_file)
		return X.astype(np.float64)

	def split_data(data_array):
		split_block_size = 6
		data_array_depth = data_array.shape[0]
		remainder = data_array_depth % split_block_size
		split_block_num = int(data_array_depth / split_block_size)
		split_data_list = np.split(
			data_array[:data_array_depth - remainder], split_block_num)

		new_data_array = np.stack(split_data_list, axis=0)
		print('new_data_array shape {}'.format(new_data_array.shape))
		return new_data_array

	def one_hour_max(input_array):
		quadrant_id = input_array[:, 0:1, :, :, 0]
		timestamp = input_array[:, 0:1, :, :, 1]
		intensity = get_one_hour_average(input_array[:, :, :, :, 2])
		coverage = get_one_hour_average(input_array[:, :, :, :, 3])

		new_array_list = [quadrant_id, timestamp, intensity, coverage] #shape(4, 264, 1, 100, 100)
		new_array = np.stack(new_array_list, axis=-1)  #put the first element to the last. ex:shape(6, 5, 4, 3, 2, 1) -> shape(5, 4, 3, 2, 1, 6)

		print('new max array shape {}'.format(new_array.shape))
		return new_array

	data_array_list = []
	data_group_para = 10
	for i, file_name in enumerate(filelist):
		data_array_list.append(load_array(input_dir + file_name))
		if i == 59:
			data_array_list.append(load_array(input_dir + "output_precipitation_2013-12-31.npy"))
		if i % data_group_para == 9:
			#file_dir, filename = os.path.split(input_dir)
			month = file_name.split('-')[-2]
			index = i // data_group_para
			data_array = np.concatenate(data_array_list, axis=0)
			data_array_list = []
			X = split_data(data_array)
			Y = one_hour_max(X)
			save_array(X, x_target_path + month + '_one_hour_' + str(index))
			save_array(Y, y_target_path + month + '_one_hour_max_' + str(index))