import tensorflow as tf
import tensorlayer as tl
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
import math
import os


class Tf_Utility:
	def weight_variable(self, shape, name):
		# initial = tf.truncated_normal(shape, stddev=0.1)
		initial = np.random.randn(*shape) * sqrt(2.0 / np.prod(shape))
		return tf.Variable(initial, dtype=tf.float32, name=name)

	def bias_variable(self, shape, name):
		# initial = tf.random_normal(shape)
		initial = np.random.randn(*shape) * sqrt(2.0 / np.prod(shape))
		return tf.Variable(initial, dtype=tf.float32, name=name)

	def write_to_Tfrecord(self, X_array, Y_array, filename):
		writer = tf.python_io.TFRecordWriter(filename)
		for index, each_record in enumerate(X_array):
			tensor_record = each_record.astype(np.float64).tobytes()
			tensor_result = Y_array[index].astype(np.float64).tobytes()
			# print('in _write_to_Tfrecord',X_array.shape,Y_array.shape)
			example = tf.train.Example(features=tf.train.Features(feature={
				'index': tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
				'record': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tensor_record])),
				'result': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tensor_result]))
			}))

			writer.write(example.SerializeToString())
		print('index in write_to_Tfrecord:%s' % (index))
		writer.close()

	def read_data_from_Tfrecord(
		self,
		filename,
		input_temporal,
		Y_temporal):
		filename_queue = tf.train.string_input_producer([filename])
		reader = tf.TFRecordReader()
		_, serialized_example = reader.read(filename_queue)
		features = tf.parse_single_example(
			serialized_example,
			features={
				'index': tf.FixedLenFeature([], tf.int64),
				'record': tf.FixedLenFeature([], tf.string),
				'result': tf.FixedLenFeature([], tf.string)
			})
		index = features['index']
		record = tf.decode_raw(features['record'], tf.float64)
		result = tf.decode_raw(features['result'], tf.float64)
		record = tf.reshape(record, [
			input_temporal])
		result = tf.reshape(result, [
			Y_temporal])

		return index, record, result

	def read_data_from_Tfrecord_2(
		self,
		filename,
		input_temporal,
		input_vertical,
		input_horizontal,
		input_channel,
		Y_temporal,
		Y_vertical,
		Y_horizontal,
		Y_channel):
		#input_channel = 2
		filename_queue = tf.train.string_input_producer([filename])
		reader = tf.TFRecordReader()
		_, serialized_example = reader.read(filename_queue)
		print('serialized_example:%s' % (serialized_example))
		features = tf.parse_single_example(
			serialized_example,
			features={
				'index': tf.FixedLenFeature([], tf.int64),
				'record': tf.FixedLenFeature([], tf.string),
				'result': tf.FixedLenFeature([], tf.string)
			})
		index = features['index']
		record = tf.decode_raw(features['record'], tf.float64)
		result = tf.decode_raw(features['result'], tf.float64)
		print(input_temporal, input_vertical, input_horizontal, input_channel)
		record = tf.reshape(record, [
			input_temporal,
			input_vertical,
			input_horizontal,
			input_channel])
		result = tf.reshape(result, [
			Y_temporal,
			Y_vertical,
			Y_horizontal,
			Y_channel])

		return index, record, result

	def read_data_from_Tfrecord_period_trend(
		self,
		filename,
		input_temporal,
		input_vertical,
		input_horizontal,
		input_channel,
		Y_temporal,
		Y_vertical,
		Y_horizontal,
		Y_channel):
		#input_channel = 2
		filename_queue = tf.train.string_input_producer([filename])
		reader = tf.TFRecordReader()
		_, serialized_example = reader.read(filename_queue)
		print('serialized_example:%s' % (serialized_example))
		features = tf.parse_single_example(
			serialized_example,
			features={
				'index': tf.FixedLenFeature([], tf.int64),
				'record': tf.FixedLenFeature([], tf.string),
				'result': tf.FixedLenFeature([], tf.string)
			})
		index = features['index']
		record = tf.decode_raw(features['record'], tf.float64)
		result = tf.decode_raw(features['result'], tf.float64)
		print(input_temporal, input_vertical, input_horizontal, input_channel)
		record = tf.reshape(record, [
			input_temporal,
			input_vertical,
			input_horizontal,
			input_channel])
		result = tf.reshape(result, [
			Y_temporal,
			Y_vertical,
			Y_horizontal,
			Y_channel])

		return index, record, result

	def read_all_data_from_Tfreoced(
		self,
		filename,
		input_temporal,
		Y_temporal):
		record_iterator = tf.python_io.tf_record_iterator(path=filename)
		record_list = []
		result_list = []
		for string_record in record_iterator:
			example = tf.train.Example()
			example.ParseFromString(string_record)
			index = example.features.feature['index'].int64_list.value[0]
			record = example.features.feature['record'].bytes_list.value[0]
			result = example.features.feature['result'].bytes_list.value[0]
			record = np.fromstring(record, dtype=np.float64)
			record = record.reshape((
				input_temporal))

			result = np.fromstring(result, dtype=np.float64)
			result = result.reshape((
				Y_temporal))
			record_list.append(record)
			result_list.append(result)

		record = np.stack(record_list)
		result = np.stack(result_list)
		return index, record, result

	def read_all_data_from_Tfreoced_2(
		self,
		filename,
		input_temporal,
		input_vertical,
		input_horizontal,
		input_channel,
		Y_temporal,
		Y_vertical,
		Y_horizontal,
		Y_channel):
		record_iterator = tf.python_io.tf_record_iterator(path=filename)
		record_list = []
		result_list = []
		for string_record in record_iterator:
			example = tf.train.Example()
			example.ParseFromString(string_record)
			index = example.features.feature['index'].int64_list.value[0]
			record = example.features.feature['record'].bytes_list.value[0]
			result = example.features.feature['result'].bytes_list.value[0]
			record = np.fromstring(record, dtype=np.float64)
			record = record.reshape((
				input_temporal,
				input_vertical,
				input_horizontal,
				input_channel))

			result = np.fromstring(result, dtype=np.float64)
			result = result.reshape((
				Y_temporal,
				Y_vertical,
				Y_horizontal,
				Y_channel))
			record_list.append(record)
			result_list.append(result)

		record = np.stack(record_list)
		result = np.stack(result_list)
		return index, record, result

	def read_all_data_from_Tfreoced_period_trend(
		self,
		filename,
		input_temporal,
		input_vertical,
		input_horizontal,
		input_channel,
		Y_temporal,
		Y_vertical,
		Y_horizontal,
		Y_channel):
		record_iterator = tf.python_io.tf_record_iterator(path=filename)
		record_list = []
		result_list = []
		for string_record in record_iterator:
			example = tf.train.Example()
			example.ParseFromString(string_record)
			index = example.features.feature['index'].int64_list.value[0]
			record = example.features.feature['record'].bytes_list.value[0]
			result = example.features.feature['result'].bytes_list.value[0]
			record = np.fromstring(record, dtype=np.float64)
			record = record.reshape((
				input_temporal,
				input_vertical,
				input_horizontal,
				input_channel))

			result = np.fromstring(result, dtype=np.float64)
			result = result.reshape((
				Y_temporal,
				Y_vertical,
				Y_horizontal,
				Y_channel))
			record_list.append(record)
			result_list.append(result)

		record = np.stack(record_list)
		result = np.stack(result_list)
		return index, record, result

	def save_model(self, sess, saver, model_path):
		# model_path = './output_model/CNN_RNN.ckpt'
		print('saving model.....')
		try:
			save_path = saver.save(sess, model_path)
			# self.pre_train_saver.save(sess, model_path + '_part')
		except Exception:
			if not os.path.exists('./output_model/temp.ckpt'):
			    os.makedirs('./output_model/temp.ckpt')
			save_path = saver.save(sess, './output_model/temp.ckpt')
		finally:
			print('save_path:{}'.format(save_path))

	def reload_model(self, sess, saver, model_path):
		print('reloading model {}.....'.format(model_path))
		saver.restore(sess, model_path)

	#def print_all_tensor(self):
	#	graph = tf.get_default_graph()
	#	all_vars = [n.name for n in graph.as_graph_def().node]
	#	for var_s in all_vars:
	#		print(var_s)

	def print_all_trainable_var(self):
		vars_list = tf.trainable_variables()
		for var_s in vars_list:
			print(var_s)


class Multitask_Neural_Network(Tf_Utility):

	def build_MTL(self, input_data_shape, output_data_shape):
		#tf.reset_default_graph()
		tl.layers.clear_layers_name()
		self.shuffle_min_after_dequeue = 600
		self.shuffle_capacity = self.shuffle_min_after_dequeue + 3 * self.batch_size

		self.input_temporal = input_data_shape[0]
		self.input_vertical = input_data_shape[1]
		self.input_horizontal = input_data_shape[2]
		self.input_channel = input_data_shape[3]

		self.output_temporal = output_data_shape[0]
		self.output_vertical = output_data_shape[1]
		self.output_horizontal = output_data_shape[2]
		self.output_channel = 1
		self.predictor_output = self.output_temporal * self.output_vertical * self.output_horizontal * self.output_channel

		self.Xs = tf.placeholder(tf.float32, shape=[
			None, self.input_temporal, self.input_vertical, self.input_horizontal, self.input_channel], name='Input_x')
		self.Ys = tf.placeholder(tf.float32, shape=[
			None, self.output_temporal, self.output_vertical, self.output_horizontal, 1], name='Input_y')
		self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

		self.add_noise = tf.placeholder(tf.bool, name='add_noise')
		self.list1 = tf.placeholder(tf.float32, shape=(100, 512))
		self.RNN_init_state = tf.placeholder(tf.float32, [self.RNN_num_layers, 2, None, self.RNN_hidden_node_size])  # 2: hidden state and cell state
		self.multi_task_dic = {}

	def build_MTL_2(self, input_data_shape, output_data_shape):
		#tf.reset_default_graph()
		tl.layers.clear_layers_name()
		self.input_temporal_2 = input_data_shape[0]
		self.input_channel_2 = 1

		self.output_temporal_2 = output_data_shape[0]
		self.output_channel_2 = 1
		self.shuffle_min_after_dequeue = 600
		self.shuffle_capacity = self.shuffle_min_after_dequeue + 3 * self.batch_size

		self.Xs_2 = tf.placeholder(tf.float32, shape=[None, self.input_temporal_2], name='Input_x_2')
		self.Ys_2 = tf.placeholder(tf.float32, shape=[None, self.input_temporal_2], name='Input_y_2')
		self.keep_prob_2 = tf.placeholder(tf.float32, name='keep_prob_2')

		self.add_noise_2 = tf.placeholder(tf.bool, name='add_noise_2')
		self.RNN_init_state_2 = tf.placeholder(tf.float32, [self.RNN_num_layers, 2, None, self.RNN_hidden_node_size])  # 2: hidden state and cell state
		self.multi_task_dic = {}

	def build_MTL_period(self, input_data_shape, output_data_shape):
		#tf.reset_default_graph()
		tl.layers.clear_layers_name()
		self.shuffle_min_after_dequeue = 600
		self.shuffle_capacity = self.shuffle_min_after_dequeue + 3 * self.batch_size

		self.input_temporal_period = input_data_shape[0]
		self.input_vertical_period = input_data_shape[1]
		self.input_horizontal_period = input_data_shape[2]
		self.input_channel_period = input_data_shape[3]

		self.output_temporal_period = output_data_shape[0]
		self.output_vertical_period = output_data_shape[1]
		self.output_horizontal_period = output_data_shape[2]
		self.output_channel_period = output_data_shape[3]
		self.predictor_output_period = self.output_temporal * self.output_vertical * self.output_horizontal * self.output_channel

		self.Xs_period = tf.placeholder(tf.float32, shape=[
			None, self.input_temporal_period, self.input_vertical_period, self.input_horizontal_period, self.input_channel_period], name='Input_x')
		self.Ys_period = tf.placeholder(tf.float32, shape=[
			None, self.output_temporal_period, self.output_vertical_period, self.output_horizontal_period, self.output_channel_period], name='Input_y')
		self.keep_prob_period = tf.placeholder(tf.float32, name='keep_prob_period')

		self.add_noise_period = tf.placeholder(tf.bool, name='add_noise_period')
		self.RNN_init_state_period = tf.placeholder(tf.float32, [self.RNN_num_layers, 2, None, self.RNN_hidden_node_size])  # 2: hidden state and cell state
		self.multi_task_dic = {}

	def build_MTL_trend(self, input_data_shape, output_data_shape):
		#tf.reset_default_graph()
		tl.layers.clear_layers_name()
		self.shuffle_min_after_dequeue = 600
		self.shuffle_capacity = self.shuffle_min_after_dequeue + 3 * self.batch_size

		self.input_temporal_trend = input_data_shape[0]
		self.input_vertical_trend = input_data_shape[1]
		self.input_horizontal_trend = input_data_shape[2]
		self.input_channel_trend = input_data_shape[3]

		self.output_temporal_trend = output_data_shape[0]
		self.output_vertical_trend = output_data_shape[1]
		self.output_horizontal_trend = output_data_shape[2]
		self.output_channel_trend = output_data_shape[3]
		self.predictor_output_trend = self.output_temporal * self.output_vertical * self.output_horizontal * self.output_channel

		self.Xs_trend = tf.placeholder(tf.float32, shape=[
			None, self.input_temporal_trend, self.input_vertical_trend, self.input_horizontal_trend, self.input_channel_trend], name='Input_x')
		self.Ys_trend = tf.placeholder(tf.float32, shape=[
			None, self.output_temporal_trend, self.output_vertical_trend, self.output_horizontal_trend, self.output_channel_trend], name='Input_y')
		self.keep_prob_trend = tf.placeholder(tf.float32, name='keep_prob_trend')

		self.add_noise_trend = tf.placeholder(tf.bool, name='add_noise_trend')
		self.RNN_init_state_trend = tf.placeholder(tf.float32, [self.RNN_num_layers, 2, None, self.RNN_hidden_node_size])  # 2: hidden state and cell state
		self.multi_task_dic = {}

	def build_flatten_layer(self, tl_input):

		flat_tl = tl.layers.FlattenLayer(tl_input, name='flatten_layer')
		network = tl.layers.DenseLayer(flat_tl, W_init=self.fully_connected_W_init, n_units=self.fully_connected_units, act=lambda x: tl.act.lrelu(x, 0.2), name='fully_connect_1')
		network = tl.layers.DropoutLayer(network, keep=self.keep_rate, name='drop_1')
		self.tl_share_output = network
		return network

	def build_flatten_layer_period(self, tl_input):

		flat_tl = tl.layers.FlattenLayer(tl_input, name='flatten_layer_period')
		network = tl.layers.DenseLayer(flat_tl, W_init=self.fully_connected_W_init, n_units=self.fully_connected_units, act=lambda x: tl.act.lrelu(x, 0.2), name='fully_connect_1_period')
		network = tl.layers.DropoutLayer(network, keep=self.keep_rate, name='drop_1_period')
		return network

	def build_flatten_layer_trend(self, tl_input):

		flat_tl = tl.layers.FlattenLayer(tl_input, name='flatten_layer_trend')
		network = tl.layers.DenseLayer(flat_tl, W_init=self.fully_connected_W_init, n_units=self.fully_connected_units, act=lambda x: tl.act.lrelu(x, 0.2), name='fully_connect_1_trend')
		network = tl.layers.DropoutLayer(network, keep=self.keep_rate, name='drop_1_trend')
		return network

	def elementwise_fusion(self, N1, N2, combine_fn = tf.add):
		if combine_fn == 'avg':
			with tf.variable_scope('fusion'):
				net1 = tl.layers.ElementwiseLayer([N1, N2], combine_fn = tf.add, name = 'fusion_layer_add')
				net2 = tl.layers.InputLayer(self.list1, name='feed_input_2')
				self.tl_share_output = tl.layers.ElementwiseLayer([net1, net2], combine_fn = tf.divide, name = 'fusion_layer_div')
		else:
			with tf.variable_scope('fusion'):
				self.tl_share_output = tl.layers.ElementwiseLayer([N1, N2], combine_fn = combine_fn, name = 'fusion_layer')
			#self.tl_share_output = (N1 + N2) / 2

	def trainable_var_fusion(self, N1, N2, N3):
		with tf.variable_scope('fusion'):
			net_1 = tl.layers.Parametric_Matrix_Fusion_Layer(N1, name = 'fusion_layer_mul_1')
			net_2 = tl.layers.Parametric_Matrix_Fusion_Layer(N2, name = 'fusion_layer_mul_2')
			net_3 = tl.layers.Parametric_Matrix_Fusion_Layer(N3, name = 'fusion_layer_mul_3')
			self.tl_share_output = tl.layers.ElementwiseLayer([net_1, net_2, net_3], combine_fn = tf.add, name = 'fusion_layer_add')

	def create_MTL_task(self, input_x, input_y, task_name, loss_type='MSE'):
		#self.multi_task_dic[task_name] = self.__create_MTL_output(self.tl_share_output, self.Ys, task_name, loss_type='MSE')
		self.multi_task_dic[task_name] = {}
		self.__set_training_data(input_x, input_y, task_name)
		self.saver = tf.train.Saver()

	def create_MTL_task_2(self, input_x, input_y, task_name, loss_type='MSE'):
		self.multi_task_dic[task_name] = self.__create_MTL_output(self.tl_share_output, self.Ys, task_name, loss_type='MSE')
		self.__set_training_data_2(input_x, input_y, task_name)
		self.saver = tf.train.Saver()

	def create_MTL_task_period(self, input_x, input_y, task_name, loss_type='MSE'):
		#self.multi_task_dic[task_name] = self.__create_MTL_output(self.tl_share_output, self.Ys, task_name, loss_type='MSE')
		self.multi_task_dic[task_name] = {}
		self.__set_training_data_period(input_x, input_y, task_name)
		self.saver = tf.train.Saver()

	def create_MTL_task_trend(self, input_x, input_y, task_name, loss_type='MSE'):
		#self.multi_task_dic[task_name] = self.__create_MTL_output(self.tl_share_output, self.Ys, task_name, loss_type='MSE')
		self.multi_task_dic[task_name] = {}
		self.__set_training_data_trend(input_x, input_y, task_name)
		self.saver = tf.train.Saver()

	def __create_MTL_output(self, tl_input, y, scope_name, loss_type='MSE'):
		def get_l2_list():
			# print('get_l2_list:')
			var_list = []
			exclude_list = ['LSTMCell/B', 'regression_op/b', 'b_conv2d']
			for v in tf.trainable_variables():
				if any(x in v.name for x in exclude_list):
					continue
				if 'prediction_layer' in v.name and scope_name not in v.name:
					continue
				# print(v)
				var_list.append(v)
			return var_list

		def get_trainable_var():
			# print('get_trainable_var:')
			var_list = []
			for v in tf.trainable_variables():
				if 'prediction_layer' in v.name:
					if scope_name not in v.name:
						continue
				# print(v)
				var_list.append(v)
			return var_list

		def get_prediction_layer_var():
			var_list = []
			for v in tf.trainable_variables():
				if 'prediction_layer' in v.name:
					if scope_name in v.name:
						var_list.append(v)
						# print(v)
			return var_list

		with tf.variable_scope('prediction_layer'):
			with tf.variable_scope(scope_name):
				print('predictor_output : %d' % (self.predictor_output))
				tl_input = tl.layers.BatchNormLayer(tl_input, is_train=False, name='batch_norm')
				tl_regression = tl.layers.DenseLayer(tl_input, W_init=self.prediction_layer_1_W_init, n_units=self.prediction_layer_1_uints, act=lambda x: tl.act.lrelu(x, 0.2), name='regression_op_1')
				tl_regression = tl.layers.DropoutLayer(tl_regression, keep=self.prediction_layer_keep_rate, name='drop_1')
				tl_regression = tl.layers.DenseLayer(tl_input, W_init=self.prediction_layer_2_W_init, n_units=self.predictor_output, act=tl.activation.identity, name='regression_op_2')
				tl_regression = tl.layers.DropoutLayer(tl_regression, keep=self.prediction_layer_keep_rate, name='drop_2')
				tl_output = tl_regression
				regression_output = tl_output.outputs
				print(regression_output.get_shape())
				# print('regression_output shape {}'.format(regression_output.get_shape().as_list()))
				output = tf.reshape(regression_output, [-1, 1, self.output_vertical, self.output_horizontal, self.output_channel], name='output_layer')

				cross_entropy = tf.nn.softmax_cross_entropy_with_logits(output, y, name='corss_entropy_op')
				MSE = tf.reduce_mean(tf.pow(output - y, 2), name='MSE_op')
				RMSE = tf.sqrt(tf.reduce_mean(tf.pow(output - y, 2)))
				MAE = tf.reduce_mean(tf.abs(output - y))
				# MAPE = tf.reduce_mean(tf.abs(tf.divide(y - output, y)), name='MAPE_OP')
				MAPE = tf.reduce_mean(tf.divide(tf.abs(y - output), tf.reduce_mean(y)), name='MAPE_OP')
				L2_list = get_l2_list()
				L2_loss = self.__L2_norm(L2_list)

				if loss_type == 'cross_entropy':
					prediction_softmax = tf.nn.softmax(output)
					output = tf.argmax(prediction_softmax, 1)
					correct_prediciton = tf.equal(tf.argmax(prediction_softmax, 1), tf.argmax(y, 1))
					accuracy = tf.reduce_mean(tf.cast(correct_prediciton, tf.float32))
					cost = tf.add(cross_entropy, L2_loss * self.weight_decay, name='cost_op')
				else:
					cost = tf.add(MSE, L2_loss * self.weight_decay, name='cost_op')
					accuracy = 1 - MAPE
				optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
				opt_vars = get_trainable_var()
				gvs = optimizer.compute_gradients(cost, var_list=opt_vars)
				capped_gvs = [(tf.clip_by_norm(grad, 5), var) for grad, var in gvs if grad is not None]
				optimizer_op = optimizer.apply_gradients(capped_gvs)

				optimizer_predict = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
				only_prediction_opt_vars = get_prediction_layer_var()
				gvs_predict = optimizer_predict.compute_gradients(MAE, var_list=only_prediction_opt_vars)
				capped_gvs_predict = [(tf.clip_by_norm(grad, 5), var) for grad, var in gvs_predict if grad is not None]
				optimizer_op_predict = optimizer.apply_gradients(capped_gvs_predict)
				task_dic = {
					'output': output,
					'optomizer': optimizer_op,
					'prediction_optimizer': optimizer_op_predict,
					'tl_output': tl_output,
					'cost': cost,
					'L2_loss': L2_loss,
					'MSE': MSE,
					'MAE': MAE,
					'MAPE': MAPE,
					'RMSE': RMSE,
					'cross_entropy': cross_entropy,
					'accurcy': accuracy,
					'training_accurcy_history': [],
					'testing_accurcy_history': [],
					'training_MSE_history': [],
					'testing_MSE_history': [],
					'training_temp_loss': 0
				}
				return task_dic

	def parse_pooling(self, type_fn='max_pool'):
			if type_fn == 'max_pool':
				func = tf.nn.max_pool
			else:
				func = tf.nn.avg_pool
			return func

	def parse_activation(self, type_fn='relu'):
		if type_fn == 'relu':
			func = tf.nn.relu
		else:
			func = tf.nn.relu
		return func

	def parse_initializer_method(self, type_fn='xavier'):
		if type_fn:
			func = tf.contrib.layers.xavier_initializer_conv2d()
		else:
			func = tf.truncated_normal_initializer(stddev=0.1)

		return func

	def parse_RNN_cell(self, type_fn='LSTMcell'):
		if type_fn:
			cell = tf.nn.rnn_cell.LSTMCell
		else:
			cell = tf.nn.rnn_cell.BasicLSTMCell

		return cell

	def __summarized_report(self):
		task_keys = self.multi_task_dic.keys()
		task_keys = sorted(task_keys)
		summary_dic = {}
		for key in task_keys:
			if key == 'rain_traffic' or 'period' or 'trend':
				continue
			task_summ_dict = {}
			train_MSE = self.multi_task_dic[key]['training_MSE_history'][-1]
			train_accu = self.multi_task_dic[key]['training_accurcy_history'][-1]
			test_MSE = self.multi_task_dic[key]['testing_MSE_history'][-1]
			test_accu = self.multi_task_dic[key]['testing_accurcy_history'][-1]

			task_summ_dict['training_MSE'] = train_MSE
			task_summ_dict['training_accurcy'] = train_accu
			task_summ_dict['testing_MSE'] = test_MSE
			task_summ_dict['testing_accurcy'] = test_accu
			summary_dic[key] = task_summ_dict
		return summary_dic

	def print_all_layers(self):
		self.tl_share_output.print_layers()
		# print(self.tl_share_output.all_layers)

	def print_all_variables(self):
		self.tl_share_output.print_params()

	def __L2_norm(self, var_list):
		L2_loss = tf.add_n([tf.nn.l2_loss(v) for v in var_list])
		return L2_loss

	def __set_training_data(self, input_x, input_y, task_name):

		# print('input_x shape:{}'.format(input_x.shape))
		# print('input_y shape:{}'.format(input_y.shape))
		self.Y_temporal_2 = input_y.shape[1]
		self.Y_channel_2 = 1
		# input_x, self.mean, self.std = self.feature_normalize_input_data(input_x)
		self.mean = 0
		self.std = 1
		X_data = input_x
		Y_data = input_y

		# Y_data = Y_data[:,np.newaxis]
		# print(X_data[1,0,0,0,-1],Y_data[0,0,0,0,-1])

		training_X = X_data[0:int(9 * X_data.shape[0] / 10)]
		training_Y = Y_data[0:int(9 * Y_data.shape[0] / 10)]
		# training_X = X_data  # todo
		# training_Y = Y_data  # todo
		testing_X = X_data[int(9 * X_data.shape[0] / 10):]
		testing_Y = Y_data[int(9 * Y_data.shape[0] / 10):]

		training_file = task_name + '_training.tfrecoeds'
		testing_file = task_name + '_testing.tfrecoeds'

		print('training X shape:{}, training Y shape:{}'.format(
			training_X.shape, training_Y.shape))
		self.write_to_Tfrecord(training_X, training_Y, training_file)
		self.write_to_Tfrecord(testing_X, testing_Y, testing_file)
		self.training_data_number = training_X.shape[0]
		self.multi_task_dic[task_name]['training_file'] = training_file
		self.multi_task_dic[task_name]['testing_file'] = testing_file

		training_data = self.read_data_from_Tfrecord(
			self.multi_task_dic[task_name]['training_file'],
			self.input_temporal_2,
			self.Y_temporal_2)
		batch_tuple_OP = tf.train.shuffle_batch(
			training_data,
			batch_size=self.batch_size,
			capacity=self.shuffle_capacity,
			min_after_dequeue=self.shuffle_min_after_dequeue)
		batch_without_shuffle_OP = tf.train.batch(
			training_data,
			batch_size=self.batch_size)

		self.multi_task_dic[task_name]['shuffle_batch_OP'] = batch_tuple_OP
		self.multi_task_dic[task_name]['batch_OP'] = batch_without_shuffle_OP

	def __set_training_data_2(self, input_x, input_y, task_name):

		# print('input_x shape:{}'.format(input_x.shape))
		# print('input_y shape:{}'.format(input_y.shape))
		self.Y_temporal = input_y.shape[1]
		self.Y_vertical = input_y.shape[2]
		self.Y_horizontal = input_y.shape[3]
		self.Y_channel = input_y.shape[4]
		# input_x, self.mean, self.std = self.feature_normalize_input_data(input_x)
		self.mean = 0
		self.std = 1
		X_data = input_x
		Y_data = input_y

		# Y_data = Y_data[:,np.newaxis]
		# print(X_data[1,0,0,0,-1],Y_data[0,0,0,0,-1])

		training_X = X_data[0:int(9 * X_data.shape[0] / 10)]
		training_Y = Y_data[0:int(9 * Y_data.shape[0] / 10)]
		# training_X = X_data  # todo
		# training_Y = Y_data  # todo
		testing_X = X_data[int(9 * X_data.shape[0] / 10):]
		testing_Y = Y_data[int(9 * Y_data.shape[0] / 10):]

		training_file = task_name + '_training.tfrecoeds'
		testing_file = task_name + '_testing.tfrecoeds'

		print('training X shape:{}, training Y shape:{}'.format(
			training_X.shape, training_Y.shape))
		self.write_to_Tfrecord(training_X, training_Y, training_file)
		self.write_to_Tfrecord(testing_X, testing_Y, testing_file)
		self.training_data_number = training_X.shape[0]
		self.multi_task_dic[task_name]['training_file'] = training_file
		self.multi_task_dic[task_name]['testing_file'] = testing_file

		training_data = self.read_data_from_Tfrecord_2(
			self.multi_task_dic[task_name]['training_file'],
			self.input_temporal,
			self.input_vertical,
			self.input_horizontal,
			self.input_channel,
			self.Y_temporal,
			self.Y_vertical,
			self.Y_horizontal,
			self.Y_channel)
		batch_tuple_OP = tf.train.shuffle_batch(
			training_data,
			batch_size=self.batch_size,
			capacity=self.shuffle_capacity,
			min_after_dequeue=self.shuffle_min_after_dequeue)
		batch_without_shuffle_OP = tf.train.batch(
			training_data,
			batch_size=self.batch_size)

		self.multi_task_dic[task_name]['shuffle_batch_OP'] = batch_tuple_OP
		self.multi_task_dic[task_name]['batch_OP'] = batch_without_shuffle_OP

	def __set_training_data_period(self, input_x, input_y, task_name):

		# print('input_x shape:{}'.format(input_x.shape))
		# print('input_y shape:{}'.format(input_y.shape))
		self.Y_temporal_period = input_y.shape[1]
		self.Y_vertical_period = input_y.shape[2]
		self.Y_horizontal_period = input_y.shape[3]
		self.Y_channel_period = input_y.shape[4]
		# input_x, self.mean, self.std = self.feature_normalize_input_data(input_x)
		self.mean = 0
		self.std = 1
		X_data = input_x
		Y_data = input_y

		# Y_data = Y_data[:,np.newaxis]
		# print(X_data[1,0,0,0,-1],Y_data[0,0,0,0,-1])

		training_X = X_data[0:int(9 * X_data.shape[0] / 10)]
		training_Y = Y_data[0:int(9 * Y_data.shape[0] / 10)]
		# training_X = X_data  # todo
		# training_Y = Y_data  # todo
		testing_X = X_data[int(9 * X_data.shape[0] / 10):]
		testing_Y = Y_data[int(9 * Y_data.shape[0] / 10):]

		training_file = task_name + '_training.tfrecoeds'
		testing_file = task_name + '_testing.tfrecoeds'

		print('training X shape:{}, training Y shape:{}'.format(
			training_X.shape, training_Y.shape))
		self.write_to_Tfrecord(training_X, training_Y, training_file)
		self.write_to_Tfrecord(testing_X, testing_Y, testing_file)
		self.training_data_number = training_X.shape[0]
		self.multi_task_dic[task_name]['training_file'] = training_file
		self.multi_task_dic[task_name]['testing_file'] = testing_file

		training_data = self.read_data_from_Tfrecord_period_trend(
			self.multi_task_dic[task_name]['training_file'],
			self.input_temporal_period,
			self.input_vertical_period,
			self.input_horizontal_period,
			self.input_channel_period,
			self.Y_temporal_period,
			self.Y_vertical_period,
			self.Y_horizontal_period,
			self.Y_channel_period)
		batch_tuple_OP = tf.train.shuffle_batch(
			training_data,
			batch_size=self.batch_size,
			capacity=self.shuffle_capacity,
			min_after_dequeue=self.shuffle_min_after_dequeue)
		batch_without_shuffle_OP = tf.train.batch(
			training_data,
			batch_size=self.batch_size)

		self.multi_task_dic[task_name]['shuffle_batch_OP'] = batch_tuple_OP
		self.multi_task_dic[task_name]['batch_OP'] = batch_without_shuffle_OP

	def __set_training_data_trend(self, input_x, input_y, task_name):

		# print('input_x shape:{}'.format(input_x.shape))
		# print('input_y shape:{}'.format(input_y.shape))
		self.Y_temporal_trend = input_y.shape[1]
		self.Y_vertical_trend = input_y.shape[2]
		self.Y_horizontal_trend = input_y.shape[3]
		self.Y_channel_trend = input_y.shape[4]
		# input_x, self.mean, self.std = self.feature_normalize_input_data(input_x)
		self.mean = 0
		self.std = 1
		X_data = input_x
		Y_data = input_y

		# Y_data = Y_data[:,np.newaxis]
		# print(X_data[1,0,0,0,-1],Y_data[0,0,0,0,-1])

		training_X = X_data[0:int(9 * X_data.shape[0] / 10)]
		training_Y = Y_data[0:int(9 * Y_data.shape[0] / 10)]
		# training_X = X_data  # todo
		# training_Y = Y_data  # todo
		testing_X = X_data[int(9 * X_data.shape[0] / 10):]
		testing_Y = Y_data[int(9 * Y_data.shape[0] / 10):]

		training_file = task_name + '_training.tfrecoeds'
		testing_file = task_name + '_testing.tfrecoeds'

		print('training X shape:{}, training Y shape:{}'.format(
			training_X.shape, training_Y.shape))
		self.write_to_Tfrecord(training_X, training_Y, training_file)
		self.write_to_Tfrecord(testing_X, testing_Y, testing_file)
		self.training_data_number = training_X.shape[0]
		self.multi_task_dic[task_name]['training_file'] = training_file
		self.multi_task_dic[task_name]['testing_file'] = testing_file

		training_data = self.read_data_from_Tfrecord_period_trend(
			self.multi_task_dic[task_name]['training_file'],
			self.input_temporal_trend,
			self.input_vertical_trend,
			self.input_horizontal_trend,
			self.input_channel_trend,
			self.Y_temporal_trend,
			self.Y_vertical_trend,
			self.Y_horizontal_trend,
			self.Y_channel_trend)
		batch_tuple_OP = tf.train.shuffle_batch(
			training_data,
			batch_size=self.batch_size,
			capacity=self.shuffle_capacity,
			min_after_dequeue=self.shuffle_min_after_dequeue)
		batch_without_shuffle_OP = tf.train.batch(
			training_data,
			batch_size=self.batch_size)

		self.multi_task_dic[task_name]['shuffle_batch_OP'] = batch_tuple_OP
		self.multi_task_dic[task_name]['batch_OP'] = batch_without_shuffle_OP

	def _MTL_testing_data(self, sess, test_x_1, test_y_1, task_name):
		task_dic = self.multi_task_dic[task_name]
		predict_list = []
		cum_MSE = 0
		cum_accu = 0
		batch_num = test_x_1.shape[0] // self.batch_size
		for batch_index in range(batch_num):
			dp_dict = tl.utils.dict_to_one(task_dic['tl_output'].all_drop)
			batch_x_1 = test_x_1[batch_index * self.batch_size: (batch_index + 1) * self.batch_size]
			batch_y_1 = test_y_1[batch_index * self.batch_size: (batch_index + 1) * self.batch_size]
			list1 = np.full((100, 512), 2)
			feed_dict = {
				self.Xs: batch_x_1,
				self.Ys: batch_y_1,
				self.keep_prob: 0.85,
				self.add_noise: 1
				}
			feed_dict.update(dp_dict)
			with tf.device('/gpu:0'):
				MSE, accu, predict = sess.run([task_dic['MSE'], task_dic['accurcy'], task_dic['output']], feed_dict=feed_dict)
			'''
			for i in range(10, 15):
				for j in range(predict.shape[1]):
					print('batch index: {} predict:{:.4f} real:{:.4f}'.format(batch_index, predict[i, j, 0, 0, 0], batch_y[i, j, 0, 0, 0]))
			print()
			'''
			for predict_element in predict:
				predict_list.append(predict_element)
			cum_MSE += MSE
			cum_accu += accu
		return cum_MSE / batch_num, cum_accu / batch_num, np.stack(predict_list)

	def save_result_report(self, dir_name='./result/temp'):
		if not os.path.exists(dir_name):
			os.makedirs(dir_name)
		with open(os.path.join(dir_name, 'report.txt'), 'w') as f:
			task_keys = self.multi_task_dic.keys()
			task_keys = sorted(task_keys)

			record_len = len(self.multi_task_dic[task_keys[0]]['training_accurcy_history'])
			for num in range(record_len):
				for key in task_keys:
					if key == 'rain_traffic' or 'period' or 'trend':
						continue
					line = key + ': \n'
					line += ' train_MSE: ' + str(self.multi_task_dic[key]['training_MSE_history'][num])
					line += ' train_accu: ' + str(self.multi_task_dic[key]['training_accurcy_history'][num])
					line += ' test_MES: ' + str(self.multi_task_dic[key]['testing_MSE_history'][num])
					line += ' test_accu: ' + str(self.multi_task_dic[key]['testing_accurcy_history'][num])
					f.write(line + '\n')
				f.write('\n')

	def __save_hyperparameter(self, config, dir_name='./result/temp'):
			if not os.path.exists(dir_name):
				os.makedirs(dir_name)

			config.save_json(os.path.join(dir_name, 'config.json'))
			'''
			sort_key_val = [(k, config[k]) for k in sorted(config.keys())]

			with open(dir_name + 'config.txt', 'w') as f:
				for element in sort_key_val:
					line = str(element[0]) + ': ' + str(element[1])
					f.write(line + '\n')
			'''

	def _plot_predict_vs_real(self, fig_instance, task_name, testing_y, testing_predict_y, training_y, training_predict_y):

		ax_1 = fig_instance.add_subplot(3, 1, 1)
		ax_2 = fig_instance.add_subplot(3, 1, 2)
		ax_3 = fig_instance.add_subplot(3, 1, 3)
		ax_1.cla()
		ax_2.cla()
		ax_3.cla()

		ax_1.plot(testing_y, label='real', marker='.')
		ax_1.plot(testing_predict_y, label='predict', marker='.')
		ax_1.set_title(task_name + ' testing data')
		ax_1.grid()
		ax_1.legend()

		ax_2.plot(training_y, label='real', marker='.')
		ax_2.plot(training_predict_y, label='predict', marker='.')
		ax_2.set_title(task_name + ' training data')
		ax_2.grid()
		ax_2.legend()

		ax_3.plot(self.multi_task_dic[task_name]['training_MSE_history'], 'g-', label=task_name + ' training losses')
		ax_3.plot(self.multi_task_dic[task_name]['testing_MSE_history'], 'b-', label=task_name + ' testing losses')
		ax_3.set_title(task_name + 'loss')
		ax_3.grid()
		ax_3.legend()
		# ax.draw()
		plt.pause(0.001)

	def start_STL_predict(self, testing_x, testing_y, model_path, task_name):

		print('testing_x shape {}'.format(testing_x.shape))
		print('testing_y shape {}'.format(testing_y.shape))
		# self.print_all_layers()
		with tf.Session() as sess:
			self.reload_model(sess, self.saver, model_path['reload_path'])
			testing_loss, testing_accu, prediction = self._MTL_testing_data(sess, testing_x, testing_y, task_name)

			print('preddict finished!')
			print('task {}: accu:{} MSE:{}'.format(task_name, testing_accu, testing_loss))

		return prediction

	def start_MTL_predict(self, testing_x, testing_y, _testing_x, _testing_y, model_path):
			def get_multi_task_batch(batch_x, batch_y):
				batch_y = np.transpose(batch_y, [4, 0, 1, 2, 3])
				batch_y = np.expand_dims(batch_y, axis=5)
				return batch_x, batch_y

			print('input_x shape {}'.format(testing_x.shape))
			print('input_y shape {}'.format(testing_y.shape))
			testing_x, testing_y = get_multi_task_batch(testing_x, testing_y)
			_testing_x, _testing_y = get_multi_task_batch(_testing_x, _testing_y)
			# tf.reset_default_graph()
			# tf.train.import_meta_graph(model_path['reload_path'] + '.meta')

			self.print_all_layers()
			with tf.Session() as sess:
				self.reload_model(sess, self.saver, model_path['reload_path'])
				testing_loss_min, testing_accu_min, prediction_min = self._MTL_testing_data(sess, testing_x, testing_y[0]
					, _testing_x, _testing_y[0], 'CDR_min_traffic')
				testing_loss_avg, testing_accu_avg, prediction_avg = self._MTL_testing_data(sess, testing_x, testing_y[1]
					, _testing_x, _testing_y[1], 'CDR_avg_traffic')
				testing_loss_max, testing_accu_max, prediction_max = self._MTL_testing_data(sess, testing_x, testing_y[2]
					, _testing_x, _testing_y[2], 'CDR_max_traffic')
			print('preddict finished!')
			print('task Min: accu:{} MSE:{}'.format(testing_accu_min, testing_loss_min))
			print('task avg: accu:{} MSE:{}'.format(testing_accu_avg, testing_loss_avg))
			print('task Max: accu:{} MSE:{}'.format(testing_accu_max, testing_loss_max))

			return [prediction_min, prediction_avg, prediction_max]

	def run_multi_task(self, sess, task_name='', optimizer='optomizer'):
		def run_task_optimizer(sess, batch_x, batch_y, task_name, optimizer='optomizer'):
			task_dic = self.multi_task_dic[task_name]
			feed_dict = {
				self.Xs: batch_x,
				self.Ys: batch_y,
				self.keep_prob: 0.85,
				self.add_noise: 1}
			feed_dict.update(task_dic['tl_output'].all_drop)
			_, cost, L2_loss = sess.run([task_dic[optimizer], task_dic['cost'], task_dic['L2_loss']], feed_dict=feed_dict)

			return cost, L2_loss
		task = self.multi_task_dic[task_name]
		training_batch_op = task['shuffle_batch_OP']
		index, batch_x, batch_y = sess.run(training_batch_op)
		# batch_x, batch_y = get_multi_task_batch(batch_x, batch_y)
		cost, L2 = run_task_optimizer(sess, batch_x, batch_y, task_name)
		task['training_temp_loss'] += cost

	def run_multi_task_2(self, sess, task_name_1='', task_name_2='', task_name_3='', optimizer='optomizer'):
		def run_task_optimizer(sess, batch_x_1, batch_y_1, batch_x_2, batch_y_2
			, batch_x_3, batch_y_3, task_name, optimizer='optomizer'):
			task_dic = self.multi_task_dic[task_name]
			list1 = np.full((100, 512), 2)
			feed_dict = {
				self.Xs: batch_x_1,
				self.Ys: batch_y_1,
				self.keep_prob: 0.85,
				self.add_noise: 1,
				self.Xs_period: batch_x_2,
				self.Ys_period: batch_y_2,
				self.keep_prob_period: 0.85,
				self.add_noise_period: 1,
				self.Xs_trend: batch_x_3,
				self.Ys_trend: batch_y_3,
				self.keep_prob_trend: 0.85,
				self.add_noise_trend: 1,
				self.list1:list1}
			feed_dict.update(task_dic['tl_output'].all_drop)
			_, cost, L2_loss = sess.run([task_dic[optimizer], task_dic['cost'], task_dic['L2_loss']], feed_dict=feed_dict)

			return cost, L2_loss

		task = self.multi_task_dic[task_name_2]
		training_batch_op = task['shuffle_batch_OP']
		index_2, batch_x_2, batch_y_2 = sess.run(training_batch_op)

		task = self.multi_task_dic[task_name_3]
		training_batch_op = task['shuffle_batch_OP']
		index_3, batch_x_3, batch_y_3 = sess.run(training_batch_op)

		task = self.multi_task_dic[task_name_1]
		training_batch_op = task['shuffle_batch_OP']
		index_1, batch_x_1, batch_y_1 = sess.run(training_batch_op)

		# batch_x, batch_y = get_multi_task_batch(batch_x, batch_y)
		cost, L2 = run_task_optimizer(sess, batch_x_1, batch_y_1, batch_x_2, batch_y_2
			, batch_x_3, batch_y_3, task_name_1)
		task['training_temp_loss'] += cost

	def run_task_evaluate(self, sess, fig, epoch, display_step=50, task_name_1=''):
			task = self.multi_task_dic[task_name_1]
			index_1, testing_X_1, testing_Y_1 = self.read_all_data_from_Tfreoced_2(
				task['testing_file'],
				self.input_temporal,
				self.input_vertical,
				self.input_horizontal,
				self.input_channel,
				self.Y_temporal,
				self.Y_vertical,
				self.Y_horizontal,
				self.Y_channel)
			index_1, batch_x_sample_1, batch_y_sample_1 = sess.run(task['batch_OP'])

			# batch_x_sample, batch_y_sample = get_multi_task_batch(batch_x_sample, batch_y_sample)
			# testing_X, testing_Y = get_multi_task_batch(testing_X, testing_Y)
			testing_loss, testing_accu, prediction = self._MTL_testing_data(sess, testing_X_1, testing_Y_1, task_name_1)
			training_loss_nodrop, training_accu, train_prediction = self._MTL_testing_data(sess, batch_x_sample_1, batch_y_sample_1, task_name_1)

			task['training_temp_loss'] /= display_step

			self.multi_task_dic[task_name_1]['testing_MSE_history'].append(testing_loss)
			self.multi_task_dic[task_name_1]['training_MSE_history'].append(training_loss_nodrop)
			self.multi_task_dic[task_name_1]['testing_accurcy_history'].append(testing_accu)
			self.multi_task_dic[task_name_1]['training_accurcy_history'].append(training_accu)
			print('task:{} epoch:{} training_cost:{:.4f} trainin_MSE(nodrop):{:.4f} training_accu:{:.4f} testing_MSE(nodrop):{:.4f} testing_accu:{:.4f}'.format(
				task_name_1,
				epoch,
				task['training_temp_loss'],
				training_loss_nodrop,
				training_accu,
				testing_loss,
				testing_accu))
			self._plot_predict_vs_real(
				fig,
				task_name_1,
				testing_Y_1[:100, 0, 0, 0, 0],
				prediction[:100, 0, 0, 0, 0],
				batch_y_sample_1[:100, 0, 0, 0, 0],
				train_prediction[:100, 0, 0, 0, 0])
			task['training_temp_loss'] = 0
	def early_stop_v2(self, epoch, stop_type=1):
		if epoch > 100:
			def max_check(_max, CDR, count):
				if _max <= CDR:
					_max = CDR
					count = 0
				else:
					count += 1
				return _max, count
			#task_keys = self.multi_task_dic.keys()
			#task_keys = sorted(task_keys)
			Flag = False
			CDR_max = self.multi_task_dic['CDR_max_traffic']['testing_accurcy_history'][-1]
			CDR_avg = self.multi_task_dic['CDR_avg_traffic']['testing_accurcy_history'][-1]
			CDR_min = self.multi_task_dic['CDR_min_traffic']['testing_accurcy_history'][-1]
			self.CDR_max_max, self.CDR_max_counter = max_check(self.CDR_max_max, CDR_max, self.CDR_max_counter)
			self.CDR_avg_max, self.CDR_avg_counter = max_check(self.CDR_avg_max, CDR_avg, self.CDR_avg_counter)
			self.CDR_min_max, self.CDR_min_counter = max_check(self.CDR_min_max, CDR_min, self.CDR_min_counter)
			if self.CDR_max_counter >= 4:
				print('CDR_max early stop!')
				Flag = True
			if self.CDR_avg_counter >= 4:
				print('CDR_avg early stop!')
				Flag = True
			if self.CDR_min_counter >= 4:
				print('CDR_min early stop!')
				Flag = True
			return Flag

	def start_MTL_train(self, model_path, reload=False):
		display_step = 50
		self.CDR_max_max, self.CDR_avg_max, self.CDR_min_max = -100, -100, -100
		self.CDR_max_counter = 0
		self.CDR_avg_counter = 0
		self.CDR_min_counter = 0
		epoch_his = []
		plt.ion()
		# loss_fig = plt.figure(0)
		min_fig = plt.figure(1)
		avg_fig = plt.figure(2)
		max_fig = plt.figure(3)
		#_10_mins_fig = plt.figure(4)

		# model_base_name = os.path.basename(model_path['save_path'])
		# model_base_name = os.path.splitext(model_base_name)[0]
		# dir_name = './result/' + model_base_name + '/'
		result_path = model_path['result_path']

		def save_figure(dir_name='./result/temp'):
			#min_fig.set_size_inches(12, 9)
			#avg_fig.set_size_inches(12, 9)
			#max_fig.set_size_inches(12, 9)
			min_fig.savefig(os.path.join(dir_name, 'min.png'), dpi=100)
			avg_fig.savefig(os.path.join(dir_name, 'avg.png'), dpi=100)
			max_fig.savefig(os.path.join(dir_name, 'max.png'), dpi=100)

		def get_multi_task_batch(batch_x, batch_y):
			batch_y = np.transpose(batch_y, [4, 0, 1, 2, 3])
			batch_y = np.expand_dims(batch_y, axis=5)
			return batch_x, batch_y

		def early_stop(epoch, stop_type=1):
			task_keys = self.multi_task_dic.keys()
			task_keys = sorted(task_keys)
			Flag = False

			if epoch >= 500:
				if self.multi_task_dic['CDR_max_traffic']['testing_accurcy_history'][-1] > 0.74:		#0.74
					if self.multi_task_dic['CDR_avg_traffic']['testing_accurcy_history'][-1] > 0.74:	#0.74
						Flag = True
				if self.multi_task_dic['CDR_max_traffic']['testing_accurcy_history'][-1] > 0.76:		#0.76
					if self.multi_task_dic['CDR_avg_traffic']['testing_accurcy_history'][-1] > 0.7:	#0.7
						Flag = True
				if self.multi_task_dic['CDR_avg_traffic']['testing_accurcy_history'][-1] > 0.8:			#0.8
					if self.multi_task_dic['CDR_max_traffic']['testing_accurcy_history'][-1] > 0.7:	#0.7
						Flag = True
			if epoch >= 800:
				if self.multi_task_dic['CDR_max_traffic']['testing_accurcy_history'][-1] > 0.7:		#0.7
					if self.multi_task_dic['CDR_avg_traffic']['testing_accurcy_history'][-1] > 0.7:	#0.7
						Flag = True
			return Flag

		def _plot_loss_rate(epoch_his):
			ax_1 = loss_fig.add_subplot(3, 1, 1)
			ax_2 = loss_fig.add_subplot(3, 1, 2)
			ax_3 = loss_fig.add_subplot(3, 1, 3)
			ax_1.cla()
			ax_2.cla()
			ax_3.cla()
			ax_1.plot(self.multi_task_dic['CDR_min_traffic']['training_cost_history'], 'g-', label='min training losses')
			ax_1.plot(self.multi_task_dic['CDR_min_traffic']['testing_cost_history'], 'b-', label='min testing losses')
			ax_1.legend()
			ax_2.plot(self.multi_task_dic['CDR_avg_traffic']['training_cost_history'], 'g-', label='avg training losses')
			ax_2.plot(self.multi_task_dic['CDR_avg_traffic']['testing_cost_history'], 'b-', label='avg testing losses')
			ax_2.legend()
			ax_3.plot(self.multi_task_dic['CDR_max_traffic']['training_cost_history'], 'g-', label='max training losses')
			ax_3.plot(self.multi_task_dic['CDR_max_traffic']['testing_cost_history'], 'b-', label='max testing losses')
			ax_3.legend()
			plt.pause(0.001)

		self.__save_hyperparameter(self.hyper_config, result_path)

		with tf.Session() as sess:
			coord = tf.train.Coordinator()
			treads = tf.train.start_queue_runners(sess=sess, coord=coord)
			tf.summary.FileWriter('logs/', sess.graph)

			if reload:
				self.reload_model(sess, self.saver, model_path['reload_path'])
			else:
				sess.run(tf.global_variables_initializer())
			with tf.device('/gpu:0'):
				try:
					for epoch in range(self.iter_epoch):
						self.run_multi_task(sess, 'CDR_max_traffic')
						self.run_multi_task(sess, 'CDR_avg_traffic')
						self.run_multi_task(sess, 'CDR_min_traffic')
						# run_multi_task(sess, '10_mins')

						if epoch % display_step == 0 and epoch is not 0:
							self.run_task_evaluate(sess, max_fig, epoch, task_name_1='CDR_max_traffic')
							self.run_task_evaluate(sess, avg_fig, epoch, task_name_1='CDR_avg_traffic')
							self.run_task_evaluate(sess, min_fig, epoch, task_name_1='CDR_min_traffic')
							# run_task_evaluate(sess, _10_mins_fig, epoch, task_name='10_mins')
							print()
							epoch_his.append(epoch)
							# _plot_loss_rate(epoch_his)
							flag = self.early_stop_v2(epoch, 0)

						if epoch % 500 == 0 and epoch is not 0:
							self.save_model(sess, self.saver, model_path['save_path'])
							self.save_result_report(result_path)
							save_figure(result_path)
						if epoch > 100:
							if flag:
								break
				except tf.errors.OutOfRangeError:
					print('Queue out of range occured! Cleaning queue...!')
				finally:
					coord.request_stop()
			coord.join(treads)
			print('training finished!')
			self.save_model(sess, self.saver, model_path['save_path'])
			self.save_result_report(result_path)
			save_figure(result_path)
		plt.ioff()
		plt.show()
		return self.__summarized_report()

	def start_STL_train(self, model_path, task_name, reload=False):
		def early_stop(epoch, task_name):
			task = self.multi_task_dic[task_name]
			Flag = False
			if epoch >= 100:
				if task['testing_accurcy_history'][-1] > 0.7:
						Flag = True
			return Flag
		display_step = 50
		epoch_his = []
		plt.ion()
		fig = plt.figure(task_name)
		with tf.Session() as sess:
			coord = tf.train.Coordinator()
			treads = tf.train.start_queue_runners(sess=sess, coord=coord)
			tf.summary.FileWriter('logs/', sess.graph)

			if reload:
					self.reload_model(sess, self.saver, model_path['reload_path'])
			else:
				sess.run(tf.global_variables_initializer())
			with tf.device('/gpu:0'):
				for epoch in range(self.iter_epoch):
					self.run_multi_task(sess, task_name)
					if epoch % display_step == 0 and epoch is not 0:
						self.run_task_evaluate(sess, fig, epoch, task_name=task_name)
						# print()
						epoch_his.append(epoch)

					if epoch % 500 == 0 and epoch is not 0:
						self.save_model(sess, self.saver, model_path['save_path'])

					# flag = early_stop(epoch, task_name)
					# if flag:
						# break
			coord.request_stop()
			coord.join(treads)
			print('training finished!')
			self.save_model(sess, self.saver, model_path['save_path'])
			# self.save_result_report(result_path)
			plt.ioff()
		# plt.show()—

class CNN_RNN_2_test(Multitask_Neural_Network):
	def __init__(self, input_data_shape_2, output_data_shape_2, config):

		network_1 = self.CNN_RNN_1(input_data_shape_2, output_data_shape_2, config)
		'''
		network_3 = self.CNN_RNN_period(input_data_shape_period, input_data_shape_period, config)

		network_4 = self.CNN_RNN_trend(input_data_shape_trend, input_data_shape_trend, config)

		if config.trainable_w_fn == True:
			self.trainable_var_fusion(network_1, network_3, network_4)
		else:
			self.elementwise_fusion(network_1, network_2, combine_fn = self.combine_fn)
		'''
		#self.tl_share_output.print_layers()

	def CNN_RNN_1(self, input_data_shape, output_data_shape, config):
		self.__parse_config(config)
		self.build_MTL(input_data_shape, output_data_shape)
		if self.STPP == True:
			tl_output = self.__build_CNN_RNN_STPP(self.Xs)
		else:
			tl_output = self.__build_CNN_RNN(self.Xs)
		return self.build_flatten_layer(tl_output)

	def CNN_RNN_period(self, input_data_shape, output_data_shape, config):
		#input_data_shape[3] = 1
		self.__parse_config(config)
		self.build_MTL_period(input_data_shape, output_data_shape)
		if self.STPP == True:
			tl_output = self.__build_CNN_RNN_STPP_period(self.Xs_period)
		else:
			tl_output = self.__build_CNN_RNN(self.Xs_period)
		return self.build_flatten_layer_period(tl_output)

	def CNN_RNN_trend(self, input_data_shape, output_data_shape, config):
		#input_data_shape[3] = 1
		self.__parse_config(config)
		self.build_MTL_trend(input_data_shape, output_data_shape)
		if self.STPP == True:
			tl_output = self.__build_CNN_RNN_STPP_trend(self.Xs_trend)
		else:
			tl_output = self.__build_CNN_RNN(self.Xs_trend)
		return self.build_flatten_layer_trend(tl_output)

	def external_FC(self, input_data_shape, output_data_shape, config):
		self.__parse_config(config)
		self.build_MTL_2(input_data_shape, output_data_shape)
		with tf.variable_scope('FC'):
			network = tl.layers.InputLayer(self.Xs_2, name='input_layer_external')
			network = tl.layers.BatchNormLayer(network, name='batchnorm_layer_FC')
			network = tl.layers.DropconnectDenseLayer(network, keep=0.85, n_units=self.embedded, act=tf.nn.relu, name='FC_1')
			network = tl.layers.DropconnectDenseLayer(network, keep=0.85, n_units=512, act=tf.nn.relu, name='FC_2')
			#network = tl.layers.DropconnectDenseLayer(network, keep=0.85, n_units=512, act=tf.nn.relu, name='FC_3')
			print('--FC shape:{}'.format(network.outputs.get_shape()))
		return network

	def __build_CNN_RNN_STPP(self, Xs):
		def build_CNN_network(input_X, is_training=1):
			with tf.variable_scope('CNN'):
				CNN_input = tf.reshape(input_X, [-1, self.input_vertical, self.input_horizontal, self.input_channel])
				print('CNN_one_input shape:{}'.format(CNN_input))
				network = tl.layers.InputLayer(CNN_input, name='input_layer')
				network = tl.layers.BatchNormLayer(network, is_train=is_training, name='batchnorm_layer_1')
				print('network output shape:{}'.format(network.outputs.get_shape()))
				network_5x5 = tl.layers.Conv2dLayer(
					network,
					act=self.CNN_layer_activation_fn,
					W_init=self.CNN_layer_1_5x5_conv_Winit,
					b_init=tf.constant_initializer(value=0.01),
					shape=self.CNN_layer_1_5x5_kernel_shape,
					strides=self.CNN_layer_1_5x5_kernel_strides,
					padding='SAME',
					name='cnn_layer_1_5x5')
				print('network 5x5 C output shape:{}'.format(network_5x5.outputs.get_shape()))

				network_5x5 = tl.layers.PoolLayer(
					network_5x5,
					ksize=self.CNN_layer_1_5x5_pooling_ksize,
					strides=self.CNN_layer_1_5x5_pooling_strides,
					padding='SAME',
					pool=self.CNN_layer_1_5x5_pooling,
					name='pool_layer_1_5x5')
				print('network 5x5 P output shape:{}'.format(network_5x5.outputs.get_shape()))

				network_3x3 = tl.layers.Conv2dLayer(
					network,
					act=self.CNN_layer_activation_fn,
					W_init=self.CNN_layer_1_3x3_conv_Winit,
					b_init=tf.constant_initializer(value=0.01),
					shape=self.CNN_layer_1_3x3_kernel_shape,
					strides=self.CNN_layer_1_3x3_kernel_strides,
					padding='SAME',
					name='cnn_layer_1_3x3')
				print('network 3x3 C output shape:{}'.format(network_3x3.outputs.get_shape()))

				network_3x3 = tl.layers.PoolLayer(
					network_3x3,
					ksize=self.CNN_layer_1_3x3_pooling_ksize,
					strides=self.CNN_layer_1_3x3_pooling_strides,
					padding='SAME',
					pool=self.CNN_layer_1_3x3_pooling,
					name='pool_layer_1_3x3')
				print('network 3x3 P output shape:{}'.format(network_3x3.outputs.get_shape()))

				network = tl.layers.ConcatLayer(layer=[network_3x3, network_5x5], concat_dim=3, name='concate_layer_1')
				print('network Cancat output shape:{}'.format(network.outputs.get_shape()))

				network = tl.layers.PoolLayer(
					network,
					ksize=self.CNN_layer_1_pooling_ksize,
					strides=self.CNN_layer_1_pooling_strides,
					padding='VALID',
					pool=self.CNN_layer_1_pooling,
					name='pool_layer_1')
				print('network L1 P output shape:{}'.format(network.outputs.get_shape()))

				network = tl.layers.DropoutLayer(network, keep=self.keep_rate, name='drop_1')
				network = tl.layers.Conv2dLayer(
					network,
					act=self.CNN_layer_activation_fn,
					W_init=self.CNN_layer_2_conv_Winit,
					b_init=tf.constant_initializer(value=0.01),
					shape=self.CNN_layer_2_kernel_shape,
					strides=self.CNN_layer_2_strides,
					padding='SAME',
					name='cnn_layer_2')
				print('network L2 C output shape:{}'.format(network.outputs.get_shape()))

				self.CNN_layer_2_pooling_ksize = [1, math.ceil(int(network.outputs.get_shape()[1]) / self.CNN_layer_2_STPP_Ps_1), 
					math.ceil(int(network.outputs.get_shape()[2]) / self.CNN_layer_2_STPP_Ps_1), 1]
				self.CNN_layer_2_pooling_strides = self.CNN_layer_2_pooling_ksize

				STPP_network_1 = tl.layers.PoolLayer(
					network,
					ksize=self.CNN_layer_2_pooling_ksize,
					strides=self.CNN_layer_2_pooling_strides,
					padding='SAME',
					pool=self.CNN_layer_2_pooling,
					name='STPP_layer_1')
				print('STPP network L2 P output shape:{}'.format(STPP_network_1.outputs.get_shape()))

				STPP_network_1 = tl.layers.FlattenLayer(STPP_network_1, name='STPP_flatten_4x4_1')

				self.CNN_layer_2_pooling_ksize = [1, math.ceil(int(network.outputs.get_shape()[1]) / self.CNN_layer_2_STPP_Ps_2), 
					math.ceil(int(network.outputs.get_shape()[2]) / self.CNN_layer_2_STPP_Ps_2), 1]
				self.CNN_layer_2_pooling_strides = self.CNN_layer_2_pooling_ksize

				STPP_network_2 = tl.layers.PoolLayer(
					network,
					ksize=self.CNN_layer_2_pooling_ksize,
					strides=self.CNN_layer_2_pooling_strides,
					padding='SAME',
					pool=self.CNN_layer_2_pooling,
					name='STPP_layer_2')
				print('STPP network L2 P output shape:{}'.format(STPP_network_2.outputs.get_shape()))

				STPP_network_2 = tl.layers.FlattenLayer(STPP_network_2, name='STPP_flatten_2x2_1')
				network = tl.layers.ConcatLayer([STPP_network_1, STPP_network_2], name = 'STPP_concat_layer_1')

				network = tl.layers.DropoutLayer(network, keep=self.keep_rate, name='drop_2')
				# network = tl.layers.FlattenLayer(network, name='flatten_layer')

				# network = tl.layers.DenseLayer(network, n_units=512, act=tf.nn.relu, name='fc_1')
				# network = tl.layers.DropoutLayer(network, keep=0.7, name='drop_3')
				print('network output shape:{}'.format(network.outputs.get_shape()))
				return network

		def build_bi_RNN_network(input_X, is_training=1):
			def make_gaussan_state_initial(scope_name, stddev=self.RNN_init_state_noise_stddev):
				with tf.variable_scope(scope_name):
					init_state = tf.Variable(np.zeros((self.RNN_num_layers, 2, self.batch_size, self.RNN_hidden_node_size), dtype=np.float32), trainable=True)
					result_intital_state = tf.cond(self.add_noise, lambda: init_state + tf.random_normal(tf.shape(init_state), stddev=stddev), lambda: init_state)

					result_intital_state = tf.unpack(result_intital_state, axis=0)
					result_intital_state = tuple(
						[tf.nn.rnn_cell.LSTMStateTuple(result_intital_state[idx][0], result_intital_state[idx][1]) for idx in range(self.RNN_num_layers)])
				return result_intital_state

			# print('rnn network input shape :{}'.format(input_X.outputs.get_shape()))
			with tf.variable_scope('BI_RNN'):
				input_X = tl.layers.BatchNormLayer(input_X, is_train=is_training, name='batchnorm_layer_1')
				print('CNN_RNN_rnn shape:{}'.format(input_X.outputs.get_shape()))
				network = tl.layers.ReshapeLayer(input_X, shape=[-1, self.RNN_num_step, int(input_X.outputs._shape[-1])], name='reshape_layer_1')
				print('CNN_RNN_rnn_before shape:{}'.format(network.outputs.get_shape()))

				network = tl.layers.BiRNNLayer(
					network,
					cell_fn=self.RNN_cell,
					cell_init_args=self.RNN_cell_init_args,
					n_hidden=self.RNN_hidden_node_size,
					initializer=self.RNN_initializer,
					n_steps=self.RNN_num_step,
					#fw_initial_state=make_gaussan_state_initial('fw'),
					#bw_initial_state=make_gaussan_state_initial('bw'),
					return_last=True,
					return_seq_2d=False,
					n_layer=self.RNN_num_layers,
					dropout=(self.keep_rate, self.keep_rate),
					name='layer_1')
				print('CNN_RNN_rnn_after shape:{}'.format(network.outputs.get_shape()))
				return network

		with tf.variable_scope('CNN_RNN'):
			tl_CNN_output = build_CNN_network(Xs, is_training=False)

			#cnn_flat = tl.layers.FlattenLayer(tl_CNN_output, name='CNN_flatten')
			network = tl.layers.BatchNormLayer(tl_CNN_output, is_train=False, name='batchnorm_layer_1')
			tl_RNN_output = build_bi_RNN_network(network, is_training=False)
			# print('CNN output:{}  rnn output:{}'.format(tl_CNN_output.outputs.get_shape().as_list(), tl_RNN_output.outputs.get_shape().as_list()))
			'''
			rnn_flat = tl.layers.FlattenLayer(tl_RNN_output, name='RNN_flatten')
			cnn_flat = tl.layers.ReshapeLayer(cnn_flat, [-1, self.input_temporal * cnn_flat.outputs.get_shape().as_list()[-1]])
			CNN_RNN_ouput_tl = tl.layers.ConcatLayer([cnn_flat, rnn_flat], concat_dim=1, name='CNN_RNN_concat')
			CNN_RNN_ouput_tl = tl.layers.DropoutLayer(CNN_RNN_ouput_tl, keep=0.9, name='dropout_layer')
			'''
			CNN_RNN_ouput_tl = tl_RNN_output

		return CNN_RNN_ouput_tl

	def __build_CNN_RNN_STPP_period(self, Xs):
		def build_CNN_network(input_X, is_training=1):
			with tf.variable_scope('CNN_period'):
				CNN_input = tf.reshape(input_X, [-1, self.input_vertical_period, self.input_horizontal_period, self.input_channel_period])
				print('CNN_one_input shape:{}'.format(CNN_input))
				network = tl.layers.InputLayer(CNN_input, name='input_layer_period')
				network = tl.layers.BatchNormLayer(network, is_train=is_training, name='batchnorm_layer_period')
				print('network output shape:{}'.format(network.outputs.get_shape()))
				network_5x5 = tl.layers.Conv2dLayer(
					network,
					act=self.CNN_layer_activation_fn,
					W_init=self.CNN_layer_1_5x5_conv_Winit,
					b_init=tf.constant_initializer(value=0.01),
					shape=self.CNN_layer_1_5x5_kernel_shape,
					strides=self.CNN_layer_1_5x5_kernel_strides,
					padding='SAME',
					name='cnn_layer_1_5x5_period')
				print('network 5x5 C output shape:{}'.format(network_5x5.outputs.get_shape()))

				network_5x5 = tl.layers.PoolLayer(
					network_5x5,
					ksize=self.CNN_layer_1_5x5_pooling_ksize,
					strides=self.CNN_layer_1_5x5_pooling_strides,
					padding='SAME',
					pool=self.CNN_layer_1_5x5_pooling,
					name='pool_layer_1_5x5_period')
				print('network 5x5 P output shape:{}'.format(network_5x5.outputs.get_shape()))

				network_3x3 = tl.layers.Conv2dLayer(
					network,
					act=self.CNN_layer_activation_fn,
					W_init=self.CNN_layer_1_3x3_conv_Winit,
					b_init=tf.constant_initializer(value=0.01),
					shape=self.CNN_layer_1_3x3_kernel_shape,
					strides=self.CNN_layer_1_3x3_kernel_strides,
					padding='SAME',
					name='cnn_layer_1_3x3_period')
				print('network 3x3 C output shape:{}'.format(network_3x3.outputs.get_shape()))

				network_3x3 = tl.layers.PoolLayer(
					network_3x3,
					ksize=self.CNN_layer_1_3x3_pooling_ksize,
					strides=self.CNN_layer_1_3x3_pooling_strides,
					padding='SAME',
					pool=self.CNN_layer_1_3x3_pooling,
					name='pool_layer_1_3x3_period')
				print('network 3x3 P output shape:{}'.format(network_3x3.outputs.get_shape()))

				network = tl.layers.ConcatLayer(layer=[network_3x3, network_5x5], concat_dim=3, name='concate_layer_period')
				print('network Cancat output shape:{}'.format(network.outputs.get_shape()))

				network = tl.layers.PoolLayer(
					network,
					ksize=self.CNN_layer_1_pooling_ksize,
					strides=self.CNN_layer_1_pooling_strides,
					padding='VALID',
					pool=self.CNN_layer_1_pooling,
					name='pool_layer_period')
				print('network L1 P output shape:{}'.format(network.outputs.get_shape()))

				network = tl.layers.DropoutLayer(network, keep=self.keep_rate, name='drop_period')
				network = tl.layers.Conv2dLayer(
					network,
					act=self.CNN_layer_activation_fn,
					W_init=self.CNN_layer_2_conv_Winit,
					b_init=tf.constant_initializer(value=0.01),
					shape=self.CNN_layer_2_kernel_shape,
					strides=self.CNN_layer_2_strides,
					padding='SAME',
					name='cnn_layer_period')
				print('network L2 C output shape:{}'.format(network.outputs.get_shape()))

				self.CNN_layer_2_pooling_ksize = [1, math.ceil(int(network.outputs.get_shape()[1]) / self.CNN_layer_2_STPP_Ps_1), 
					math.ceil(int(network.outputs.get_shape()[2]) / self.CNN_layer_2_STPP_Ps_1), 1]
				self.CNN_layer_2_pooling_strides = self.CNN_layer_2_pooling_ksize

				STPP_network_1 = tl.layers.PoolLayer(
					network,
					ksize=self.CNN_layer_2_pooling_ksize,
					strides=self.CNN_layer_2_pooling_strides,
					padding='SAME',
					pool=self.CNN_layer_2_pooling,
					name='STPP_layer_1_period')
				print('STPP network L2 P output shape:{}'.format(STPP_network_1.outputs.get_shape()))

				STPP_network_1 = tl.layers.FlattenLayer(STPP_network_1, name='STPP_flatten_4x4_1_period')

				self.CNN_layer_2_pooling_ksize = [1, math.ceil(int(network.outputs.get_shape()[1]) / self.CNN_layer_2_STPP_Ps_2), 
					math.ceil(int(network.outputs.get_shape()[2]) / self.CNN_layer_2_STPP_Ps_2), 1]
				self.CNN_layer_2_pooling_strides = self.CNN_layer_2_pooling_ksize

				STPP_network_2 = tl.layers.PoolLayer(
					network,
					ksize=self.CNN_layer_2_pooling_ksize,
					strides=self.CNN_layer_2_pooling_strides,
					padding='SAME',
					pool=self.CNN_layer_2_pooling,
					name='STPP_layer_2_period')
				print('STPP network L2 P output shape:{}'.format(STPP_network_2.outputs.get_shape()))

				STPP_network_2 = tl.layers.FlattenLayer(STPP_network_2, name='STPP_flatten_2x2_1_period')
				network = tl.layers.ConcatLayer([STPP_network_1, STPP_network_2], name = 'STPP_concat_layer_1_period')

				network = tl.layers.DropoutLayer(network, keep=self.keep_rate, name='drop_2_period')
				# network = tl.layers.FlattenLayer(network, name='flatten_layer')

				# network = tl.layers.DenseLayer(network, n_units=512, act=tf.nn.relu, name='fc_1')
				# network = tl.layers.DropoutLayer(network, keep=0.7, name='drop_3')
				print('network output shape:{}'.format(network.outputs.get_shape()))
				return network

		def build_bi_RNN_network(input_X, is_training=1):
			def make_gaussan_state_initial(scope_name, stddev=self.RNN_init_state_noise_stddev):
				with tf.variable_scope(scope_name):
					init_state = tf.Variable(np.zeros((self.RNN_num_layers, 2, self.batch_size, self.RNN_hidden_node_size), dtype=np.float32), trainable=True)
					result_intital_state = tf.cond(self.add_noise_period, lambda: init_state + tf.random_normal(tf.shape(init_state), stddev=stddev), lambda: init_state)

					result_intital_state = tf.unpack(result_intital_state, axis=0)
					result_intital_state = tuple(
						[tf.nn.rnn_cell.LSTMStateTuple(result_intital_state[idx][0], result_intital_state[idx][1]) for idx in range(self.RNN_num_layers)])
				return result_intital_state

			# print('rnn network input shape :{}'.format(input_X.outputs.get_shape()))
			with tf.variable_scope('BI_RNN_period'):
				input_X = tl.layers.BatchNormLayer(input_X, is_train=is_training, name='batchnorm_layer_1_period')
				print('CNN_RNN_rnn shape:{}'.format(input_X.outputs.get_shape()))
				network = tl.layers.ReshapeLayer(input_X, shape=[-1, self.RNN_num_step, int(input_X.outputs._shape[-1])], name='reshape_layer_1_period')
				print('CNN_RNN_rnn_before shape:{}'.format(network.outputs.get_shape()))

				network = tl.layers.BiRNNLayer(
					network,
					cell_fn=self.RNN_cell,
					cell_init_args=self.RNN_cell_init_args,
					n_hidden=self.RNN_hidden_node_size,
					initializer=self.RNN_initializer,
					n_steps=self.RNN_num_step,
					#fw_initial_state=make_gaussan_state_initial('fw'),
					#bw_initial_state=make_gaussan_state_initial('bw'),
					return_last=True,
					return_seq_2d=False,
					n_layer=self.RNN_num_layers,
					dropout=(self.keep_rate, self.keep_rate),
					name='layer_1_period')
				print('CNN_RNN_rnn_after shape:{}'.format(network.outputs.get_shape()))
				return network

		with tf.variable_scope('CNN_RNN_period'):
			tl_CNN_output = build_CNN_network(Xs, is_training=True)

			#cnn_flat = tl.layers.FlattenLayer(tl_CNN_output, name='CNN_flatten')
			network = tl.layers.BatchNormLayer(tl_CNN_output, is_train=False, name='batchnorm_layer_period')
			tl_RNN_output = build_bi_RNN_network(network, is_training=True)
			# print('CNN output:{}  rnn output:{}'.format(tl_CNN_output.outputs.get_shape().as_list(), tl_RNN_output.outputs.get_shape().as_list()))
			'''
			rnn_flat = tl.layers.FlattenLayer(tl_RNN_output, name='RNN_flatten')
			cnn_flat = tl.layers.ReshapeLayer(cnn_flat, [-1, self.input_temporal * cnn_flat.outputs.get_shape().as_list()[-1]])
			CNN_RNN_ouput_tl = tl.layers.ConcatLayer([cnn_flat, rnn_flat], concat_dim=1, name='CNN_RNN_concat')
			CNN_RNN_ouput_tl = tl.layers.DropoutLayer(CNN_RNN_ouput_tl, keep=0.9, name='dropout_layer')
			'''
			CNN_RNN_ouput_tl = tl_RNN_output

		return CNN_RNN_ouput_tl

	def __build_CNN_RNN_STPP_trend(self, Xs):
		def build_CNN_network(input_X, is_training=1):
			with tf.variable_scope('CNN_trend'):
				CNN_input = tf.reshape(input_X, [-1, self.input_vertical_trend, self.input_horizontal_trend, self.input_channel_trend])
				print('CNN_one_input shape:{}'.format(CNN_input))
				network = tl.layers.InputLayer(CNN_input, name='input_layer_trend')
				network = tl.layers.BatchNormLayer(network, is_train=is_training, name='batchnorm_layer_1_trend')
				print('network output shape:{}'.format(network.outputs.get_shape()))
				network_5x5 = tl.layers.Conv2dLayer(
					network,
					act=self.CNN_layer_activation_fn,
					W_init=self.CNN_layer_1_5x5_conv_Winit,
					b_init=tf.constant_initializer(value=0.01),
					shape=self.CNN_layer_1_5x5_kernel_shape,
					strides=self.CNN_layer_1_5x5_kernel_strides,
					padding='SAME',
					name='cnn_layer_1_5x5_trend')
				print('network 5x5 C output shape:{}'.format(network_5x5.outputs.get_shape()))

				network_5x5 = tl.layers.PoolLayer(
					network_5x5,
					ksize=self.CNN_layer_1_5x5_pooling_ksize,
					strides=self.CNN_layer_1_5x5_pooling_strides,
					padding='SAME',
					pool=self.CNN_layer_1_5x5_pooling,
					name='pool_layer_1_5x5_trend')
				print('network 5x5 P output shape:{}'.format(network_5x5.outputs.get_shape()))

				network_3x3 = tl.layers.Conv2dLayer(
					network,
					act=self.CNN_layer_activation_fn,
					W_init=self.CNN_layer_1_3x3_conv_Winit,
					b_init=tf.constant_initializer(value=0.01),
					shape=self.CNN_layer_1_3x3_kernel_shape,
					strides=self.CNN_layer_1_3x3_kernel_strides,
					padding='SAME',
					name='cnn_layer_1_3x3_trend')
				print('network 3x3 C output shape:{}'.format(network_3x3.outputs.get_shape()))

				network_3x3 = tl.layers.PoolLayer(
					network_3x3,
					ksize=self.CNN_layer_1_3x3_pooling_ksize,
					strides=self.CNN_layer_1_3x3_pooling_strides,
					padding='SAME',
					pool=self.CNN_layer_1_3x3_pooling,
					name='pool_layer_1_3x3_trend')
				print('network 3x3 P output shape:{}'.format(network_3x3.outputs.get_shape()))

				network = tl.layers.ConcatLayer(layer=[network_3x3, network_5x5], concat_dim=3, name='concate_layer_1')
				print('network Cancat output shape:{}'.format(network.outputs.get_shape()))

				network = tl.layers.PoolLayer(
					network,
					ksize=self.CNN_layer_1_pooling_ksize,
					strides=self.CNN_layer_1_pooling_strides,
					padding='VALID',
					pool=self.CNN_layer_1_pooling,
					name='pool_layer_1_trend')
				print('network L1 P output shape:{}'.format(network.outputs.get_shape()))

				network = tl.layers.DropoutLayer(network, keep=self.keep_rate, name='drop_1_trend')
				network = tl.layers.Conv2dLayer(
					network,
					act=self.CNN_layer_activation_fn,
					W_init=self.CNN_layer_2_conv_Winit,
					b_init=tf.constant_initializer(value=0.01),
					shape=self.CNN_layer_2_kernel_shape,
					strides=self.CNN_layer_2_strides,
					padding='SAME',
					name='cnn_layer_2_trend')
				print('network L2 C output shape:{}'.format(network.outputs.get_shape()))

				self.CNN_layer_2_pooling_ksize = [1, math.ceil(int(network.outputs.get_shape()[1]) / self.CNN_layer_2_STPP_Ps_1), 
					math.ceil(int(network.outputs.get_shape()[2]) / self.CNN_layer_2_STPP_Ps_1), 1]
				self.CNN_layer_2_pooling_strides = self.CNN_layer_2_pooling_ksize

				STPP_network_1 = tl.layers.PoolLayer(
					network,
					ksize=self.CNN_layer_2_pooling_ksize,
					strides=self.CNN_layer_2_pooling_strides,
					padding='SAME',
					pool=self.CNN_layer_2_pooling,
					name='STPP_layer_1_trend')
				print('STPP network L2 P output shape:{}'.format(STPP_network_1.outputs.get_shape()))

				STPP_network_1 = tl.layers.FlattenLayer(STPP_network_1, name='STPP_flatten_4x4_1_trend')

				self.CNN_layer_2_pooling_ksize = [1, math.ceil(int(network.outputs.get_shape()[1]) / self.CNN_layer_2_STPP_Ps_2), 
					math.ceil(int(network.outputs.get_shape()[2]) / self.CNN_layer_2_STPP_Ps_2), 1]
				self.CNN_layer_2_pooling_strides = self.CNN_layer_2_pooling_ksize

				STPP_network_2 = tl.layers.PoolLayer(
					network,
					ksize=self.CNN_layer_2_pooling_ksize,
					strides=self.CNN_layer_2_pooling_strides,
					padding='SAME',
					pool=self.CNN_layer_2_pooling,
					name='STPP_layer_2_trend')
				print('STPP network L2 P output shape:{}'.format(STPP_network_2.outputs.get_shape()))

				STPP_network_2 = tl.layers.FlattenLayer(STPP_network_2, name='STPP_flatten_2x2_1_trend')
				network = tl.layers.ConcatLayer([STPP_network_1, STPP_network_2], name = 'STPP_concat_layer_1_trend')

				network = tl.layers.DropoutLayer(network, keep=self.keep_rate, name='drop_2_trend')
				# network = tl.layers.FlattenLayer(network, name='flatten_layer')

				# network = tl.layers.DenseLayer(network, n_units=512, act=tf.nn.relu, name='fc_1')
				# network = tl.layers.DropoutLayer(network, keep=0.7, name='drop_3')
				print('network output shape:{}'.format(network.outputs.get_shape()))
				return network

		def build_bi_RNN_network(input_X, is_training=1):
			def make_gaussan_state_initial(scope_name, stddev=self.RNN_init_state_noise_stddev):
				with tf.variable_scope(scope_name):
					init_state = tf.Variable(np.zeros((self.RNN_num_layers, 2, self.batch_size, self.RNN_hidden_node_size), dtype=np.float32), trainable=True)
					result_intital_state = tf.cond(self.add_noise_trend, lambda: init_state + tf.random_normal(tf.shape(init_state), stddev=stddev), lambda: init_state)

					result_intital_state = tf.unpack(result_intital_state, axis=0)
					result_intital_state = tuple(
						[tf.nn.rnn_cell.LSTMStateTuple(result_intital_state[idx][0], result_intital_state[idx][1]) for idx in range(self.RNN_num_layers)])
				return result_intital_state

			# print('rnn network input shape :{}'.format(input_X.outputs.get_shape()))
			with tf.variable_scope('BI_RNN_trend'):
				input_X = tl.layers.BatchNormLayer(input_X, is_train=is_training, name='batchnorm_layer_1_trend')
				print('CNN_RNN_rnn shape:{}'.format(input_X.outputs.get_shape()))
				network = tl.layers.ReshapeLayer(input_X, shape=[-1, self.RNN_num_step, int(input_X.outputs._shape[-1])], name='reshape_layer_1_trend')
				print('CNN_RNN_rnn_before shape:{}'.format(network.outputs.get_shape()))

				network = tl.layers.BiRNNLayer(
					network,
					cell_fn=self.RNN_cell,
					cell_init_args=self.RNN_cell_init_args,
					n_hidden=self.RNN_hidden_node_size,
					initializer=self.RNN_initializer,
					n_steps=self.RNN_num_step,
					#fw_initial_state=make_gaussan_state_initial('fw'),
					#bw_initial_state=make_gaussan_state_initial('bw'),
					return_last=True,
					return_seq_2d=False,
					n_layer=self.RNN_num_layers,
					dropout=(self.keep_rate, self.keep_rate),
					name='layer_1_trend')
				print('CNN_RNN_rnn_after shape:{}'.format(network.outputs.get_shape()))
				return network

		with tf.variable_scope('CNN_RNN_trend'):
			tl_CNN_output = build_CNN_network(Xs, is_training=True)

			#cnn_flat = tl.layers.FlattenLayer(tl_CNN_output, name='CNN_flatten')
			network = tl.layers.BatchNormLayer(tl_CNN_output, is_train=False, name='batchnorm_layer_1_trend')
			tl_RNN_output = build_bi_RNN_network(network, is_training=True)
			# print('CNN output:{}  rnn output:{}'.format(tl_CNN_output.outputs.get_shape().as_list(), tl_RNN_output.outputs.get_shape().as_list()))
			'''
			rnn_flat = tl.layers.FlattenLayer(tl_RNN_output, name='RNN_flatten')
			cnn_flat = tl.layers.ReshapeLayer(cnn_flat, [-1, self.input_temporal * cnn_flat.outputs.get_shape().as_list()[-1]])
			CNN_RNN_ouput_tl = tl.layers.ConcatLayer([cnn_flat, rnn_flat], concat_dim=1, name='CNN_RNN_concat')
			CNN_RNN_ouput_tl = tl.layers.DropoutLayer(CNN_RNN_ouput_tl, keep=0.9, name='dropout_layer')
			'''
			CNN_RNN_ouput_tl = tl_RNN_output

		return CNN_RNN_ouput_tl

	def __build_CNN_RNN(self, Xs):
		def build_CNN_network(input_X, is_training=1):
			with tf.variable_scope('CNN'):
				CNN_input = tf.reshape(input_X, [-1, self.input_vertical, self.input_horizontal, self.input_channel])
				# print('CNN_input shape:{}'.format(CNN_input))
				network = tl.layers.InputLayer(CNN_input, name='input_layer')
				network = tl.layers.BatchNormLayer(network, is_train=is_training, name='batchnorm_layer_1')
				network_5x5 = tl.layers.Conv2dLayer(
					network,
					act=self.CNN_layer_activation_fn,
					W_init=self.CNN_layer_1_5x5_conv_Winit,
					b_init=tf.constant_initializer(value=0.01),
					shape=self.CNN_layer_1_5x5_kernel_shape,
					strides=self.CNN_layer_1_5x5_kernel_strides,
					padding='SAME',
					name='cnn_layer_1_5x5')

				network_5x5 = tl.layers.PoolLayer(
					network_5x5,
					ksize=self.CNN_layer_1_5x5_pooling_ksize,
					strides=self.CNN_layer_1_5x5_pooling_strides,
					padding='SAME',
					pool=self.CNN_layer_1_5x5_pooling,
					name='pool_layer_1_5x5')

				network_3x3 = tl.layers.Conv2dLayer(
					network,
					act=self.CNN_layer_activation_fn,
					W_init=self.CNN_layer_1_3x3_conv_Winit,
					b_init=tf.constant_initializer(value=0.01),
					shape=self.CNN_layer_1_3x3_kernel_shape,
					strides=self.CNN_layer_1_3x3_kernel_strides,
					padding='SAME',
					name='cnn_layer_1_3x3')

				network_3x3 = tl.layers.PoolLayer(
					network_3x3,
					ksize=self.CNN_layer_1_3x3_pooling_ksize,
					strides=self.CNN_layer_1_3x3_pooling_strides,
					padding='SAME',
					pool=self.CNN_layer_1_3x3_pooling,
					name='pool_layer_1_3x3')

				network = tl.layers.ConcatLayer(layer=[network_3x3, network_5x5], concat_dim=3, name='concate_layer_1')

				network = tl.layers.PoolLayer(
					network,
					ksize=self.CNN_layer_1_pooling_ksize,
					strides=self.CNN_layer_1_pooling_strides,
					padding='SAME',
					pool=self.CNN_layer_1_pooling,
					name='pool_layer_1')

				network = tl.layers.DropoutLayer(network, keep=self.keep_rate, name='drop_1')
				network = tl.layers.Conv2dLayer(
					network,
					act=self.CNN_layer_activation_fn,
					W_init=self.CNN_layer_2_conv_Winit,
					b_init=tf.constant_initializer(value=0.01),
					shape=self.CNN_layer_2_kernel_shape,
					strides=self.CNN_layer_2_strides,
					padding='SAME',
					name='cnn_layer_2')

				network = tl.layers.PoolLayer(
					network,
					ksize=self.CNN_layer_2_pooling_ksize,
					strides=self.CNN_layer_2_pooling_strides,
					padding='SAME',
					pool=self.CNN_layer_2_pooling,
					name='pool_layer_2')
				network = tl.layers.DropoutLayer(network, keep=self.keep_rate, name='drop_2')
				# network = tl.layers.FlattenLayer(network, name='flatten_layer')

				# network = tl.layers.DenseLayer(network, n_units=512, act=tf.nn.relu, name='fc_1')
				# network = tl.layers.DropoutLayer(network, keep=0.7, name='drop_3')
				print('network output shape:{}'.format(network.outputs.get_shape()))
				return network

		def build_bi_RNN_network(input_X, is_training=1):
			def make_gaussan_state_initial(scope_name, stddev=self.RNN_init_state_noise_stddev):
				with tf.variable_scope(scope_name):
					init_state = tf.Variable(np.zeros((self.RNN_num_layers, 2, self.batch_size, self.RNN_hidden_node_size), dtype=np.float32), trainable=True)
					result_intital_state = tf.cond(self.add_noise, lambda: init_state + tf.random_normal(tf.shape(init_state), stddev=stddev), lambda: init_state)

					result_intital_state = tf.unstack(result_intital_state, axis=0)
					result_intital_state = tuple(
						[tf.nn.rnn_cell.LSTMStateTuple(result_intital_state[idx][0], result_intital_state[idx][1]) for idx in range(self.RNN_num_layers)])
				return result_intital_state

			# print('rnn network input shape :{}'.format(input_X.outputs.get_shape()))
			with tf.variable_scope('BI_RNN'):
				input_X = tl.layers.BatchNormLayer(input_X, is_train=is_training, name='batchnorm_layer_1')
				# print(input_X.outputs.get_shape())
				#logger.debug('before input reshape for biRNN:{}'.format(input_X.outputs.get_shape()))
				network = tl.layers.ReshapeLayer(input_X, shape=[-1, self.RNN_num_step, int(input_X.outputs._shape[-1])], name='reshape_layer_1')
				#logger.debug('before input biRNN shape:{}'.format(network.outputs.get_shape()))
				network = tl.layers.BiRNNLayer(
					network,
					cell_fn=self.RNN_cell,
					cell_init_args=self.RNN_cell_init_args,
					n_hidden=self.RNN_hidden_node_size,
					initializer=self.RNN_initializer,
					n_steps=self.RNN_num_step,
					fw_initial_state=make_gaussan_state_initial('fw'),
					bw_initial_state=make_gaussan_state_initial('bw'),
					return_last=False,  # only output the last step
					return_seq_2d=False,
					n_layer=self.RNN_num_layers,
					dropout=(self.keep_rate, self.keep_rate),
					name='layer_1')
				#logger.debug('biRNN output shape:{}'.format(network.outputs.get_shape()))
				return network  # [batch_size, 2 * n_hidden]

		with tf.variable_scope('CNN_RNN'):
			tl_CNN_output = build_CNN_network(Xs, is_training=True)

			cnn_flat = tl.layers.FlattenLayer(tl_CNN_output, name='CNN_flatten')
			network = tl.layers.BatchNormLayer(cnn_flat, is_train=False, name='batchnorm_layer_1')
			tl_RNN_output = build_bi_RNN_network(network, is_training=True)
			# print('CNN output:{}  rnn output:{}'.format(tl_CNN_output.outputs.get_shape().as_list(), tl_RNN_output.outputs.get_shape().as_list()))
			'''
			rnn_flat = tl.layers.FlattenLayer(tl_RNN_output, name='RNN_flatten')
			cnn_flat = tl.layers.ReshapeLayer(cnn_flat, [-1, self.input_temporal * cnn_flat.outputs.get_shape().as_list()[-1]])
			CNN_RNN_ouput_tl = tl.layers.ConcatLayer([cnn_flat, rnn_flat], concat_dim=1, name='CNN_RNN_concat')
			CNN_RNN_ouput_tl = tl.layers.DropoutLayer(CNN_RNN_ouput_tl, keep=0.9, name='dropout_layer')
			'''
			CNN_RNN_ouput_tl = tl_RNN_output

		return CNN_RNN_ouput_tl

	def __build_CNN_RNN_2(self, Xs):
		def build_CNN_network(input_X, is_training=1):
			with tf.variable_scope('CNN_2'):
				CNN_input = tf.reshape(input_X, [-1, self.input_vertical, self.input_horizontal, self.input_channel])
				# print('CNN_input shape:{}'.format(CNN_input))
				network = tl.layers.InputLayer(CNN_input, name='input_layer_2')
				network = tl.layers.BatchNormLayer(network, is_train=is_training, name='batchnorm_layer_1_2')
				network_5x5 = tl.layers.Conv2dLayer(
					network,
					act=self.CNN_layer_activation_fn,
					W_init=self.CNN_layer_1_5x5_conv_Winit,
					b_init=tf.constant_initializer(value=0.01),
					shape=self.CNN_layer_1_5x5_kernel_shape,
					strides=self.CNN_layer_1_5x5_kernel_strides,
					padding='SAME',
					name='cnn_layer_1_5x5_2')

				network_5x5 = tl.layers.PoolLayer(
					network_5x5,
					ksize=self.CNN_layer_1_5x5_pooling_ksize,
					strides=self.CNN_layer_1_5x5_pooling_strides,
					padding='SAME',
					pool=self.CNN_layer_1_5x5_pooling,
					name='pool_layer_1_5x5_2')

				network_3x3 = tl.layers.Conv2dLayer(
					network,
					act=self.CNN_layer_activation_fn,
					W_init=self.CNN_layer_1_3x3_conv_Winit,
					b_init=tf.constant_initializer(value=0.01),
					shape=self.CNN_layer_1_3x3_kernel_shape,
					strides=self.CNN_layer_1_3x3_kernel_strides,
					padding='SAME',
					name='cnn_layer_1_3x3_2')

				network_3x3 = tl.layers.PoolLayer(
					network_3x3,
					ksize=self.CNN_layer_1_3x3_pooling_ksize,
					strides=self.CNN_layer_1_3x3_pooling_strides,
					padding='SAME',
					pool=self.CNN_layer_1_3x3_pooling,
					name='pool_layer_1_3x3_2')

				network = tl.layers.ConcatLayer(layer=[network_3x3, network_5x5], concat_dim=3, name='concate_layer_1_2')

				network = tl.layers.PoolLayer(
					network,
					ksize=self.CNN_layer_1_pooling_ksize,
					strides=self.CNN_layer_1_pooling_strides,
					padding='SAME',
					pool=self.CNN_layer_1_pooling,
					name='pool_layer_1_2')

				network = tl.layers.DropoutLayer(network, keep=self.keep_rate, name='drop_1_2')
				network = tl.layers.Conv2dLayer(
					network,
					act=self.CNN_layer_activation_fn,
					W_init=self.CNN_layer_2_conv_Winit,
					b_init=tf.constant_initializer(value=0.01),
					shape=self.CNN_layer_2_kernel_shape,
					strides=self.CNN_layer_2_strides,
					padding='SAME',
					name='cnn_layer_2_2')

				network = tl.layers.PoolLayer(
					network,
					ksize=self.CNN_layer_2_pooling_ksize,
					strides=self.CNN_layer_2_pooling_strides,
					padding='SAME',
					pool=self.CNN_layer_2_pooling,
					name='pool_layer_2_2')
				network = tl.layers.DropoutLayer(network, keep=self.keep_rate, name='drop_2_2')
				# network = tl.layers.FlattenLayer(network, name='flatten_layer')

				# network = tl.layers.DenseLayer(network, n_units=512, act=tf.nn.relu, name='fc_1')
				# network = tl.layers.DropoutLayer(network, keep=0.7, name='drop_3')
				print('network output shape:{}'.format(network.outputs.get_shape()))
				return network

		def build_bi_RNN_network(input_X, is_training=1):
			def make_gaussan_state_initial(scope_name, stddev=self.RNN_init_state_noise_stddev):
				with tf.variable_scope(scope_name):
					init_state = tf.Variable(np.zeros((self.RNN_num_layers, 2, self.batch_size, self.RNN_hidden_node_size), dtype=np.float32), trainable=True)
					result_intital_state = tf.cond(self.add_noise, lambda: init_state + tf.random_normal(tf.shape(init_state), stddev=stddev), lambda: init_state)

					result_intital_state = tf.unstack(result_intital_state, axis=0)
					result_intital_state = tuple(
						[tf.nn.rnn_cell.LSTMStateTuple(result_intital_state[idx][0], result_intital_state[idx][1]) for idx in range(self.RNN_num_layers)])
				return result_intital_state

			# print('rnn network input shape :{}'.format(input_X.outputs.get_shape()))
			with tf.variable_scope('BI_RNN_2'):
				input_X = tl.layers.BatchNormLayer(input_X, is_train=is_training, name='batchnorm_layer_1_2')
				# print(input_X.outputs.get_shape())
				#logger.debug('before input reshape for biRNN:{}'.format(input_X.outputs.get_shape()))
				network = tl.layers.ReshapeLayer(input_X, shape=[-1, self.RNN_num_step, int(input_X.outputs._shape[-1])], name='reshape_layer_1')
				#logger.debug('before input biRNN shape:{}'.format(network.outputs.get_shape()))
				network = tl.layers.BiRNNLayer(
					network,
					cell_fn=self.RNN_cell,
					cell_init_args=self.RNN_cell_init_args,
					n_hidden=self.RNN_hidden_node_size,
					initializer=self.RNN_initializer,
					n_steps=self.RNN_num_step,
					fw_initial_state=make_gaussan_state_initial('fw'),
					bw_initial_state=make_gaussan_state_initial('bw'),
					return_last=False,  # only output the last step
					return_seq_2d=False,
					n_layer=self.RNN_num_layers,
					dropout=(self.keep_rate, self.keep_rate),
					name='layer_1_2')
				#logger.debug('biRNN output shape:{}'.format(network.outputs.get_shape()))
				return network  # [batch_size, 2 * n_hidden]

		with tf.variable_scope('CNN_RNN_2'):
			tl_CNN_output = build_CNN_network(Xs, is_training=True)

			cnn_flat = tl.layers.FlattenLayer(tl_CNN_output, name='CNN_flatten_2')
			network = tl.layers.BatchNormLayer(cnn_flat, is_train=False, name='batchnorm_layer_1_2')
			tl_RNN_output = build_bi_RNN_network(network, is_training=True)
			# print('CNN output:{}  rnn output:{}'.format(tl_CNN_output.outputs.get_shape().as_list(), tl_RNN_output.outputs.get_shape().as_list()))
			'''
			rnn_flat = tl.layers.FlattenLayer(tl_RNN_output, name='RNN_flatten')
			cnn_flat = tl.layers.ReshapeLayer(cnn_flat, [-1, self.input_temporal * cnn_flat.outputs.get_shape().as_list()[-1]])
			CNN_RNN_ouput_tl = tl.layers.ConcatLayer([cnn_flat, rnn_flat], concat_dim=1, name='CNN_RNN_concat')
			CNN_RNN_ouput_tl = tl.layers.DropoutLayer(CNN_RNN_ouput_tl, keep=0.9, name='dropout_layer')
			'''
			CNN_RNN_ouput_tl = tl_RNN_output

		return CNN_RNN_ouput_tl

	def __parse_config(self, config):
		self.iter_epoch = config.iter_epoch
		self.batch_size = config.batch_size
		self.learning_rate = config.learning_rate
		self.weight_decay = config.weight_decay
		self.keep_rate = config.keep_rate
		self.combine_fn = config.combine_fn
		self.trainable_w_fn = config.trainable_w_fn
		self.STPP = config.STPP
		self.embedded = config.embedded

		self.RNN_num_layers = config.RNN_num_layers
		self.RNN_num_step = config.RNN_num_step
		self.RNN_hidden_node_size = config.RNN_hidden_node_size

		self.RNN_cell = self.parse_RNN_cell(config.RNN_cell)
		self.RNN_cell_init_args = config.RNN_cell_init_args
		self.RNN_init_state_noise_stddev = config.RNN_init_state_noise_stddev
		self.RNN_initializer = self.parse_initializer_method(config.RNN_initializer)

		self.CNN_layer_activation_fn = self.parse_activation(config.CNN_layer_activation_fn)
		self.CNN_layer_1_5x5_kernel_shape = config.CNN_layer_1_5x5_kernel_shape
		self.CNN_layer_1_5x5_kernel_strides = config.CNN_layer_1_5x5_kernel_strides
		self.CNN_layer_1_5x5_conv_Winit = self.parse_initializer_method(config.CNN_layer_1_5x5_conv_Winit)

		self.CNN_layer_1_5x5_pooling = self.parse_pooling(config.CNN_layer_1_5x5_pooling)
		self.CNN_layer_1_5x5_pooling_ksize = config.CNN_layer_1_5x5_pooling_ksize
		self.CNN_layer_1_5x5_pooling_strides = config.CNN_layer_1_5x5_pooling_strides

		self.CNN_layer_1_3x3_kernel_shape = config.CNN_layer_1_3x3_kernel_shape
		self.CNN_layer_1_3x3_kernel_strides = config.CNN_layer_1_3x3_kernel_strides
		self.CNN_layer_1_3x3_conv_Winit = self.parse_initializer_method(config.CNN_layer_1_3x3_conv_Winit)

		self.CNN_layer_1_3x3_pooling = self.parse_pooling(config.CNN_layer_1_3x3_pooling)
		self.CNN_layer_1_3x3_pooling_ksize = config.CNN_layer_1_3x3_pooling_ksize
		self.CNN_layer_1_3x3_pooling_strides = config.CNN_layer_1_3x3_pooling_strides

		self.CNN_layer_1_pooling = self.parse_pooling(config.CNN_layer_1_pooling)
		self.CNN_layer_1_pooling_ksize = config.CNN_layer_1_pooling_ksize
		self.CNN_layer_1_pooling_strides = config.CNN_layer_1_pooling_strides

		self.CNN_layer_2_kernel_shape = config.CNN_layer_2_kernel_shape
		self.CNN_layer_2_strides = config.CNN_layer_2_strides
		self.CNN_layer_2_conv_Winit = self.parse_initializer_method(config.CNN_layer_2_conv_Winit)

		self.CNN_layer_2_pooling_ksize = config.CNN_layer_2_pooling_ksize
		self.CNN_layer_2_STPP_Ps_1 = config.CNN_layer_2_STPP_Ps_1
		self.CNN_layer_2_STPP_Ps_2 = config.CNN_layer_2_STPP_Ps_2
		self.CNN_layer_2_pooling_strides = config.CNN_layer_2_pooling_strides
		self.CNN_layer_2_pooling = self.parse_pooling(config.CNN_layer_2_pooling)

		self.fully_connected_W_init = self.parse_initializer_method(config.fully_connected_W_init)
		self.fully_connected_units = config.fully_connected_units
		self.prediction_layer_1_W_init = self.parse_initializer_method(config.prediction_layer_1_W_init)
		self.prediction_layer_1_uints = config.prediction_layer_1_uints
		self.prediction_layer_2_W_init = self.parse_initializer_method(config.prediction_layer_2_W_init)
		self.prediction_layer_keep_rate = config.prediction_layer_keep_rate

		self.hyper_config = config