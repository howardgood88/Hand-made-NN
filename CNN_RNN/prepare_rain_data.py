'''
generate the 10 minutes rolling data
'''

import data_utility as du
from natsort import natsorted

input_dir_list = ["./data_preproccessed/"]


def print_function_name(method):
	def print_name(*args, **kw):
		print('Function', method.__name__)
		result = method(*args, **kw)
		return result
	return print_name


@print_function_name
def rolling_10_minutes():
	for input_dir in input_dir_list:
		filelist = du.list_all_input_file(input_dir)
		filelist = natsorted(filelist)
		du.load_data_format_roll_10mins(input_dir, filelist)


@print_function_name
def one_hour_max_value():
	for input_dir in input_dir_list:
		filelist = du.list_all_input_file(input_dir)
		filelist = natsorted(filelist)
		du.load_data_hour_max(input_dir, filelist)


@print_function_name
def one_hour_average_value():
	for input_dir in input_dir_list:
	    filelist = du.list_all_input_file(input_dir)
	    filelist = natsorted(filelist)
	    du.load_data_hour_average(input_dir, filelist)


@print_function_name
def one_hour_min_value():
	for input_dir in input_dir_list:
		filelist = du.list_all_input_file(input_dir)
		filelist = natsorted(filelist)
		du.load_data_hour_min(input_dir, filelist)


one_hour_average_value()
one_hour_max_value()
#rolling_10_minutes()
one_hour_min_value()
