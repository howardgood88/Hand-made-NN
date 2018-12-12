from datetime import datetime, tzinfo, timedelta
from time import sleep, time, mktime, strptime
import calendar
import numpy as np
import pytz
import pickle
import os
import csv
import re
import math
root_path = "/home/qiuhui/Desktop/howard/rain_CDR/"
input_file = "./precipitation.csv"
#input_file = "precipitation.csv"
#ouput_dir = "/home/qiuhui/Desktop"
#ouput_file = "output_precipitation_2013-11-01.csv"

def add_data(rain_data, rain_data_processed):
    
    #prev_intensity = int(rain_data['rain_intensity'][0])
    #prev_coverage = int(rain_data['coverage'][0])
    minute_counter = 0
    hour_counter = 0
    
    # rain_data_processed={}
    for i, each_line in enumerate(rain_data['quadrant_id']):
        date_time = int(rain_data['date_time'][i])
        quadrant_id = int(rain_data['quadrant_id'][i])
        rain_intensity = int(rain_data['rain_intensity'][i])
        coverage = int(rain_data['coverage'][i])
        #rain_type = int(rain_data['rain_type'][i])
        flag = 0
        prev_time = int(rain_data['date_time'][i - 1])
        
        #print("hour_counter = {} hour = {}".format(date_time, date_time) )

        hour = int((date_time % 10000) / 100)
        minute = int(date_time % 100)

        #print("date_time = {} hour_counter = {} hour = {}".format(date_time, hour_counter, hour) )

        while hour_counter != hour or minute != minute_counter:
            if minute_counter - 10 < 0:
                sub = 50
            else:
                sub = 10
            for j in range(4):
                rain_data_processed['date_time'].append(prev_time + sub)
                rain_data_processed['quadrant_id'].append(j + 1)
                rain_data_processed['rain_intensity'].append(int(rain_data['rain_intensity'][i + j]))
                rain_data_processed['coverage'].append(int(rain_data['coverage'][i + j]))
                print("find missing data between {} and {} ,adding new date_time = {} quadrant_id = {} rain_intensity = {} coverage = {}"
                    .format(int(rain_data['date_time'][i - 1]), date_time, prev_time + sub, j + 1, int(rain_data['rain_intensity'][i + j])
                    , int(rain_data['coverage'][i + j])))
            prev_time = prev_time + sub
            print()
            if hour_counter != 23 and minute_counter == 50:
                hour_counter += 1
            elif hour_counter == 23 and minute_counter == 50:
                hour_counter = 0

            if minute_counter != 50:
                minute_counter += 10
            else:
                minute_counter = 0
            flag = 1

        if flag != 1 and quadrant_id == 4:
            if hour_counter != 23 and minute_counter == 50:
                hour_counter += 1
                #print("1")
            elif hour_counter == 23 and minute_counter == 50:
                hour_counter = 0
                #print("2")

            if minute_counter != 50:
                minute_counter += 10
                #print("3")
            else:
                minute_counter = 0
                #print("4")
            #print("if")
        rain_data_processed['date_time'].append(date_time)
        rain_data_processed['quadrant_id'].append(quadrant_id)
        rain_data_processed['rain_intensity'].append(rain_intensity)
        rain_data_processed['coverage'].append(coverage)
    return rain_data_processed

def process_data_to_mildan_grid(rain_data_processed, line): #one day data
    grid_row_num = 100
    grid_column_num = 100
    features_num = 4
    
    grid_list = [ [ [ [ 0 for i in range(features_num) ] for j in range(grid_column_num) ]
        for k in range(grid_row_num) ] for l in range(144) ]

    for i in range(144):
        temp = [ [ 0 for i in range(features_num)] for j in range(4)]
        for j in range(4):
            temp[j][0] = int(rain_data_processed['quadrant_id'][line + i * 4 + j])
            temp[j][1] = int(rain_data_processed['date_time'][line + i * 4 + j])
            temp[j][2] = int(rain_data_processed['rain_intensity'][line + i * 4 + j])
            temp[j][3] = int(rain_data_processed['coverage'][line + i * 4 + j])
        #print(temp[1])
        #print()   

        for row in range(grid_row_num):
            for col in range(grid_column_num):
                if col < 50 and row < 50:
                    grid_list[i][row][col] = temp[0]
                elif col >= 50 and row < 50:
                    grid_list[i][row][col] = temp[1]
                elif col >= 50 and row >= 50:
                    grid_list[i][row][col] = temp[2]
                else:
                    grid_list[i][row][col] = temp[3]
    return grid_list

def save_array(x_array, out_file):
    print('saving file to {}...'.format(out_file))
    np.save(out_file, x_array, allow_pickle=True)

def date_to_timestamp(rain_data_processed):
    for index, element in enumerate(rain_data_processed['date_time']):
        year = int(element / 100000000)
        month = int(element / 1000000 % 100)
        day = int(element / 10000 % 100)
        hour = int(element / 100 % 100)
        minute = int(element % 100)
        second = 0
        #print(element)
        if hour == 24:
            hour = 23
        date_time = datetime(year, month, day, hour, minute, second)#.timetuple()
        date_time = strptime(str(date_time), '%Y-%m-%d %H:%M:%S')
        #print(date_time)
        rain_data_processed['date_time'][index] = mktime(date_time)
        #print(rain_data_processed['date_time'][index])
    return rain_data_processed

def load_data_from_file(file):

    rain_data = {
        'quadrant_id': [],
        'date_time': [],
        'rain_intensity': [],
        'coverage': [],
        'rain_type': []
    }

    rain_data_processed = {
        'date_time': [],
        'quadrant_id': [],
        'rain_intensity': [],
        'coverage': [],
        #'rain_type': []
    }

    with open(input_file, errors='replace') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        print('start to load data from {}..'.format(input_file))
        for split_line in readCSV:
            quadrant_id = int(split_line[1].strip())
            #print(date_time)
            date_time = int(split_line[0].strip())
            rain_intensity = int(split_line[2].strip())
            coverage = int(split_line[3].strip())
            rain_type = int(split_line[4].strip())
            #print("date_time = {}".format(date_time))
            
            if rain_type != 2:
                rain_data['date_time'].append(date_time)
                rain_data['quadrant_id'].append(quadrant_id)
                rain_data['rain_intensity'].append(rain_intensity)
                rain_data['coverage'].append(coverage)
                #rain_data['rain_type'].append(rain_type)
            else:
                rain_data['date_time'].append(date_time)
                rain_data['quadrant_id'].append(quadrant_id)
                rain_data['rain_intensity'].append(0)
                rain_data['coverage'].append(0)
                #rain_data['rain_type'].append(0)
    
    rain_data_processed = add_data(rain_data, rain_data_processed)
    rain_data_processed = date_to_timestamp(rain_data_processed)
    del rain_data
    #print(rain_data_processed)
    line = 0
    month = 11
    date = 1
    for i in range(61):
        X_image = process_data_to_mildan_grid(rain_data_processed, line)
        #print(X_image)
        line += 576
        output_dir = os.path.dirname(file) + '/data_preproccessed/'
        output_filename = 'output_' + \
            (os.path.splitext(os.path.basename(file))[0] + '_2013-' + str(month) + '-' + str(date))
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        
        save_array(X_image, output_dir + output_filename)
        date += 1
        if month == 11 and int(date) == 31:
            month += 1
            date = 1
        #X_image = load_array(ouput_dir+ouput_file)

load_data_from_file(input_file)
