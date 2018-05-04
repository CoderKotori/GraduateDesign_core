from BloomFilter import BloomFilter
from Data import Data
import numpy as np
from init import init_input
#  initial step: load data
d = Data()

tmp = init_input(d.load_data(), d.binary_result)

# data_str = np.load('files/data_str.npy')
# features_normal = np.load('files/features_normal.npy')
# result = d.load_data()[:, d.binary_result].astype(int)
#
# test_start = 5000
# test_end = 10000
# after_bf = BloomFilter(mode='verify').run(data_str[test_start:test_end])  # index of passed data

#  First step: split raw data, three lines in a group, make sure all data in the group is normal

#  Second: using pre-defined model to discrete data, and generate signature

#  Third: using pre-trained Bloom Filter to verify data, if normal, go on

#  Forth: using pre-trained LSTM network to verify data, return the result

#  Fifth(optional): compare the calculated result with the true result, count tp, tn, fp, fn
