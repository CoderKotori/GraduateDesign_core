from Data import Data
import numpy as np
import matplotlib.pyplot as plt

data = Data()
data_npy = data.load_data()
print data_npy
assert True
normal_setpoint = 0
attack_setpoint = 0

normal_count = 0
attack_count = 0
time_pre = 0
for i in range(data_npy.shape[0]):
    # if data_npy[i, data.setpoint] != None:
    #     if data_npy[i, data.binary_result] == '0':
    #         normal_setpoint += float(data_npy[i, data.setpoint])
    #         normal_count += 1
    #     else:
    #         attack_setpoint += float(data_npy[i, data.setpoint])
    #         attack_count += 1
    if data_npy[i, data.time] >= time_pre:
        time_pre = data_npy[i, data.time]
    else:
        print 'Sequence is not continued'
# print 'normal mean: ', normal_setpoint / normal_count
# print 'attack mean: ', attack_setpoint / attack_count
