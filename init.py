import arff
import os
import numpy as np
from Kmeans import *
from Discretization import *
from Data import *

'''import raw file'''
PROJECT_ROOT = os.path.dirname(os.path.realpath(__file__))
data_file_path = os.path.join(PROJECT_ROOT, "files/IanArffDataset.arff")
data_raw = arff.load(open(data_file_path, 'rb'))

'''initialize data '''
data = data_raw['data']
data = np.array(data)

'''add time interval as a new row'''
time_interval = []
for i in range(data.shape[0]):
    if i == 0:
        time_interval.append(0)
    else:
        delta = data[i, 16] - data[i - 1, 16]
        time_interval.append(delta)
data = np.column_stack((data, time_interval))
np.save('data.npy', data)

d = Data()

'''discrete the following features'''
time_interval = data[:, d.time_interval].reshape(data.shape[0], 1)
kmeans_ti = Kmeans(time_interval, None, k=2)
kmeans_ti.calc('pred_timeinterval.npy', 20, 2000)

crc_rate = data[:, d.crc_rate].reshape(data.shape[0], 1)
kmeans_cr = Kmeans(crc_rate, None, k=2)
kmeans_cr.calc('pred_crcrate.npy', 20, 200)

pid = data[:, d.gain:d.rate + 1]
kmeans_pid = Kmeans(pid, None, k=32)
kmeans_pid.calc('pred_pid.npy', 20, 2000)

pressure_measurement = data[:, d.pressure_measurement]
setpoint = data[:, d.setpoint]
disc_pm = discrete(pressure_measurement, 20)
disc_sp = discrete(setpoint, 10)
np.save('disc_pm.npy', disc_pm)
np.save('disc_setpoint.mpy', disc_sp)

pm = np.load('disc_pm.npy')
setpoint = np.load('disc_setpoint.npy')
crc_rate = np.load('pred_crcrate.npy')
pid = np.load('pred_pid.npy')
ti = np.load('pred_timeinterval.npy')
data[:, d.pressure_measurement] = nearest(pm, data[:, d.pressure_measurement])
data[:, d.setpoint] = nearest(setpoint, data[:, d.setpoint])
data[:, d.crc_rate] = nearest(crc_rate, data[:, d.crc_rate])
data[:, d.time_interval] = nearest(ti, data[:, d.time_interval])
data[:, d.gain:d.rate + 1] = nearest_plus(pid, data[:, d.gain:d.rate + 1])
np.save('data.npy', data)

'''generate signature'''
data = data[:, d.address:d.time]
data = np.concatenate((data, d.load_data()[:, d.time_interval].reshape(-1, 1)), axis=1)
data_str = []
for i in range(data.shape[0]):
    tmp = ''
    for j in range(data.shape[1]):
        if d.gain <= j <= d.rate:
            tmp += '$' + str(data[i, j])
        else:
            tmp += '@' + str(data[i, j])
    data_str.append(tmp)
data_str = np.array(data_str)
np.save('data_str.npy', data_str)

'''
save features
'''
count = True
features = []
for i in range(data_str.shape[0]):
    if count:
        features.append(data_str[i])
        count = False
    else:
        if data_str[i] not in features:
            features.append(data_str[i])
np.save('features.npy', features)

'''
save normal features
'''
count = True
features_normal = []
result = d.load_data()[:, d.binary_result].astype(int)
for i in range(data_str.shape[0]):
    if result[i] == 0:
        if count:
            features_normal.append(data_str[i])
            count = False
        else:
            if data_str[i] not in features_normal:
                features_normal.append(data_str[i])
features_normal = np.array(features_normal)
np.save('features_normal.npy', features_normal)

