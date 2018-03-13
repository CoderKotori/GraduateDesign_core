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
print 'import raw file'

'''initialize data '''
data = data_raw['data']
data = np.array(data)
print 'initialize data'

'''add time interval as a new row'''
time_interval = []
for i in range(data.shape[0]):
    if i == 0:
        time_interval.append(0)
    else:
        delta = data[i, 16] - data[i - 1, 16]
        time_interval.append(delta)
data = np.column_stack((data, time_interval))
np.save('files/data.npy', data)
print 'add time interval as a new row'

d = Data()

'''discrete the following features'''
time_interval = data[:, d.time_interval].reshape(data.shape[0], 1)
kmeans_ti = Kmeans(time_interval, None, k=2)
kmeans_ti.calc('files/pred_timeinterval.npy', 20, 2000)
print 'time interval done'

crc_rate = data[:, d.crc_rate].reshape(data.shape[0], 1)
kmeans_cr = Kmeans(crc_rate, None, k=2)
kmeans_cr.calc('files/pred_crcrate.npy', 20, 200)
print 'crc rate done'

pid = data[:, d.gain:d.rate + 1]
kmeans_pid = Kmeans(pid, None, k=32)
kmeans_pid.calc('files/pred_pid.npy', 20, 2000)
print 'pid done'

pressure_measurement = data[:, d.pressure_measurement]
setpoint = data[:, d.setpoint]
disc_pm = discrete(pressure_measurement, 20)
disc_sp = discrete(setpoint, 10)
np.save('files/disc_pm.npy', disc_pm)
np.save('files/disc_setpoint.mpy', disc_sp)
print 'pressure measurement and  setpoint done'

pm = np.load('files/disc_pm.npy')
setpoint = np.load('files/disc_setpoint.npy')
crc_rate = np.load('files/pred_crcrate.npy')
pid = np.load('files/pred_pid.npy')
ti = np.load('files/pred_timeinterval.npy')
data[:, d.pressure_measurement] = nearest(pm, data[:, d.pressure_measurement])
data[:, d.setpoint] = nearest(setpoint, data[:, d.setpoint])
data[:, d.crc_rate] = nearest(crc_rate, data[:, d.crc_rate])
data[:, d.time_interval] = nearest(ti, data[:, d.time_interval])
data[:, d.gain:d.rate + 1] = nearest_plus(pid, data[:, d.gain:d.rate + 1])
np.save('files/data.npy', data)
print 'discrete data'

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
np.save('files/data_str.npy', data_str)
print 'generate signature'

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
np.save('files/features.npy', features)
print 'save all features'

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
np.save('files/features_normal.npy', features_normal)
print 'save normal features'
