import arff
import os
import numpy as np
from Kmeans import *
from Discretization import *
from Data import *


def init_input(data, data_str, result, output_num=None, length=10):
    if len(data.shape) > 2:
        raise Exception('unexpected data shape')
    N, _ = data.shape
    pos = 0
    input = []
    output = []
    count = 0
    while pos < N:
        verify = True
        for i in range(length + 1):
            if pos + i < N:
                if int(result[pos + i]) == 1:
                    verify = False
                    pos += i + 1
                    break
            else:
                print 'position out of range'
                return np.array(input), np.array(output)
        if verify:
            if output_num is None:
                pass
            else:
                if count >= output_num:
                    print 'reach the expected count'
                    return np.array(input), np.array(output)
            add_input = data[pos:pos + length]
            add_output = data_str[pos + 1:pos + length + 1]
            input.append(add_input)
            output.append(add_output)
            count += 1
            pos += length
    input = np.array(input)
    output = np.array(output)
    return input, output


def signature(rows, crc, ti, pid, pm, sp):
    data_row = rows.copy()
    data_row[:, 13] = nearest(pm, data_row[:, 13])
    data_row[:, 3] = nearest(sp, data_row[:, 3])
    data_row[:, 14] = nearest(crc, data_row[:, 14])
    time_interval = nearest(ti, data_row[:, 20])
    data_row[:, 4:9] = nearest_plus(pid, data_row[:, 4:9])
    data_row = data_row[:, :16]
    data_row = np.concatenate((data_row, time_interval.reshape(-1, 1)), axis=1)
    data_row_str = []
    for ii in range(data_row.shape[0]):
        sig = ''
        for i in range(data_row.shape[1]):
            if 4 <= i <= 8:
                sig += '$' + str(data_row[ii, i])
            else:
                sig += '@' + str(data_row[ii, i])
        data_row_str.append(sig)
    return np.array(data_row_str)


def signature_all(d, crc, ti, pid, pm, sp):
    data_tmp = d.load_data().copy()
    data_tmp[:, d.pressure_measurement] = nearest(pm, data_tmp[:, d.pressure_measurement])
    data_tmp[:, d.setpoint] = nearest(sp, data_tmp[:, d.setpoint])
    data_tmp[:, d.crc_rate] = nearest(crc, data_tmp[:, d.crc_rate])
    time_interval = nearest(ti, data_tmp[:, d.time_interval])
    data_tmp[:, d.gain:d.rate + 1] = nearest_plus(pid, data_tmp[:, d.gain:d.rate + 1])
    data_tmp = data_tmp[:, d.address:d.time]
    data_tmp = np.concatenate((data_tmp, time_interval.reshape(-1, 1)), axis=1)
    data_str_tmp = []
    for ii in range(data_tmp.shape[0]):
        tmp = ''
        for j in range(data_tmp.shape[1]):
            if d.gain <= j <= d.rate:
                tmp += '$' + str(data_tmp[ii, j])
            else:
                tmp += '@' + str(data_tmp[ii, j])
        data_str_tmp.append(tmp)
    return np.array(data_str_tmp)


if __name__ == '__main__':
    # '''import raw file'''
    # PROJECT_ROOT = os.path.dirname(os.path.realpath(__file__))
    # data_file_path = os.path.join(PROJECT_ROOT, "files/IanArffDataset.arff")
    # data_raw = arff.load(open(data_file_path, 'rb'))
    # print 'import raw file'
    #
    # '''initialize data '''
    # data = data_raw['data']
    # data = np.array(data)
    # print 'initialize data'
    #
    # '''add time interval as a new row'''
    # time_interval = []
    # for i in range(data.shape[0]):
    #     if i == 0:
    #         time_interval.append(0)
    #     else:
    #         delta = data[i, 16] - data[i - 1, 16]
    #         time_interval.append(delta)
    # ti = np.array(time_interval)
    # data = np.column_stack((data, time_interval))
    # np.save('files/data.npy', data)
    # print 'add time interval as a new row'
    #
    d = Data()
    data = d.load_data()
    # assert False
    # '''discrete the following features'''
    # time_interval = data[:, d.time_interval].reshape(data.shape[0], 1)
    # kmeans_ti = Kmeans(time_interval, None, k=2)
    # kmeans_ti.calc('files/pred_timeinterval.npy', 20, 2000)
    # print 'time interval done'
    #
    # crc_rate = data[:, d.crc_rate].reshape(data.shape[0], 1)
    # kmeans_cr = Kmeans(crc_rate, None, k=2)
    # kmeans_cr.calc('files/pred_crcrate.npy', 20, 200)
    # print 'crc rate done'

    # pid = data[:, d.gain:d.rate + 1]
    # kmeans_pid = Kmeans(pid, None, k=32)
    # kmeans_pid.calc('files/pred_pid.npy', 20, 2000)
    # print 'pid done'
    #
    # pressure_measurement = data[:, d.pressure_measurement]
    # setpoint = data[:, d.setpoint]
    # disc_pm = discrete_plus(pressure_measurement, 20, 0.95)
    # disc_sp = discrete_plus(setpoint, 10, 0.95)
    # np.save('files/disc_pm.npy', disc_pm)
    # np.save('files/disc_setpoint.mpy', disc_sp)
    # print 'pressure measurement and  setpoint done'

    # crc = np.load('files/pred_crcrate.npy')
    # crc.sort()
    # np.append(crc, 2 * crc[-1] - crc[0])
    # ti = np.load('files/pred_timeinterval.npy')
    # ti.sort()
    # np.append(ti, 2 * ti[-1] - ti[0])
    # np.save(ti, 'files/pred_timeinterval.npy')
    # np.save(crc, 'files/pred_crcrate.npy')

    pm = np.load('files/disc_pm.npy')
    setpoint = np.load('files/disc_setpoint.npy')
    crc_rate = np.load('files/pred_crcrate.npy')
    pid = np.load('files/pred_pid.npy')
    ti = np.load('files/pred_timeinterval.npy')
    # data[:, d.pressure_measurement] = nearest(pm, data[:, d.pressure_measurement])
    # data[:, d.setpoint] = nearest(setpoint, data[:, d.setpoint])
    # data[:, d.crc_rate] = nearest(crc_rate, data[:, d.crc_rate])
    # data[:, d.time_interval] = nearest(ti, data[:, d.time_interval])
    # data[:, d.gain:d.rate + 1] = nearest_plus(pid, data[:, d.gain:d.rate + 1])
    # np.save('files/data.npy', data)
    # print 'discrete data'

    '''generate signature'''
    data_str = signature_all(d, crc_rate, ti, pid, pm, setpoint)
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
