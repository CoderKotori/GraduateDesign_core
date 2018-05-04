import numpy as np


def minmax(array):
    l = array.shape[0]
    max = -99999999
    min = 99999999
    for i in range(l):
        if array[i] is None:
            continue
        else:
            num = float(array[i])
            if num < min:
                min = num
            if num > max:
                max = num
    return min, max


def discrete(array, classes):
    min, max = minmax(array)
    print min, max
    interval = (max - min) / classes
    disc = []
    pre_item = min
    for i in range(classes):
        print pre_item
        disc.append(pre_item + 0.5 * interval)
        pre_item += interval
    return np.array(disc)


def nearest(disc, array):
    for i in range(array.shape[0]):
        item = array[i]
        if item is None:
            continue
        else:
            tmp = (disc - item) ** 2
            array[i] = disc[np.argmin(tmp)]
    return array


def nearest_plus(classes, data):
    shape_c = classes.shape
    shape_d = data.shape
    if len(shape_c) < 2 or len(shape_d) < 2 or shape_c[1] != shape_d[1]:
        print 'dimensions not match'
        return None
    else:
        num_d = shape_d[0]
        for i in range(num_d):
            if data[i, 0] is None:
                continue
            else:
                tmp = classes - data[i]
                tmp = tmp ** 2
                tmp = np.sum(tmp, axis=1)
                data[i] = classes[np.argmin(tmp), :]
        return data


if __name__ == '__main__':
    from Data import *
    from Kmeans import *
    import init
    from BloomFilter import *
    import math
    import time
    # optimize the granularity of discretization
    d = Data()
    data = d.load_data()
    # np.savetxt('data.txt', data.astype(float))
    raw_pid = data[:, d.gain:d.rate + 1]
    pressure_measurement = data[:, d.pressure_measurement]
    setpoint = data[:, d.setpoint]

    crc = np.load('files/pred_crcrate.npy')
    ti = np.load('files/pred_timeinterval.npy')
    result = d.load_data()[:, d.binary_result].astype(int)

    disc_result = []
    for i in range(5, 50, 5):
        for j in range(2, 30, 2):
            for k in range(2, 30, 2):
                before = time.time()
                kmeans_pid = Kmeans(raw_pid, None, k=i)
                pid = kmeans_pid.calc(None, 20, 2000)
                pm = discrete(pressure_measurement.sort()[:int(pressure_measurement.shape[0] * 0.9)], j)
                sp = discrete(setpoint, k)
                data_str = init.signature_all(d, crc, ti, pid, pm, sp)

                count = True
                features_normal = []
                for r in range(data_str.shape[0]):
                    if result[r] == 0:
                        if count:
                            features_normal.append(data_str[r])
                            count = False
                        else:
                            if data_str[r] not in features_normal:
                                features_normal.append(data_str[r])
                features_normal = np.array(features_normal)

                n = features_normal.shape[0]
                p = 0.01
                m = int(math.ceil(-n * np.log(p) / np.log(2)**2))
                kk = int(math.ceil(np.log(2) * m / n))
                bf_train = BloomFilter(mode='train', m=m, k=kk)
                bf_train.run(features_normal)
                test_start = 5000
                test_end = 10000
                tp, tn, fp, fn, count = BloomFilter(mode='test').run(data_str[test_start:test_end],
                                                                     result[test_start:test_end])
                disc_result.append([i, j, k, tp, tn, fp, fn])
                print 'number of pid: ', i
                print 'number of pressure measurement: ', j
                print 'number of set point: ', k
                print 'true positive: ', tp / count
                print 'true negative: ', tn / count
                print 'false positive: ', fp / count
                print 'false negative: ', fn / count
                now = time.time()
                print 'run time: ', now - before
    np.save('files/disc_result', np.array(disc_result))