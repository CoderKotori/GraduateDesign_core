import numpy as np


def minmax(array):
    l = array.shape[0]
    max = np.nan
    min = np.inf
    for i in range(l):
        if array[i] is None:
            continue
        else:
            num = float(array[i])
            if min > num:
                min = num
            if max < num:
                max = num
    return min, max


def minmax_plus(array, ratio):
    result_fuck = []
    N = array.shape[0]
    count = 0
    for i in range(N):
        if array[i] is not None:
            result_fuck.append(array[i])
            count += 1
    result_fuck = np.array(result_fuck)
    result_fuck.sort()
    min = result_fuck[0]
    max = result_fuck[(int)(count * ratio)]
    max_real = result_fuck[-1]
    return min, max, max_real


def discrete(array, classes):
    min, max = minmax(array)
    # print min, max
    interval = (max - min) / classes
    disc = []
    pre_item = min
    for i in range(classes):
        # print pre_item
        disc.append(pre_item + 0.5 * interval)
        pre_item += interval
    return np.array(disc)


def discrete_plus(array, classes, ratio):
    min, max, max_real = minmax_plus(array, ratio)
    # print min, max
    interval = (max - min) / classes
    disc = []
    pre_item = min
    for i in range(classes):
        # print pre_item
        disc.append(pre_item + 0.5 * interval)
        pre_item += interval
    disc.append((max + max_real) / 2)
    return np.array(disc)


def nearest(disc, array):
    min, max = minmax(disc)
    distance_max = (max - min)/2
    aaaa = array.copy()
    for i in range(aaaa.shape[0]):
        item = aaaa[i]
        if item is None:
            continue
        else:
            ddisc = disc[:-1]
            tmp = (ddisc - item) ** 2
            aaaa[i] = ddisc[np.argmin(tmp)]
            dist = aaaa[i] - item
            if dist > distance_max:
                aaaa[i] = disc[-1]
    return aaaa


def nearest_plus(classes, data):
    shape_c = classes.shape
    shape_d = data.shape
    if len(shape_c) < 2 or len(shape_d) < 2 or shape_c[1] != shape_d[1]:
        print 'dimensions not match'
        return None
    else:
        num_d = shape_d[0]
        dddd = data.copy()
        for i in range(num_d):
            if dddd[i, 0] is None:
                continue
            else:
                tmp = classes - dddd[i]
                tmp = tmp ** 2
                tmp = np.sum(tmp, axis=1)
                dddd[i] = classes[np.argmin(tmp), :]
        return dddd


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
    crc.sort()
    np.append(crc, 2 * crc[-1] - crc[0])
    ti = np.load('files/pred_timeinterval.npy')
    ti.sort()
    np.append(ti, 2 * ti[-1] - ti[0])
    result = d.load_data()[:, d.binary_result].astype(int)

    disc_result = []
    for i in range(5, 50, 5):
        print 'pid'
        kmeans_pid = Kmeans(raw_pid, None, k=i)
        pid = kmeans_pid.calc(None, 20, 2000)
        for j in range(2, 30, 2):
            print 'pressure measurement'
            pm = discrete_plus(pressure_measurement, j, 0.9)
            for k in range(2, 30, 2):
                # before = time.time()
                print 'set point'
                sp = discrete_plus(setpoint, k, 0.9)
                data_str = init.signature_all(d, crc, ti, pid, pm, sp)

                count = True
                features_normal = []
                for r in range(data_str.shape[0]):
                    if result[r] == 0:
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
                test_end = 15000
                bf_train.mode = 'test'
                tp, tn, fp, fn, count = bf_train.run(data_str[test_start:test_end],
                                                                     result[test_start:test_end])
                disc_result.append([i, j, k, tp, tn, fp, fn])
                print 'number of pid: ', i
                print 'number of pressure measurement: ', j
                print 'number of set point: ', k
                print 'number of normal features and all :', n, ' ', data_str.shape[0]
                print 'true positive: ', tp / count
                print 'true negative: ', tn / count
                print 'false positive: ', fp / count
                print 'false negative: ', fn / count
                # now = time.time()
                # print 'run time: ', now - before
    np.save('files/disc_result', np.array(disc_result))
