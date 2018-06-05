from solver import Solver
import time

if __name__ == '__main__':
    from BloomFilter import BloomFilter
    from Data import Data
    import numpy as np
    from init import *
    from Discretization import *

    #  initial step: load data
    from rnn import CaptioningRNN

    d = Data()
    data = d.load_data()
    data_str = np.load('files/data_str.npy')
    features = np.load('files/features_normal.npy')
    output_dim = features.shape[0]
    num_train = 1000
    seq_length = 3
    # tmp = init_input(d.load_data(), d.binary_result)

    # data_str = np.load('files/data_str.npy')
    # features_normal = np.load('files/features_normal.npy')
    # result = d.load_data()[:, d.binary_result].astype(int)
    #
    # test_start = 5000
    # test_end = 10000
    # after_bf = BloomFilter(mode='verify').run(data_str[test_start:test_end])  # index of passed data

    result = d.load_data()[:, d.binary_result].astype(int)
    #  First step: split raw data, three lines in a group, make sure all data in the group is normal
    # result_grouped, _ = init_input(result.reshape((result.shape[0], 1)), data_str, np.zeros_like(result),
    #                                output_num=num_train,
    #                                length=seq_length)
    # result_grouped = result_grouped.reshape((num_train, -1))
    # tmp_result = np.zeros(num_train)
    # for i in range(num_train):
    #     passed = 0
    #     for j in range(result_grouped.shape[1]):
    #         if result_grouped[i, j] == 1:
    #             passed = 1
    #     tmp_result[i] = passed
    # result_grouped = tmp_result
    # data_in, data_out = init_input(data, data_str, np.zeros_like(result), output_num=num_train, length=seq_length)
    #  Second: using pre-defined model to discrete data, and generate signature

    pm = np.load('files/disc_pm.npy')
    sp = np.load('files/disc_setpoint.npy')
    pid = np.load('files/pred_pid.npy')
    crc = np.load('files/pred_crcrate.npy')
    ti = np.load('files/pred_timeinterval.npy')
    data_in = data[5000:8000]
    result_grouped = result[5000:8000]
    bf_in = data_in.copy()
    N, D = bf_in.shape
    # bf_in = bf_in.reshape((-1, D))
    bf_in = signature(bf_in, crc, ti, pid, pm, sp)

    data_in[:, d.pressure_measurement] = nearest(pm, data_in[:, d.pressure_measurement])
    data_in[:, d.setpoint] = nearest(sp, data_in[:, d.setpoint])
    data_in[:, d.crc_rate] = nearest(crc, data_in[:, d.crc_rate])
    data_in[:, d.time_interval] = nearest(ti, data_in[:, d.time_interval])
    data_in[:, d.gain:d.rate + 1] = nearest_plus(pid, data_in[:, d.gain:d.rate + 1])
    data_in = np.concatenate((data_in[:, d.address:d.time], data_in[:, d.time_interval].reshape(-1, 1)),
                             axis=1)
    min_max = np.load('files/min_max.npy')
    for i in range(data_in.shape[0]):
        for j in range(data_in.shape[1]):
            min, max = min_max[j]
            if data_in[i, j] is None:
                data_in[i, j] = -1.0
            else:
                data_in[i, j] = float(data_in[i, j])
                data_in[i, j] = (data_in[i, j] - min) / (max - min)
    c_in = data_in.astype(float).reshape((N, 1, -1))
    # data_in = data_in.reshape((N, T, -1))
    # lstm_in = data_in[:, :-1]
    # lstm_out = data_out[:, :-1]
    # lstm_out = lstm_out.reshape(-1)
    # for i in range(lstm_out.shape[0]):
    #     pos = np.where(features == lstm_out[i])[0]
    #     if len(pos) == 0:
    #         pos = 0
    #     else:
    #         pos = pos[0]
    #     lstm_out[i] = int(pos)
    # lstm_out = lstm_out.astype(int)
    # lstm_out = lstm_out.reshape((N, -1))

    bf = BloomFilter(mode='verify')
    lstm = CaptioningRNN(D, output_dim, hidden_dim=3096, cell_type='lstm', load_param='files/lstm_params.npy')
    solver = Solver(lstm, {})
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    h10 = np.zeros((1, 3096))
    h20 = np.zeros((1, 3096))
    c10 = 0
    c20 = 0
    c_pre = None
    prev_h1, prev_c1, prev_h2, prev_c2 = h10, c10, h20, c20
    count = 0
    for i in range(N):
        #  Third: using pre-trained Bloom Filter to verify data, if normal, go on
        verify = bf.run(bf_in[i])
        # count = 0
        if verify == 0:
            if count == 0:
                prev_h1, prev_c1, prev_h2, prev_c2 = h10, c10, h20, c20
                c_pre = c_in[i]
                count += 1
            elif count == 1:
                prev_h1, prev_c1, prev_h2, prev_c2, _ = lstm.verify(c_pre, prev_h1, prev_c1, prev_h2, prev_c2, features,
                                                                    bf_in[i])
                c_pre = c_in[i]
                count += 1
            else:
                prev_h1, prev_c1, prev_h2, prev_c2, verify = lstm.verify(c_pre, prev_h1, prev_c1, prev_h2, prev_c2,
                                                                         features, bf_in[i])
                if verify == 1:
                    c_pre = None
                    count = 0
                else:
                    c_pre = c_in[i]
                    count += 1

            if verify == 0:
                if result_grouped[i] == 0:
                    tn += 1.0
                else:
                    fn += 1.0
            else:
                if result_grouped[i] == 0:
                    fp += 1.0
                else:
                    tp += 1.0
        else:
            count = 0
            if result_grouped[i] == 0:
                fp += 1.0
            else:
                tp += 1.0

    count = N + 0.0
    print 'true positive: ', tp
    print 'true negative: ', tn
    print 'false positive: ', fp
    print 'false negative: ', fn
    # Forth: using pre-trained LSTM network to verify data, return the result

    #  Fifth(optional): compare the calculated result with the true result, count tp, tn, fp, fn
