from solver import Solver

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
    num_train = 10000
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
    result_grouped, _ = init_input(result.reshape((result.shape[0], 1)), data_str, np.zeros_like(result),
                                   output_num=num_train,
                                   length=seq_length)
    result_grouped = result_grouped.reshape((num_train, -1))
    tmp_result = np.zeros(num_train)
    for i in range(num_train):
        passed = 0
        for j in range(result_grouped.shape[1]):
            if result_grouped[i, j] == 1:
                passed = 1
        tmp_result[i] = passed
    result_grouped = tmp_result
    data_in, data_out = init_input(data, data_str, np.zeros_like(result), output_num=num_train, length=seq_length)
    #  Second: using pre-defined model to discrete data, and generate signature

    pm = np.load('files/disc_pm.npy')
    sp = np.load('files/disc_setpoint.npy')
    pid = np.load('files/pred_pid.npy')
    crc = np.load('files/pred_crcrate.npy')
    ti = np.load('files/pred_timeinterval.npy')

    bf_in = data_in.copy()
    N, T, D = bf_in.shape
    bf_in = bf_in.reshape((-1, D))
    bf_in = signature(bf_in, crc, ti, pid, pm, sp)
    bf_in = bf_in.reshape((N, T))

    data_in = data_in.reshape((-1, D))
    data_in[:, d.pressure_measurement] = nearest(pm, data_in[:, d.pressure_measurement])
    data_in[:, d.setpoint] = nearest(sp, data_in[:, d.setpoint])
    data_in[:, d.crc_rate] = nearest(crc, data_in[:, d.crc_rate])
    data_in[:, d.time_interval] = nearest(ti, data_in[:, d.time_interval])
    data_in[:, d.gain:d.rate + 1] = nearest_plus(pid, data_in[:, d.gain:d.rate + 1])
    data_in = np.concatenate((data_in[:, d.address:d.time], data_in[:, d.time_interval].reshape(-1, 1)),
                             axis=1)
    for i in range(data_in.shape[0]):
        for j in range(data_in.shape[1]):
            if data_in[i, j] is None:
                data_in[i, j] = -1.0
            else:
                data_in[i, j] = float(data_in[i, j])
    data_in = data_in.astype(float)
    data_in = data_in.reshape((N, T, -1))
    lstm_in = data_in[:, :-1]
    lstm_out = data_out[:, :-1]
    lstm_out = lstm_out.reshape(-1)
    for i in range(lstm_out.shape[0]):
        pos = np.where(features == lstm_out[i])[0]
        if len(pos) == 0:
            pos = 0
        else:
            pos = pos[0]
        lstm_out[i] = int(pos)
    lstm_out = lstm_out.astype(int)
    lstm_out = lstm_out.reshape((N, -1))

    bf = BloomFilter(mode='verify')
    lstm = CaptioningRNN(D, output_dim, hidden_dim=512, cell_type='lstm', load_param='files/lstm_params.npy')
    solver = Solver(lstm, {}, update_rule='adam',
                    num_epochs=10,
                    batch_size=100,
                    optim_config={
                        'learning_rate': 5e-3,
                    },
                    lr_decay=0.995,
                    verbose=True, print_every=10)
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(N):
        #  Third: using pre-trained Bloom Filter to verify data, if normal, go on
        if bf.run(bf_in[i]):
            data_verify = {}
            data_verify['verify_in'] = lstm_in[i].reshape((1, T - 1, -1))
            data_verify['verify_out'] = lstm_out[i].reshape((1, -1))
            if solver.verify(data_verify, features):
                if result_grouped[i] == 0:
                    tn += 1
                else:
                    fn += 1
            else:
                if result_grouped[i] == 0:
                    fp += 1
                else:
                    tp += 1
        else:
            if result_grouped[i] == 0:
                fp += 1
            else:
                tp += 1
    count = N + 0.0
    print 'true positive: ', tp
    print 'true negative: ', tn
    print 'false positive: ', fp
    print 'false negative: ', fn
    # Forth: using pre-trained LSTM network to verify data, return the result

    #  Fifth(optional): compare the calculated result with the true result, count tp, tn, fp, fn
