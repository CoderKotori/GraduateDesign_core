import mmh3
from bitarray import bitarray
import numpy as np


class BloomFilter:
    def __init__(self, mode='train', m=1, k=1):
        '''
        :param m: number of vector length, which stores features
        :param k: number of hash function
        :param mode: if mode=train, the initialize vector v with 0, use insert function to add new features
                        if mode=test, initialize vector v with file 'bf.npy', use judge function to judge whether
                        the new element is probably in the feature vector v
        '''

        if mode is 'train':
            self.v = bitarray(m)
            self.v.setall(0)
            self.m = m
            self.k = k
        elif mode in ['test', 'verify']:
            param = np.load('files/bf_param.npy')
            self.m = param[0]
            self.k = param[1]
            self.v = bitarray()
            with open('files/bf.bin', 'rb') as fr:
                self.v.fromfile(fr)
        else:
            raise ValueError('Invalid mode: "%s"' % mode)
        self.mode = mode

    def calc_position(self, e):
        e = str(e)
        position = np.zeros(self.k)
        for i in range(self.k):
            position[i] = mmh3.hash(e, 28389400292 + i * 10) % self.m
        return position

    def _insert(self, e):
        e = str(e)
        position = self.calc_position(e)
        for i in position:
            self.v[int(i)] = 1

    def _judge(self, e):
        e = str(e)
        position = self.calc_position(e)
        res = True
        for i in position:
            res = res and self.v[int(i)]
        return res

    def run(self, data, result=None):
        if self.mode is 'train':
            param = [self.m, self.k]
            param = np.array(param)
            np.save('files/bf_param.npy', param)
            N = data.shape[0]
            for i in range(N):
                self._insert(data[i])
            with open('files/bf.bin', 'wb') as fw:
                self.v.tofile(fw)
            return None
        elif self.mode is 'test':
            tp = 0
            tn = 0
            fp = 0
            fn = 0
            count = data.shape[0]
            if result is None or result.shape[0] != count:
                raise Exception('data not match')
            for j in range(count):
                if self._judge(data[j]) is True:

                    if result[j] == 0:
                        tn += 1.0
                    else:
                        fn += 1.0
                else:
                    if result[j] == 0:
                        fp += 1.0
                    else:
                        tp += 1.0
            return tp, tn, fp, fn, count
        elif self.mode is 'verify':
            tmp = data.reshape(-1)
            verify = True
            for i in range(tmp.shape[0]):
                verify = verify and self._judge(tmp[i])
            return verify
        else:
            print 'this situation can not happen'


if __name__ == '__main__':
    from Data import *
    import math
    d = Data()

    data_str = np.load('files/data_str.npy')
    features_normal = np.load('files/features_normal.npy')
    result = d.load_data()[:, d.binary_result].astype(int)
    n = features_normal.shape[0]
    p = 0.01
    m = int(math.ceil(-n * np.log(p) / np.log(2) ** 2))
    k = int(math.ceil(np.log(2) * m / n))
    bf_train = BloomFilter(mode='train', m=m, k=k)
    bf_train.run(features_normal)
    test_start = 0
    test_end = 200000
    tp, tn, fp, fn, count = BloomFilter(mode='test').run(data_str[test_start:test_end], result[test_start:test_end])
    print 'true positive: ', tp
    print 'true negative: ', tn
    print 'false positive: ', fp
    print 'false negative: ', fn