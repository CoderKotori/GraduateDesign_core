import numpy as np


class Kmeans:
    def __init__(self, data, datapath=None, k=0):
        '''

        :param k: number of clusters, if data path is not None, read k from data file
        :param data: shape(N, D)
        '''
        self.data = None
        N, D = data.shape
        for i in range(N):
            if data[i, 0] is not None:
                if self.data is None:
                    self.data = data[i, :].reshape(1, D)
                    print 'the first'
                else:
                    self.data = np.concatenate((self.data, data[i, :].reshape(1, D)), axis=0)
                    print 'add column ', i
        print self.data.shape
        self.N, self.D = self.data.shape
        if datapath is not None:
            tmp = np.load(datapath)
            self.k = tmp[0]
            self.pred = tmp[1:-1]
            if self.pred.shape[1] != self.D:
                raise Exception("dimensions not match")
        else:
            self.k = k
            self.pred = np.zeros((self.k, self.D))
            for j in range(self.k):
                index = np.random.randint(0, self.N)
                self.pred[j] = self.data[index, :]

    def distance(self, batch):
        distance = []
        iter = int(self.N / batch)
        if batch * iter < self.N:
            iter += 1
        for i in range(iter):
            cur = i * batch
            if i == iter - 1:
                next = self.N
            else:
                next = cur + batch
            data = self.data[cur:next]
            print 'calculating distance: ', cur, '~', next
            dist = data.reshape((next - cur, 1, self.D)) - self.pred.reshape((1, self.k, self.D))
            dist = np.sum(dist ** 2, axis=-1)
            dist = np.float64(dist)
            dist = np.sqrt(dist)
            if i == 0:
                distance = dist
            else:
                distance = np.concatenate((distance, dist), axis=0)
        return distance

    def update(self, dist):
        print 'before update: ', self.pred
        c = np.zeros(self.N)
        for i in range(self.N):
            c[i] = np.argmin(dist[i])
        # assert False
        for k in range(self.k):
            mask = c == k
            count = np.sum(mask)
            print count
            if count == 0:
                count = 1
            val = np.zeros(self.D)
            for i in range(mask.shape[0]):
                if mask[i]:
                    val += self.data[i]
            # assert False
            self.pred[k] = val / count
        print 'after update: ', self.pred

    def calc(self, file_name, iterator, batch_size):
        for i in range(iterator):
            dist = self.distance(batch_size)
            self.update(dist)
        np.save(file_name, self.pred)
