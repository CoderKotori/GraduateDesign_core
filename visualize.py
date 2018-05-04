# some tests
tmp_pid = data[:, 4:9]
crc = data[:, 14]
pid = None
pm = data[:, 13]
sp = data[:, 3]
D = tmp_pid.shape[1]
for i in range(tmp_pid.shape[0]):
    if tmp_pid[i, 0] is not None:
        if pid is None:
            pid = tmp_pid[i, :].reshape(1, D)
        else:
            pid = np.concatenate((pid, tmp_pid[i, :].reshape(1, D)), axis=0)

'''add time interval as a new row'''
time_interval = []
for i in range(data.shape[0]):
    if i == 0:
        time_interval.append(0)
    else:
        delta = data[i, 16] - data[i - 1, 16]
        time_interval.append(delta)
ti = np.array(time_interval)
print 'OK'
import matplotlib.pyplot as plt

plt.figure()
plt.subplot(221)
plt.hist(ti, 100)
plt.xlabel('time interval')

plt.subplot(222)
plt.hist(crc, 100)
plt.xlabel('crc rate')

plt.subplot(223)
plt.hist(pm, 100)
plt.xlabel('pressure measurement')

plt.subplot(224)
plt.hist(sp, 100)
plt.xlabel('set point')

plt.show()
N = data.shape[0]
assert False
# myset = set(ti)
# for item in myset:
#     y.append(time_interval.count(item))
from sklearn.decomposition import PCA

pca = PCA(n_components=3)
new_pid = pca.fit_transform(pid)
x = new_pid[:, 0]
y = new_pid[:, 1]
z = new_pid[:, 2]

import mpl_toolkits.mplot3d

ax = plt.subplot(111, projection='3d')
ax.scatter(x, y, z, marker='.')
plt.show()
assert False
