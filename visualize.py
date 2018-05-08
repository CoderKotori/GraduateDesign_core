from Data import *
import matplotlib.pyplot as plt

# some tests
d = Data()
data = d.load_data()
tmp_pid = data[:, d.gain:d.rate+1]
crc = data[:, d.crc_rate]
pid = []
tmp_pm = data[:, d.pressure_measurement]
tmp_sp = data[:, d.setpoint]
ti = data[:, d.time_interval]
D = tmp_pid.shape[1]

for i in range(data.shape[0]):
    if tmp_pid[i, 0] is not None:
        pid.append(tmp_pid[i, :])

# plt.figure()
# plt.subplot(221)
# _, bin = np.histogram(ti, bins=100)
plt.hist(ti, bins=100)
plt.title('time interval')
plt.show()
print 'time interval done'

plt.subplot(222)
plt.hist(crc, 100)
plt.title('crc rate')
print 'crc rate done'

plt.subplot(223)
plt.hist(pm, 100)
plt.title('pressure measurement')
print 'pressure measurement done'

plt.subplot(224)
plt.hist(sp, 100)
plt.title('set point')
print 'set point done'

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
