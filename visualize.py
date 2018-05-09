from Data import *
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import mpl_toolkits.mplot3d


# some tests
d = Data()
data = d.load_data()
tmp_pid = data[:, d.gain:d.rate+1]
crc = data[:, d.crc_rate]
pid = []
pm = []
sp = []
tmp_pm = data[:, d.pressure_measurement]
tmp_sp = data[:, d.setpoint]
ti = data[:, d.time_interval]

for i in range(data.shape[0]):
    if tmp_pid[i, 0] is not None:
        pid.append(tmp_pid[i, :])
    if tmp_pm[i] is not None:
        pm.append(tmp_pm[i])
    if tmp_sp[i] is not None:
        sp.append(tmp_sp[i])


# plt.hist(ti.tolist(), 100)
# plt.title('time interval')
# plt.savefig('files/time_interval.png')
# plt.show()
# print 'time interval done'
#
# plt.hist(crc.tolist(), 100)
# plt.title('crc rate')
# plt.savefig('files/crc_rate.png')
# plt.show()
# print 'crc rate done'
#
# plt.hist(pm, 100)
# plt.title('pressure measurement')
# plt.savefig('files/pressure_measurement.png')
# plt.show()
# print 'pressure measurement done'
#
# plt.hist(sp, 100)
# plt.title('set point')
# plt.savefig('files/setpoint.png')
# plt.show()
# print 'set point done'

# pca = PCA(n_components=3)
# new_pid = pca.fit_transform(pid)
# x = new_pid[:, 0]
# y = new_pid[:, 1]
# z = new_pid[:, 2]

# ax = plt.subplot(111, projection='3d')
# color = np.random.rand(x.shape[0])
# aa = ax.scatter(x, y, z, marker='.', c=color)
# plt.title('pid-3d')
# plt.colorbar(aa)
# plt.show()


test_result = np.load('files/disc_result.npy')
pid = test_result[:, 0]
pm = test_result[:, 1]
sp = test_result[:, 2]
res = test_result[:, 5]
res = (res - np.min(res)) / (np.max(res) - np.min(res))
ax = plt.subplot(111, projection='3d')
aa = ax.scatter(pid, pm, sp, marker='.', c=res)
plt.colorbar(aa)
plt.show()