import numpy as np


def minmax(array):
    l = array.shape[0]
    max = -9999
    min = 9999
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
    pass

