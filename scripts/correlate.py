#!/usr/bin/python3

# numpy.correlate may perform slowly in large arrays (i.e. n = 1e5) because it
# does not use the FFT to compute the convolution; in that case,
# scipy.signal.correlate might be preferable.

import random
import numpy as np
import scipy

#
# Für Step=2 sieht es so aus:
#
# ref    ref[2] ref[3] ref[4] ref[5] ref[6] ref[7] ref[8] ...
# a      ref[0] ref[1] ref[2] ref[3] ref[4] ref[5] ref[6] ...
#

def correlate_pearson(ref, a):
    print(f"ref={ref}, a={a}")
    steps = len(ref) - len(a) + 1
    assert(steps > 0)
    len_a = len(a)

    R = [r for r in range(steps)]
    print(f"R={R}, len(R)={len(R)}")

    R = [np.corrcoef(ref[r:r+len_a], a)[0][1] for r in range(steps)]
    max_index = np.argmax(R)
    print(R)
    for r in R:
        print(r)
    print(f"len(R)={len(R)}, max_index={max_index}, R-max={R[max_index]}")


# Barker code
c = [1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1]

y = [1 if random.random() > 0.5 else -1 for n in range(32)]
y += c
y += [1 if random.random() > 0.5 else -1 for n in range(32)]
Y = np.array(y)
print(f"y: {y}")


def correlate():
    Ryc = np.correlate(Y, c, mode='full')
    T = np.argmax(Ryc)
    print(f"y: {y}")
    print(f"Ryc: {Ryc}")
    print(f"T: {T}, Ryc[T]={Ryc[T]}")

    Ryc = scipy.signal.correlate(Y, c, mode='full')
    T = np.argmax(Ryc)
    print(f"y: {y}")
    print(f"Ryc: {Ryc}")
    print(f"T: {T}, Ryc[T]={Ryc[T]}")

def correlate1():
    print("correlate1()")
    a = np.array([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]])
    out = scipy.spatial.distance.cdist(a, a, 'sqeuclidean')
    print(out)

# r = correlate_pearson(y, c)
correlate1()
