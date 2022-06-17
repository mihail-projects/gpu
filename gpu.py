from numba import jit
from numba import cuda
import numpy as np
from timeit import default_timer as timer

#CPU
def func(a):
    for i in range(10000000):
        a[i] += 1

#GPU
@jit(nopython=True, parallel = True, fastmath = True)
def func2(x):
    return x+1

if __name__ == "__main__":

    a = np.ones(10000000, dtype = np.float64)

    start = timer()
    func(a)
    print("CPU:", round(timer()-start, 1))

    start = timer()
    func2(a)
    cuda.profile_stop()
    print("GPU:", round(timer()-start, 1))