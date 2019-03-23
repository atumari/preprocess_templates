import os
# set env for numba.cuda
os.environ['NUMBAPRO_NVVM']='C:\Program Files\\NVIDIA GPU Computing Toolkit\CUDA\\v9.0\\nvvm\\bin\\nvvm64_32_0.dll' 
os.environ['NUMBAPRO_LIBDEVICE']='C:\Program Files\\NVIDIA GPU Computing Toolkit\CUDA\\v9.0\\nvvm\libdevice'

import numpy as np
import pandas as pd 
import feather
from tqdm import tqdm
import math

import numba
from numba import cuda


data_name = ""
save_name = ""


@cuda.jit('void(f8[:], f8[:])')
def task_1d(data, res):
    # ここに処理を追加
    i = cuda.grid(1)
    cuda.atomic.max(res, 0, data[i])


@cuda.jit('void(f8[:, :], f8[:, :])')
def task_2d(data, res):
    # ここに処理を追加
    M = data.shape[0]
    N = data.shape[1]

    i, j = cuda.grid(2)

    if i < M and j < M:
        d = 0.0
        for k in range(N):
            tmp = data[i, k] - data[j, k]
            d += tmp * tmp
            
        res[i, j] = math.sqrt(d)


if __name__ == "__main__":
    #data = feather.read_dataframe(data_name).values

    # 1d
    data = np.random.rand(100)
    res = np.zeros(1, dtype=np.float64)
    threadsperblock = 512
    blockspergrid = (data.shape[0] + (threadsperblock - 1)) // threadsperblock
    task_1d[blockspergrid, threadsperblock](data, res)
    print(res)
    print(max(data))

    # 2d
    data = np.random.rand(10000, 3)
    res = np.zeros((10000, 10000), dtype=np.float64)
    threadsperblock = (16, 16)
    blockspergrid_x = math.ceil(data.shape[0] / threadsperblock[0])
    #blockspergrid_y = math.ceil(data.shape[1] / threadsperblock[1])
    blockspergrid_y = math.ceil(data.shape[0] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    task_2d[blockspergrid, threadsperblock](data, res)
    print(res[-5:, -5:])

    #features = pd.DataFrame(res, columns=[])
    #print(features.tail())
    #features.to_feather(save_name)