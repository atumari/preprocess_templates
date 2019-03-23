import numpy as np
import pandas as pd 
import feather
from tqdm import tqdm
import numba


data_name = ""
save_name = ""


@numba.jit('void(f8[:, :], f8[:, :])')
def task(data, res):
    # ここに処理を追加
    M = data.shape[0]
    N = data.shape[1]
    for i in range(M):
        for j in range(M):
            d = 0.0
            for k in range(N):
                tmp = data[i, k] - data[j, k]
                d += tmp * tmp
            res[i, j] = np.sqrt(d)


if __name__ == "__main__":
    #data = feather.read_dataframe(data_name).values
    data = np.random.rand(10000, 3)
    res = np.zeros((10000, 10000), dtype=np.float64)
    task(data, res)
    print(res[-5:, -5:])

    #features = pd.DataFrame(res, columns=[])
    #print(features.tail())
    #features.to_feather(save_name)