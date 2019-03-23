import numpy as np
import pandas as pd 
import feather
from tqdm import tqdm
import multiprocessing


data_name = ""
save_name = ""
num_worker = 8


def task(data):
    # ここに処理を追加
    res = 0
    for i in data:
        res += i
    return [res]


if __name__ == "__main__":
    #data = feather.read_dataframe(data_name).values
    data = np.arange(100000 * 10000, dtype=np.int64).reshape(100000, 10000)
    pool = multiprocessing.Pool(processes=num_worker)
    it = pool.imap_unordered(task, data)
    results = []
    for res in tqdm(it, total=len(data)):
        results.append(res)
    pool.close()

    features = pd.DataFrame(results, columns=["sum"])
    print(features.tail())
    #features.to_feather(save_name)