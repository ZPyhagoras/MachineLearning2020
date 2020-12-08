import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

dataset_path = 'C:\\Users\\DWQE\\LinuxFiles\\MachineLearning2020\\PersonalProject\\Dataset\\'


def data_process(df_name, is_train=True):
    pos_list = ['rw', 'rb', 'st', 'lw', 'cf', 'cam', 'cm', 'cdm', 'cb', 'lb', 'gk']
    pos_cnt = 3

    def date2age(ddf):
        year = int(ddf['birth_date'].split('/')[-1])
        return 2020 - (year + 1900 if year != 0 else year + 2000)

    def level2int(ddf, attribute):
        levels = {'Low': 1, 'Medium': 2, 'High': 3}
        return levels[ddf[attribute]]

    def df_ln(ddf, attribute):
        return math.log(ddf[attribute])

    def max_val(ddf):
        return sum(sorted(list(ddf[pos_list]))[::-1][:pos_cnt]) / pos_cnt

    df = pd.read_csv(dataset_path + df_name + '.csv').fillna(0)
    df['birth_date'] = df.apply(date2age, axis=1)
    df['work_rate_att'] = df.apply(level2int, args=('work_rate_att',), axis=1)
    df['work_rate_def'] = df.apply(level2int, args=('work_rate_def',), axis=1)
    df['gk'] = df['gk'] * 4
    df['max_val'] = df.apply(max_val, axis=1)
    if is_train:
        pos_list = pos_list + ['y']
        df['ln_y'] = df.apply(df_ln, args=('y',), axis=1)

    df = df.drop(pos_list, axis=1)
    df.to_csv(dataset_path + 'Processed' + df_name + '.csv', index=False)


data_process('train')
data_process('test', is_train=False)
