'''
'''

import os
import numpy as np
from utils import reseed, paths, pre

SAMPLES_FILE_NM = paths('train_list')
HARDNESS_FILE_NM = paths('hardness')
NB_SAMPLES = 100

def get_grids(i, dset='train'):
    with open(SAMPLES_FILE_NM) as f:
        lines = f.read().split('\n')
        file_nm = lines[i]
    with open(file_nm) as f:
        json = eval(f.read())
    return [
        tuple(np.array(pair[role]) for role in ('input', 'output'))
        for pair in json[dset]
    ]

def get_hardness(i):
    with open(HARDNESS_FILE_NM) as f:
        lines = [ln for ln in f.read().split('\n') if ln.strip()]
        task_nb, hardness = map(int, lines[i].split())
        pre(task_nb==i, 'not found!')
        return hardness

if __name__=='__main__':
    reseed(1729)
    train_file_nms = paths('ARC_train')
    x = np.random.choice(train_file_nms, NB_SAMPLES, replace=False)
    with open(SAMPLES_FILE_NM, 'w') as f:
        f.write('\n'.join(x))
