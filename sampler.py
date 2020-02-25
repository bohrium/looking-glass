'''
'''

import os
import numpy as np
from utils import reseed, ARC_path

SAMPLES_FILE_NM = 'samples.txt'
HARDNESS_FILE_NM = 'hardness.txt'
NB_SAMPLES = 50

def get_grids(i, dset='train'):
    with open(SAMPLES_FILE_NM) as f:
        lines = f.read().split('\n')
        file_nm = lines[i]
    with open('{}/data/training/{}'.format(ARC_path, file_nm)) as f:
        json = eval(f.read())
    return [
        tuple(np.array(pair[role]) for role in ('input', 'output'))
        for pair in json[dset]
    ]

def get_hardness(i):
    with open(HARDNESS_FILE_NM) as f:
        lines = f.read().split('\n')
        return int(lines[i].split()[1])

if __name__=='__main__':
    reseed(1729)
    train_files = os.listdir('{}/data/training/'.format(ARC_path))
    x = np.random.choice(train_files, NB_SAMPLES, replace=False)
    with open(SAMPLES_FILE_NM, 'w') as f:
        f.write('\n'.join(x))

