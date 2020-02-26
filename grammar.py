''' author: samtenka
    change: 2020-02-25
    create: 2019-02-25
    descrp:
    to use: 
'''

import numpy as np

from utils import ARC_path
from utils import CC, pre                       # ansi
from utils import secs_endured, megs_alloced    # profiling
from utils import reseed, bernoulli, geometric  # math

from lg_types import tInt, tCell, tColor, tBlock, tGrid 

from collections import namedtuple

class Grammar:
    def __init__(self):
        self.types_by_primitive = {}

    def eligible(goal):
        pass

    def sample_tree(goal):
        pass
