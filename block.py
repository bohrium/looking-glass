''' author: samtenka
    change: 2020-02-20
    create: 2019-02-20
    descrp:
    to use: 
'''

from utils import CC, pre                               # ansi
from utils import secs_endured, megs_alloced            # profiling
from utils import reseed, bernoulli, geometric, uniform # math

import numpy as np

from shape import ShapeGen

from collections import namedtuple
Block = namedtuple('Block', ['shape', 'color']) 

SPECIAL_COLORS = 'KA'
GENERIC_COLORS = 'BRGYPOCN'
COLORS = SPECIAL_COLORS + GENERIC_COLORS

def block_equals(lhs, rhs):
    return (
            lhs.color==rhs.color
        and lhs.shape.shape==rhs.shape.shape
        and np.sum(np.abs(lhs.shape-rhs.shape))==0
    )
