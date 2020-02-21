''' author: samtenka
    change: 2020-02-20
    create: 2019-02-20
    descrp:
    to use: 
'''

from utils import CC, pre                               # ansi
from utils import secs_endured, megs_alloced            # profiling
from utils import reseed, bernoulli, geometric, uniform # math
from collections import namedtuple

import numpy as np

from shape import ShapeGen

SPECIAL_COLORS = 'KA'
GENERIC_COLORS = 'BRGYPOCN'
COLORS = SPECIAL_COLORS + GENERIC_COLORS

Block = namedtuple('Block', ['shape', 'color']) 

