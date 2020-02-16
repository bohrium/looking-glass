''' author: samtenka
    change: 2020-02-15
    create: 2019-06-12
    descrp: helpers for ansi commands, resource profiling, and math
    to use: import:
                from utils import CC, pre                       # ansi
                from utils import secs_endured, megs_alloced    # profiling
                from utils import prod, reseed, randint         # math
'''

import functools
import time
import sys
import random
import numpy as np

#=============================================================================#
#       0. ANSI COMMANDS                                                      #
#=============================================================================#

#-----------------------------------------------------------------------------#
#               0.0 define ansi command abbreviations                         #
#-----------------------------------------------------------------------------#

class Colorizer(object):
    ''' Text modifier, used as in `print(CC+'@R i am red @D ')`. '''
    def __init__(self):
        self.ANSI_by_name = {
            '@K ': '\033[38;2;000;000;000m',  # color: black
            '@R ': '\033[38;2;255;064;064m',  # color: red
            '@O ': '\033[38;2;255;128;000m',  # color: orange
            '@Y ': '\033[38;2;192;192;000m',  # color: yellow
            '@L ': '\033[38;2;128;255;000m',  # color: lime 
            '@G ': '\033[38;2;064;255;064m',  # color: green
            '@J ': '\033[38;2;000;255;192m',  # color: jade
            '@C ': '\033[38;2;000;192;192m',  # color: cyan
            '@T ': '\033[38;2;000;192;255m',  # color: teal
            '@B ': '\033[38;2;064;064;255m',  # color: blue
            '@P ': '\033[38;2;128;000;255m',  # color: purple  
            '@M ': '\033[38;2;192;000;192m',  # color: magenta
            '@S ': '\033[38;2;255;000;128m',  # color: salmon  
            '@W ': '\033[38;2;255;255;255m',  # color: white
            '@^ ': '\033[1A',                 # motion: up
        }
        self.ANSI_by_name['@D '] = self.ANSI_by_name['@C '] # default is cyan 
        self.text = ''

    def __add__(self, rhs):
        ''' Transition method of type Colorizer -> String -> Colorizer '''
        assert type(rhs) == type(''), 'expected types (Colorizer + string)'
        for name, ansi in self.ANSI_by_name.items():
            rhs = rhs.replace(name, ansi)
        self.text += rhs
        return self

    def __str__(self):
        ''' Emission method of type Colorizer -> String '''
        rtrn = self.text 
        self.text = ''
        return rtrn

#-----------------------------------------------------------------------------#
#               0.1 global initializations                                    #
#-----------------------------------------------------------------------------#

CC = Colorizer()
print(CC+'@D @^ ')

#-----------------------------------------------------------------------------#
#               0.2 helper colorations for errors etc                         #
#-----------------------------------------------------------------------------#

def pre(condition, message): 
    ''' assert precondition; if fail, complain in red '''
    assert condition, CC+'@R '+message+'@D '

#=============================================================================#
#       1. MEMORY MANAGEMENT                                                  #
#=============================================================================#

#-----------------------------------------------------------------------------#
#               1.0 memory and time profiling                                 #
#-----------------------------------------------------------------------------#

try:
    import memory_profiler
except ImportError:
    print(CC + '@R failed attempt to import `memory_profiler` @D ')

start_time = time.time()
secs_endured = lambda: (time.time()-start_time) 
megs_alloced = None if 'memory_profile' not in sys.modules else lambda: (
    memory_profiler.memory_usage(
        -1, interval=0.001, timeout=0.0011
    )[0]
)

#=============================================================================#
#       2. MATH and RANDOMNESS                                                #
#=============================================================================#

prod = lambda seq: functools.reduce(lambda a,b:a*b, seq, 1) 

#-----------------------------------------------------------------------------#
#               2.1 random seed and generators                                #
#-----------------------------------------------------------------------------#

def reseed(s):
    random.seed(s)
    np.random.seed(s)

def bernoulli(p):
    return np.random.binomial(1, p)

def geometric(scale):
    ''' Support on the (non-negative) naturals, with mean specified by `scale` 
    '''
    return np.random.geometric(1.0/(1.0 + scale)) - 1

#=============================================================================#
#       3. ILLUSTRATE UTILITIES                                               #
#=============================================================================#

if __name__=='__main__':

    #-------------------------------------------------------------------------#
    #           3.0 display a rainbow                                         #
    #-------------------------------------------------------------------------#

    print(CC + '@K moo')

    print(CC + '@R moo')
    print(CC + '@O moo')
    print(CC + '@Y moo')
    print(CC + '@L moo')
    print(CC + '@G moo')
    print(CC + '@J moo')
    print(CC + '@C moo')
    print(CC + '@T moo')
    print(CC + '@B moo')
    print(CC + '@P moo')
    print(CC + '@M moo')
    print(CC + '@S moo')
    print(CC + '@R moo')

    print(CC + '@W moo')
    print(CC + '@R moo')
    print(CC + 'hi @M moo' + 'cow @C \n')
