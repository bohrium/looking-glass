''' author: samtenka
    change: 2020-03-17
    create: 2019-06-12
    descrp: Helpers for ANSI screen coloration, resource profiling, math,
            maybe types, and project paths.
    to use: Import:
                from utils import CC, pre, status                   # ansi
                from utils import secs_endured, megs_alloced        # profiling
                from utils import reseed, geometric, bernoulli      # math
                from utils import InternalError, internal_assert    # maybe
                from utils import path                              # paths 
            Or, run as is to see a pretty rainbow:
                python utils.py
'''

import functools
import time
import sys
import random
import numpy as np
import glob

#=============================================================================#
#=====  0. ANSI CONTROL FOR RICH OUTPUT TEXT =================================#
#=============================================================================#

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~  0.0. Define Text Modifier  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

class Colorizer(object):
    '''
        Text modifier class, used as in 
            print(CC+'@R i am red @D ')
        where CC is an instance of this class. 
    '''
    def __init__(self):

        #-------------  0.0.0. ANSI command abbreviations  -------------------#

        self.ANSI_by_name = {
            '@^ ': '\033[1A',                 # motion: up

            '@K ': '\033[38;2;000;000;000m',  # color: black
            '@A ': '\033[38;2;128;128;128m',  # color: gray  
            '@W ': '\033[38;2;255;255;255m',  # color: white

            '@R ': '\033[38;2;240;032;032m',  # color: red
            '@O ': '\033[38;2;224;128;000m',  # color: orange 
            '@Y ': '\033[38;2;255;224;000m',  # color: yellow

            '@G ': '\033[38;2;064;224;000m',  # color: green
            '@F ': '\033[38;2;000;224;000m',  # color: forest
            '@C ': '\033[38;2;000;192;192m',  # color: cyan

            '@B ': '\033[38;2;096;064;255m',  # color: blue
            '@P ': '\033[38;2;192;000;192m',  # color: purple  
            '@N ': '\033[38;2;128;032;000m',  # color: brown
        }

        #-------------  0.0.1. default color is cyan  ------------------------#

        self.ANSI_by_name['@D '] = self.ANSI_by_name['@C ']

        self.text = ''

    #-----------------  0.0.2. define application to strings  ----------------#

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

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~  0.1. Global Initializations  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

CC = Colorizer()
print(CC+'@D @^ ')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~  0.2. Styles for Special Message Types  ~~~~~~~~~~~~~~~~~~~~~~~#

def pre(condition, message): 
    ''' assert precondition; if fail, complain in red '''
    assert condition, CC+'@R '+message+'@D '

def status(message, **kwargs): 
    ''' print in orange, with bracketed info in purple '''
    if 'mood' not in kwargs:
        kwargs['mood'] = 'desert' 
    pre(kwargs['mood'] in {'desert', 'forest', 'sea'}, 'unknown mood!')
    back, fore = {
        'desert': ('@O', '@P'),
        'forest': ('@G', '@N'),
        'sea'   : ('@C', '@B'),
    }[kwargs['mood']]
    message = (
        message
        .replace('[', '[{} '.format(fore))
        .replace(']', '{} ]'.format(back))
    )
    del kwargs['mood']
    print(CC+'{} {}@D '.format(back, message), **kwargs)

#=============================================================================#
#=====  1. RESOURCE PROFILING  ===============================================#
#=============================================================================#

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~  1.0. Memory Profiling  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

#---------------------  1.0.0. check memory profiler  ------------------------#

try:
    import memory_profiler
except ImportError:
    status('failed attempt to import [memory_profiler]')

#---------------------  1.0.1. set memory profiler  --------------------------#

megs_alloced = None if 'memory_profile' not in sys.modules else lambda: (
    memory_profiler.memory_usage(
        -1, interval=0.001, timeout=0.0011
    )[0]
)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~  1.1. Time Profiling  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

start_time = time.time()
secs_endured = lambda: (time.time()-start_time) 

#=============================================================================#
#=====  2. MATH and RANDOMNESS  ==============================================#
#=============================================================================#

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~  2.0. Random Seed and Samplers  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

def reseed(s):
    random.seed(s)
    np.random.seed(s)

def bernoulli(p):
    return np.random.binomial(1, p)

def binomial(n, p):
    return np.random.binomial(n, p)

def uniform(n):
    if type(n) in [int, np.int64]:
        return np.random.randint(n)
    elif type(n)==float:
        return n * np.random.random() 
    else:
        return random.choice(n)

def geometric(scale):
    ''' Support on the (non-negative) naturals, with mean specified by `scale` 
    '''
    return np.random.geometric(1.0/(1.0 + scale)) - 1

def poisson(l):
    return np.random.poisson(l)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~  2.1. Log Densities and Distributions  ~~~~~~~~~~~~~~~~~~~~~~~~#

def _log_choice_(n, a):
    return (
          np.sum(np.log(np.arange((n-a)+1, n+1)))
        - np.sum(np.log(np.arange(1, a+1)))
    )

def log_binom_dist(n_and_p, obs):
    ''' return log ( (n choose obs) p^(obs) (1-p)^(n-obs)  ) '''
    n, p = n_and_p
    return (
          _log_choice_(n, obs)
        +      obs * np.log(    p)
        + (n - obs)* np.log(1.0-p)
    )

def log_poisson(l, k):
    ''' return log ( ( l^k / k! ) / e^l ) '''
    return (
        k * np.log(l)
        -np.sum(np.log(np.arange(1, k+1)))
        -l
    )

#=============================================================================#
#=====  3. SIMULATE MAYBE TYPES VIA EXCEPTIONS  ==============================#
#=============================================================================#

class InternalError(Exception):
    def __init__(self, msg):
        self.msg = msg

def internal_assert(condition, message):
    if not condition:
        raise InternalError(message)

#=============================================================================#
#=====  4. USEFUL PATHS  =====================================================#
#=============================================================================#

with open('paths.json') as f:
    paths_by_descrp = eval(f.read())

def paths(descrp):
    pattern = paths_by_descrp[descrp]
    return sorted(glob.glob(pattern)) if '*' in pattern else pattern

#=============================================================================#
#=====  5. ILLUSTRATE UTILITIES  =============================================#
#=============================================================================#

if __name__=='__main__':

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #~~~~~~~~~  5.0. Display a Rainbow  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    print(CC + '@D moo')

    print(CC + '@W moo')
    print(CC + '@A moo')
    print(CC + '@K moo')

    print(CC + '@R moo')
    print(CC + '@O moo')
    print(CC + '@Y moo')
    print(CC + '@G moo')
    print(CC + '@F moo')
    print(CC + '@C moo')
    print(CC + '@B moo')
    print(CC + '@P moo')
    print(CC + '@N moo')
    print(CC + '@R moo')

    print(CC + 'hi @P moo' + 'cow @C \n')

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #~~~~~~~~~  5.1. Sample from Uniform  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    print()
    print(CC + 'unif @G {}@D '.format(uniform(1.0)))
    print(CC + 'unif @G {}@D '.format(uniform(1.0)))
    print(CC + 'unif @G {}@D '.format(uniform(1.0)))
    print(CC + 'unif @G {}@D '.format(uniform(1.0)))
