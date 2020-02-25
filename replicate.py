''' author: samtenka
    change: 2020-02-24
    create: 2020-02-24
    descrp: replicate first few tasks
    to use:
'''

from utils import CC, pre                               # ansi
from utils import secs_endured, megs_alloced            # profiling
from utils import reseed, bernoulli, geometric, uniform # math

from lg_types import tInt, tCell, tColor, tShape, tBlock, tGrid, tDir, tNoise
from lg_types import tCount_, tFilter_, tArgmax_, tMap_, tRepeat_

from shape import ShapeGen 
from block import GENERIC_COLORS, Block, block_equals
from grid import Grid

import numpy as np
from vis import str_from_grids, render_color

small_plus = np.array([
    [0,1,0],
    [1,1,1],
    [0,1,0],
]) 
small_times = np.array([
    [1,0,1],
    [0,1,0],
    [1,1,1],
]) 
large_plus = np.array([
    [0,0,1,0,0],
    [0,0,1,0,0],
    [1,1,1,1,1],
    [0,0,1,0,0],
    [0,0,1,0,0],
]) 
large_times = np.array([
    [1,0,0,0,1],
    [0,1,0,1,0],
    [0,0,1,0,0],
    [0,1,0,1,0],
    [1,0,0,0,1],
])
large_square = np.array([
    [1,1,1,1,1],
    [1,1,1,1,1],
    [1,1,1,1,1],
    [1,1,1,1,1],
    [1,1,1,1,1],
])

def monochrome(shape, color):
    return np.array([
        [color if el else 'K' for el in row]
        for row in shape
    ]) 

def sample_xy_003():
    side = 11 + geometric(0.5)
    nb_objs = 2 + geometric(0.25)
    x = Grid(H=side, W=side)
    y = Grid(H=side, W=side)
    color_a = uniform(GENERIC_COLORS)
    color_b = uniform(GENERIC_COLORS)
    assert color_a!=color_b

    for _ in range(nb_objs):
        cell = y.reserve_shape(large_square) 
        x.paint_sprite(monochrome(small_plus, color_a), cell)
        x.paint_sprite(monochrome([[1]], color_b), cell)
        y.paint_sprite(monochrome(large_plus, color_a), cell)
        y.paint_sprite(monochrome(large_times, color_b), cell)
    return x.colors, y.colors

def tenacious_gen(f, nb_iters=100):
    for _ in range(nb_iters):
        try:
            return f()
        except:
            continue

if __name__=='__main__':
    while True:
        x,y = tenacious_gen(sample_xy_003)
        print(CC+str_from_grids([x, y], render_color))
        input('next?')
