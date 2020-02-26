''' author: samtenka
    change: 2020-02-24
    create: 2020-02-24
    descrp: replicate first few tasks
    to use:
'''

from utils import InternalError, internal_assert        # maybe
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

def gen_shape(noise, side, crop=True):
    internal_assert(1<=side<=6, 'requested shape sidelength illegal')
    SG = ShapeGen()
    SG.set_side(side)
    shape = SG.search(crop) 
    return shape

def monochrome(shape, color):
    G = Grid(H=shape.shape[0], W=shape.shape[1])
    G.colors = np.array([
        [color if el else 'K' for el in row]
        for row in shape
    ]) 
    return G

def center(shape):
    return (shape.shape[0]//2, shape.shape[1]//2)

def volume(shape):
    return np.sum(shape)
def shape_eq(lhs, rhs):
    return (
        lhs.shape==rhs.shape
        and np.sum(np.abs(lhs-rhs))==0
    )

def displace(cell, offset):
    return tuple(cell[i]+offset[i] for i in range(2))

def split_int(v, f): return f(v)
def split_grid(v, f): return f(v)
def split_color(v, f): return f(v)
def split_cell(v, f): return f(v)
def split_ptdgrid(v, f): return f(v)
def pair(x,y): return (x,y)
def fst(z): return z[0]
def snd(z): return z[1]
def repeat(n,i,f):
    for _ in range(n):
        i = f(i)
    return i

def reserve_shape(grid, shape, cell_in_shape):
    new_grid = grid.copy()
    cell = new_grid.reserve_shape(shape, cell_in_shape, spacious=True)
    return (new_grid, cell) 
def paint_sprite(field, sprite, cell_in_field, cell_in_sprite):
    new_grid = field.copy()
    new_grid.paint_sprite(sprite, cell_in_field, cell_in_sprite)
    return new_grid
def paint_cell(field, cell_in_field, color):
    new_grid = field.copy()
    new_grid.paint_cell(cell_in_field, color)
    return new_grid

def sample_003_functional():
    return (
    split_int(11+geometric(0.5), lambda side:
    split_int(2+geometric(0.5), lambda nb_objs:
    split_grid(Grid(side,side), lambda blank:
    split_color(uniform(GENERIC_COLORS), lambda color_a:
    split_color(uniform(GENERIC_COLORS), lambda color_b:
    repeat(nb_objs,
        pair(blank,blank),
        lambda xy: ( 
            split_ptdgrid(reserve_shape(snd(xy), large_square, center(large_square)), lambda ptdgrid:
            pair(
                split_grid(paint_sprite(fst(xy), monochrome(small_plus, color_a), snd(ptdgrid), center(small_plus)), lambda half_paint:  
                    paint_cell(half_paint, snd(ptdgrid), color_b)
                ),
                split_grid(paint_sprite(fst(ptdgrid), monochrome(large_plus, color_a), snd(ptdgrid), center(large_plus)), lambda half_paint: 
                    paint_sprite(half_paint, monochrome(large_times, color_b), snd(ptdgrid), center(large_times))
                )
            ))
        )
    )))))))

def sample_003():
    side = 11 + geometric(0.5)
    nb_objs = 2 + geometric(0.5)
    x = Grid(H=side, W=side)
    y = Grid(H=side, W=side)
    color_a = uniform(GENERIC_COLORS)
    color_b = uniform(GENERIC_COLORS)
    internal_assert(color_a!=color_b, 'need distinct colors') 

    for _ in range(nb_objs):
        cell = y.reserve_shape(large_square, center(large_square), spacious=True) 
        x.paint_sprite(monochrome(small_plus, color_a), cell, center(small_plus))
        x.paint_cell(cell, color_b)
        y.paint_sprite(monochrome(large_plus, color_a), cell, center(large_plus))
        y.paint_sprite(monochrome(large_times, color_b), cell, center(large_times))
    return x, y

def sample_006():
    side = 3 + geometric(0.1) 
    x = Grid(H=side, W=side)
    y = Grid(H=side, W=side)
    color_a = uniform(GENERIC_COLORS)
    color_b = uniform(GENERIC_COLORS)
    for r in range(side):
        if bernoulli(0.5):
            color = uniform(GENERIC_COLORS)
            internal_assert(len({color, color_a, color_b})==3, 'need distinct colors') 
            x.paint_row(r, color)
            y.paint_row(r, 'A')
        else:
            colors = [uniform([color_a, color_b]) for c in range(side)] 
            internal_assert(len(set(colors))!=1, 'need polychromatic row') 
            x.colors[r,:] = np.array(colors)
    return x, y

def sample_007():
    side = 10 + geometric(0.1) 
    nb_objs = 3 + geometric(0.5) 
    z = Grid(H=side, W=side)
    shape_big = gen_shape(None, side=4)
    shape_small = gen_shape(None, side=4)
    internal_assert(volume(shape_small) < volume(shape_big), 'shapes need to differ in size')
    for _ in range(nb_objs-1):
        cell = z.reserve_shape(shape_big, (0,0), spacious=True) 
        z.paint_sprite(monochrome(shape_big, 'B'), cell, (0,0))
    cell = z.reserve_shape(shape_small, (0,0), spacious=True) 
    z.paint_sprite(monochrome(shape_small, 'R'), cell, (0,0))

    y = z.copy()
    x = z.copy()
    x.replace_color('R', 'C')
    x.replace_color('B', 'C')
    return x, y

def sample_008():
    side = 13 + geometric(1.0) 
    nb_shapes = 2 + geometric(0.5) 
    shapes = [gen_shape(None, side=3) for _ in range(nb_shapes)]
    internal_assert(
        len(set(map(lambda row:','.join(map(''.join, map(str,row))), shapes)))==nb_shapes,
        'shapes should be distinct'
    )
    multiples = [1 + geometric(1.0) for _ in range(nb_shapes)] 
    max_mult = max(multiples)
    internal_assert(
        len([m for m in multiples if m==max_mult])==1,
        'plurality should be unique'
    )
    colors = [uniform(GENERIC_COLORS) for _ in range(nb_shapes)]
    z = Grid(H=side, W=side)
    for shape, color, multiple in zip(shapes, colors, multiples):
        for _ in range(multiple):
            cell = z.reserve_shape(shape, (0,0), spacious=True) 
            z.paint_sprite(monochrome(shape, color), cell, (0,0))
        if multiple==max_mult:
            y = monochrome(shape, color)
    x = z.copy()
    return x, y

def sample_016():
    height = 8 + geometric(1.5) 
    width = 8 + geometric(1.5) 
    z = Grid(H=height, W=width)

    nb_shapes = 2 + geometric(1.0) 
    shapes = [gen_shape(None, side=2+bernoulli(0.8)+bernoulli(0.2)) for _ in range(nb_shapes)]
    for shape in shapes:
        cell = z.reserve_shape(shape, (0,0), spacious=True) 
        z.paint_sprite(monochrome(shape, 'C'), cell, (0,0))
    x = z.copy()
    
    y = Grid(nb_shapes, nb_shapes)
    cell = y.get_border_cell((0,0), (-1,-1))
    while y.cell_in_bounds(cell): 
        y.paint_cell(cell, 'C')
        cell = displace(cell, (1,1))
    return x, y

def sample_022():
    height = 8 + geometric(1.0) 
    width = 8 + geometric(1.0) 
    z = Grid(H=height, W=width)

    nb_rows = 1+geometric(2.0)
    cells = [(uniform(height), uniform(width)) for _ in range(nb_rows)] 
    internal_assert(len(set(c[0] for c in cells))==nb_rows, 'rows should be distinct')
    colors = [uniform(GENERIC_COLORS) for _ in range(nb_rows)]

    for cell, color in zip(cells, colors):
        z.colors[cell] = color
    x = z.copy()

    for cell, color in zip(cells, colors):
        tile = Grid(H=1,W=2)
        tile.fill('A', (0,0))
        tile.paint_cell((0,0), color)
        while z.cell_in_bounds(cell):
            z.paint_sprite(tile, cell, sprite_cell=(0,0))
            cell = displace(cell, (0,2))
    y = z.copy()
    return x, y

def sample_023():
    height = 12 + geometric(2.0) 
    width = 12 + geometric(2.0) 
    z = Grid(H=height, W=width)
    z.fill('N', (0,0))

    colors = []
    nb_objs = 2+geometric(1.0) 
    for _ in range(nb_objs):
        side = 3+geometric(1.0)
        shape = gen_shape(None, side, crop=False)
        cell = z.reserve_shape(shape, (0,0), spacious=True)
        SG = ShapeGen()
        SG.side=side
        color = 'C' if not SG.check_top_req(shape, 'simpl') else 'B'
        colors.append(color)
        z.paint_sprite(monochrome(shape, color), cell, (0,0))
    internal_assert('C' in colors, 'need at least one cyan hole')

    y = z.copy()
    x = replace_color(y, 'C', 'B')
    return x,y

def sample_032():
    small_side = 2
    big_side = 2*small_side+1
    z = Grid(H=big_side, W=big_side)

    color = uniform(GENERIC_COLORS)
    shape_a = gen_shape(None, side=2, crop=False)
    shape_b = gen_shape(None, side=2, crop=False)
    internal_assert(not shape_eq(shape_a, shape_b), 'shapes should be distinct')

    z.paint_sprite(monochrome(shape_a, color), (1,1), (0,0))
    z.paint_sprite(monochrome(shape_a, color), (1,4), (0,0))
    z.paint_sprite(monochrome(shape_a, color), (4,1), (0,0))
    z.paint_sprite(monochrome(shape_b, color), (4,4), (0,0))

    x = z.copy()
    y = monochrome(shape_b, color)

    rotations = uniform(4)
    x.rotate(rotations)
    x.rotate(rotations)
    return x,y

def sample_034():
    side = 7 + geometric(1.0)
    z = Grid(H=side, W=side)
    shape = gen_shape(None, side=3)  
    color = uniform(GENERIC_COLORS)
    cell = z.reserve_shape(shape, (0,0))
    z.paint_sprite(monochrome(shape, color), cell, (0,0))
    x = z.copy()

    h, w = shape.shape
    y = Grid(H=h, W=2*w)
    cell = (0,0)
    while y.cell_in_bounds(cell):
        y.paint_sprite(monochrome(shape, color), cell, (0,0))
        cell = displace(cell, (0,w))
    return x,y

def sample_037():
    side = 3 + geometric(0.1)
    z = Grid(H=side, W=side)
    z.fill('A', (0,0))
    z.noise(uniform(GENERIC_COLORS), density=0.5)
    z.noise(uniform(GENERIC_COLORS), density=0.5)
    z.noise(uniform(GENERIC_COLORS), density=0.5)
    x = z.copy()
    z.reflect((1,1))
    y = z.copy()
    return x,y

def tenacious_gen(f, nb_iters=100):
    for _ in range(nb_iters):
        try:
            return f()
        except InternalError:
            continue

routines = [
    sample_003_functional,
    #sample_006,
    #sample_007,
    #sample_008,
    #sample_016,
    #sample_022,
    #sample_023,
    #sample_032,
    #sample_034,
    #sample_037,
]

if __name__=='__main__':

    while True:
        for sample in routines:
            x,y = tenacious_gen(sample)
            print(CC+str_from_grids([x.colors, y.colors], render_color))
            input('next?')
