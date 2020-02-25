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
    return np.array([
        [color if el else 'K' for el in row]
        for row in shape
    ]) 

def replace_color(grid, color_source, color_target):
    return np.array([
        [color_target if el==color_source else el for el in row]
        for row in grid  
    ]) 

def sample_xy_003():
    side = 11 + geometric(0.5)
    nb_objs = 2 + geometric(0.5)
    x = Grid(H=side, W=side)
    y = Grid(H=side, W=side)
    color_a = uniform(GENERIC_COLORS)
    color_b = uniform(GENERIC_COLORS)
    internal_assert(color_a!=color_b, 'need distinct colors') 

    for _ in range(nb_objs):
        cell = y.reserve_shape(large_square, spacious=True) 
        x.paint_sprite(monochrome(small_plus, color_a), cell)
        x.paint_sprite(monochrome([[1]], color_b), cell)
        y.paint_sprite(monochrome(large_plus, color_a), cell)
        y.paint_sprite(monochrome(large_times, color_b), cell)
    return x.colors, y.colors

def sample_xy_006():
    side = 3 + geometric(0.1) 
    x = Grid(H=side, W=side)
    y = Grid(H=side, W=side)
    color_a = uniform(GENERIC_COLORS)
    color_b = uniform(GENERIC_COLORS)
    for r in range(side):
        if bernoulli(0.5):
            color = uniform(GENERIC_COLORS)
            internal_assert(len({color, color_a, color_b})==3, 'need distinct colors') 
            x.colors[r,:] = np.array([color]*3)
            y.colors[r,:] = np.array(['A']*3)
        else:
            colors = [uniform([color_a, color_b]) for c in range(side)] 
            internal_assert(len(set(colors))!=1, 'need polychromatic row') 
            x.colors[r,:] = np.array(colors)
    return x.colors, y.colors

def sample_xy_007():
    side = 10 + geometric(0.1) 
    nb_objs = 3 + geometric(0.5) 
    z = Grid(H=side, W=side)
    shape_big = gen_shape(None, side=4)
    shape_small = gen_shape(None, side=4)
    internal_assert(np.sum(shape_small) < np.sum(shape_big), 'shapes need to differ in size')
    for _ in range(nb_objs-1):
        cell = z.reserve_shape(shape_big, spacious=True) 
        z.paint_sprite(monochrome(shape_big, 'B'), cell)
    cell = z.reserve_shape(shape_small, spacious=True) 
    z.paint_sprite(monochrome(shape_small, 'R'), cell)

    y = np.copy(z.colors)
    x = np.copy(z.colors)
    x = replace_color(x, 'R', 'C')
    x = replace_color(x, 'B', 'C')
    return x, y

def sample_xy_008():
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
            cell = z.reserve_shape(shape, spacious=True) 
            z.paint_sprite(monochrome(shape, color), cell)
        if multiple==max_mult:
            y = monochrome(shape, color)
    x = np.copy(z.colors)
    return x, y

def sample_xy_016():
    height = 8 + geometric(1.5) 
    width = 8 + geometric(1.5) 
    z = Grid(H=height, W=width)

    nb_shapes = 2 + geometric(1.0) 
    shapes = [gen_shape(None, side=2+bernoulli(0.8)+bernoulli(0.2)) for _ in range(nb_shapes)]
    for shape in shapes:
        cell = z.reserve_shape(shape, spacious=True) 
        z.paint_sprite(monochrome(shape, 'C'), cell)
    x = np.copy(z.colors)
    
    y = Grid(nb_shapes, nb_shapes)
    for i in range(nb_shapes):
        y.colors[i,i]='C'
    return x, y.colors

def sample_xy_022():
    height = 8 + geometric(1.0) 
    width = 8 + geometric(1.0) 
    z = Grid(H=height, W=width)

    nb_rows = 1+geometric(2.0)
    cells = [(uniform(height), uniform(width)) for _ in range(nb_rows)] 
    internal_assert(len(set(c[0] for c in cells))==nb_rows, 'rows should be distinct')
    colors = [uniform(GENERIC_COLORS) for _ in range(nb_rows)]

    for cell, color in zip(cells, colors):
        z.colors[cell] = color
    x = np.copy(z.colors)

    for (r,c), color in zip(cells, colors):
        while c<width:
            if c>=width: break
            z.colors[r,c] = color
            c = c+1    
            if c>=width: break
            z.colors[r,c] = 'A'
            c = c+1
    y = z.colors

    return x, y

def sample_xy_023():
    height = 12 + geometric(2.0) 
    width = 12 + geometric(2.0) 
    z = Grid(H=height, W=width)
    z.fill('N', (0,0))

    colors = []
    nb_objs = 2+geometric(1.0) 
    for _ in range(nb_objs):
        side = 3+geometric(1.0)
        shape = gen_shape(None, side, crop=False)
        cell = z.reserve_shape(shape, spacious=True)
        SG = ShapeGen()
        SG.side=side
        color = 'C' if not SG.check_top_req(shape, 'simpl') else 'B'
        colors.append(color)
        z.paint_sprite(monochrome(shape, color), cell)
    internal_assert('C' in colors, 'need at least one cyan hole')

    y = np.copy(z.colors)
    x = replace_color(y, 'C', 'B')
    return x,y

def sample_xy_032():
    small_side = 2
    big_side = 2*small_side+1
    z = Grid(H=big_side, W=big_side)

    color = uniform(GENERIC_COLORS)
    shape_a = gen_shape(None, side=2, crop=False)
    shape_b = gen_shape(None, side=2, crop=False)
    internal_assert(np.sum(np.abs(shape_a-shape_b))!=0, 'shapes should be distinct')

    z.paint_sprite(monochrome(shape_a, color), (1,1))
    z.paint_sprite(monochrome(shape_a, color), (1,4))
    z.paint_sprite(monochrome(shape_a, color), (4,1))
    z.paint_sprite(monochrome(shape_b, color), (4,4))

    x = z.colors
    y = monochrome(shape_b, color)

    rotations = uniform(4)
    x = np.rot90(x, rotations)
    y = np.rot90(y, rotations)
    return x,y

def sample_xy_034():
    side = 7 + geometric(1.0)
    z = Grid(H=side, W=side)
    shape = gen_shape(None, side=3)  
    color = uniform(GENERIC_COLORS)
    cell = z.reserve_shape(shape)
    z.paint_sprite(monochrome(shape, color), cell)
    x = z.colors

    h, w = shape.shape
    y = Grid(H=h, W=2*w)
    y.paint_sprite(monochrome(shape, color), (h//2,w//2))
    y.paint_sprite(monochrome(shape, color), (h//2,w+w//2))
    return x,y.colors

def sample_xy_037():
    side = 3 + geometric(0.1)
    z = Grid(H=side, W=side)
    z.fill('A', (0,0))
    z.noise(uniform(GENERIC_COLORS), density=0.5)
    z.noise(uniform(GENERIC_COLORS), density=0.5)
    z.noise(uniform(GENERIC_COLORS), density=0.5)
    x = z.colors
    y = np.transpose(z.colors)
    return x,y

def tenacious_gen(f, nb_iters=100):
    for _ in range(nb_iters):
        try:
            return f()
        except InternalError:
            continue

if __name__=='__main__':
    while True:
        x,y = tenacious_gen(sample_xy_037)
        print(CC+str_from_grids([x, y], render_color))
        input('next?')
