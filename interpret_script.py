''' author: samtenka
    change: 2020-02-20
    create: 2020-02-17
    descrp: implement primitives
    to use:
'''

from utils import CC, pre                               # ansi
from utils import secs_endured, megs_alloced            # profiling
from utils import reseed, bernoulli, geometric, uniform # math

from lg_types import tInt, tCell, tColor, tShape, tBlock, tGrid, tDir, tNoise
from lg_types import tCount_, tFilter_, tArgmax_, tMap_, tRepeat_

from shape import ShapeGen 
from scene import Scene # aka grid
from block import GENERIC_COLORS, Block
from grid import Grid

import numpy as np

#=============================================================================#
#=====  0. PRIMITIVES  =======================================================#
#=============================================================================#

class InternalError(Exception):
    def __init__(self, msg):
        self.msg = msg
def internal_assert(condition, message):
    if not condition:
        raise InternalError(message)

def gen_cell(noise, grid):
    return (uniform(grid.H),
            uniform(grid.W))

SG = ShapeGen()
def gen_shape(noise, side):
    global SG
    internal_assert(1<=side<=6, 'requested shape sidelength illegal')
    SG.set_side(side)
    shape = SG.search() 
    return shape

def amax(items, score_func):
    internal_assert(items, 'argmax fails existence')
    scored = [(score_func(x), i) for i, x in enumerate(items)] 
    best_score = max(scored)[0] 
    best_items = [items[i] for s,i in scored if s==best_score]
    internal_assert(len(best_items)==1, 'argmax fails uniqueness')
    return best_items[0]

def render_blocks(grid, blocks, nb_tries=5):
    for _ in range(nb_tries):
        new_grid = grid.copy()
        success = new_grid.render_blocks(blocks)
        if success is not None: return new_grid
    raise InternalError('unable to render blocks')

def render_block_in_corner(grid, block):
    G = grid.copy()
    G.place_block(block.shape, 0, 0, block.color, block_id=9)
    return G

def gen_blocks(nb, block_gen, noise):
    blocks = [block_gen(noise) for _ in range(nb)]
    internal_assert(len(blocks)<=6, 'too many blocks')
    return blocks

def blank_grid(height, width):
    internal_assert(1<=height<=30 and 1<=width<=30, 'illegal grid shape' )
    return Grid(height, width)

impls_by_nm = {
    # basic samplers:
    'gen_some': lambda noise: uniform([1,2,3]),
    'gen_svrl': lambda noise: uniform([3,4,5]),
    'gen_many': lambda noise: uniform(range(6,20)),
    'gen_cell': lambda noise: lambda grid: gen_cell(noise, grid),
    'gray'    : 'A',
    'gen_rain': lambda noise: uniform(GENERIC_COLORS),
    'gen_shap': lambda noise: lambda side: gen_shape(noise, side),
    # product types:
    'blok_cons': lambda color: lambda shape: Block(shape, color),
    'shap_blok': lambda block: block.shape,
    'colr_blok': lambda block: block.color,
    # render a grid:
    'fill_grd': (
        lambda grid: lambda color: lambda cell_pos:
            grid.copy().fill(color, cell_pos)
    ),
    'noise_grd': (
        lambda grid: lambda noise: lambda color: grid.copy().noise([color])
    ),
    'blnk_grd': lambda height: lambda width: blank_grid(height, width),
    'rndr_blks': (
        lambda noise: lambda grid: lambda blocks: render_blocks(grid, blocks)
    ),
    'rndr_blk': (
        lambda grid: lambda block: render_block_in_corner(grid, block)
    ),
    # numerical concepts:
    'volume_shap': lambda shape: np.sum(shape),
    'height_shap': lambda shape: shape.shape[0],
    'width_shap':  lambda shape: shape.shape[1],
    'amax_blocks': lambda blocks: lambda score: amax(blocks, score),
    # list helpers:
    'gen_blks': (
        lambda nb: lambda block_gen: lambda noise: 
            gen_blocks(nb, block_gen, noise)
    ),
    'cons_blks': lambda blocks: lambda block: blocks+[block],
}

#=============================================================================#
#=====  1. LAMBDA EVALUATION  ================================================#
#=============================================================================#

if __name__=='__main__':
    from generate_script import get_script
    while True:
        while True:
            print()
            blocks, get_block, grid, X, Y = get_script()
            blocks_impl = eval(blocks['pyth'], impls_by_nm)
            get_block_impl = eval(get_block['pyth'], impls_by_nm)
            grid_impl = eval(grid['pyth'], impls_by_nm)
            X_impl = eval(X['pyth'], impls_by_nm)
            Y_impl = eval(Y['pyth'], impls_by_nm)
            if ('rndr' in grid['pyth'] and
                X['text']!=Y['text']): break

        try:
            noise = None
            blocks = blocks_impl(noise)
            block = get_block_impl(blocks)
            grid = grid_impl(noise)(blocks)(block)
            X = X_impl(noise)(block)(grid)
            Y = Y_impl(block)(grid)
        except InternalError: continue
        break
    print(str(X))
    print(str(Y))

