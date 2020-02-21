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
from block import GENERIC_COLORS, Block, block_equals
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

def gen_shape(noise, side):
    internal_assert(1<=side<=6, 'requested shape sidelength illegal')
    SG = ShapeGen()
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

def render_blocks(grid, blocks, nb_tries=50):
    for _ in range(nb_tries):
        new_grid = grid.copy()
        success = new_grid.render_blocks(blocks)
        if success is not None: return new_grid
    print(blocks)
    raise InternalError('unable to render blocks')

def render_block_in_corner(grid, block):
    G = grid.copy()
    G.place_block(block.shape, 0, 0, block.color, block_id=9)
    return G

def gen_blocks(nb, block_gen, noise):
    internal_assert(nb<=6, 'too many blocks')
    blocks = [block_gen(noise) for _ in range(nb)]
    return blocks

def blank_grid(height, width):
    internal_assert(1<=height<=30 and 1<=width<=30, 'illegal grid shape' )
    return Grid(height, width)

def uniq_blocks(blocks):
    uniq = []
    for i in range(len(blocks)):
        for j in range(i):
            if block_equals(blocks[i], blocks[j]): break
        else:
            uniq.append(blocks[i])
    return uniq

impls_by_nm = {
    # basic samplers:
    'gen_some': lambda noise: uniform([1,2,3]),
    'gen_svrl': lambda noise: uniform([3,4,5]),
    'gen_many': lambda noise: uniform(range(15,20)),
    'gen_cell': lambda noise: lambda grid: gen_cell(noise, grid),
    'gray'    : 'A',
    'gen_rain': lambda noise: uniform(GENERIC_COLORS),
    'gen_shap': lambda noise: lambda side: gen_shape(noise, side),
    # product types:
    'blok_cons': lambda color: lambda shape: Block(shape=shape, color=color),
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
    'sing_blks': lambda block: [block],
    'gen_blks': (
        lambda nb: lambda block_gen: lambda noise: 
            gen_blocks(nb, block_gen, noise)
    ),
    'cons_blks': lambda blocks: lambda block: blocks+[block],
    'uniq_blks': lambda blocks: uniq_blocks(blocks),
}

#=============================================================================#
#=====  1. LAMBDA EVALUATION  ================================================#
#=============================================================================#

def concat_multilines(displays):
    lines = [d.split('\n') for d in displays]
    heights = [len(ls) for ls in lines] 
    pre(heights == sorted(heights, reverse=True), '!')
    return '\n'.join(
        ' '.join(
            ls[h] if h<len(ls) else ''
            for ls in lines
        )
        for h in range(max(heights))
    )


if __name__=='__main__':
    from generate_script import get_script

    for _ in range(12):
        pairs = []
        print()
        blocks, get_block, X, Y = get_script()
        blocks_impl = eval(blocks['pyth'], impls_by_nm)
        get_block_impl = eval(get_block['pyth'], impls_by_nm)
        X_impl = eval(X['pyth'], impls_by_nm)
        Y_impl = eval(Y['pyth'], impls_by_nm)

        try:
            for _ in range(3):
                noise = None
                blocks = blocks_impl(noise)
                block = get_block_impl(blocks)
                X = X_impl(noise)(blocks)
                Y = Y_impl(block)
                pairs.append((X, Y))
        except InternalError as e:
            print(e.msg)
            continue
        break

    for X, Y in pairs:
        print(concat_multilines([str(X), str(Y)]))
