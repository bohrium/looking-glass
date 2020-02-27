''' author: samtenka
    change: 2020-02-26
    create: 2019-02-26
    descrp: translate textual code to python function or program tree 
    to use: 
'''

import numpy as np

from utils import ARC_path, InternalError
from utils import CC, pre                               # ansi
from utils import secs_endured, megs_alloced            # profiling
from utils import reseed, bernoulli, geometric, uniform # math

from shape import ShapeGen 
from block import GENERIC_COLORS, Block, block_equals
from grid import Grid

from vis import str_from_grids, render_color

from lg_types import tInt, tCell, tColor, tBlock, tGrid 

class Parser:
    def __init__(self, string):
        self.string = string
        self.i=0

    def get_tree(self):
        self.skip_space()
        tree = self.get_term()
        pre(self.at_end(), 'unable to parse whole string')
        return tree

    def at_end(self):
        return self.i==len(self.string)
    def peek(self):
        return self.string[self.i] if self.i!=len(self.string) else '\0'
    def match(self, s):
        assert self.string[self.i:self.i+len(s)]==s
        self.i+=len(s)
    def march(self):
        self.i+=1
    def skip_space(self):
        while not self.at_end() and (self.peek() in ' \n'):
            self.march()

    def get_identifier(self): 
        old_i = self.i
        while self.peek() in 'abcdefghijklmnopqrstuvwxyz_': self.march()
        return self.string[old_i:self.i]

    def get_term(self): 
        if self.peek()=='(':
            self.match('(')
            self.skip_space()
            tree = [self.get_term()]
            while self.peek()!=')': 
                tree.append(self.get_term())
            self.match(')')
        elif self.peek()=='\\':
            self.match('\\')
            var_nm = self.get_identifier()
            self.skip_space()
            self.match(':')
            self.skip_space()
            type_nm = self.get_identifier()
            self.skip_space()
            self.match('->')
            self.skip_space()
            body = self.get_term() 
            tree = {(var_nm, type_nm):body} 
        elif self.peek() in 'abcdefghijklmnopqrstuvwxyz':
            tree = self.get_identifier()
        else:
            pre(False, 'unknown character #{}#'.format(self.peek()))

        self.skip_space()
        return tree

code = '''(
split_int (many noise) \\side:int -> (
split_int (afew noise) \\nb_objs:int -> (
split_grid (new_grid side side) \\blank:grid -> (
split_color (rainbow noise) \\color_a:grid -> (
split_color (rainbow noise) \\color_b:grid -> (
repeat nb_objs (pair blank blank) \\xy:gridpair -> (
    split_ptdgrid (reserve_shape (snd xy) large_square (center large_square)) \\pg:ptdgrid -> (
        pair
        (split_grid (paint_sprite (fst xy) (monochrome small_plus color_a) (snd pg) (center small_plus)) \\half_painted:grid -> (
            paint_cell half_painted (snd pg) color_b
        ))
        (split_grid (paint_sprite (fst pg) (monochrome large_plus color_a) (snd pg) (center large_plus)) \\half_painted:grid -> (
            paint_sprite half_painted (monochrome large_times color_b) (snd pg) (center large_times)
        ))
    )
)))))))
'''

def evaluate_tree(tree, resources):
    if type(tree)==type(''):
        rtrn = resources[tree]
        if type(rtrn)==type('') and (rtrn in resources):
            rtrn = evaluate_tree(rtrn, resources)
        return rtrn
    elif type(tree)==type({}):
        for (var_nm, var_type), body in tree.items():
            return (lambda x:
                evaluate_tree(body, {
                    k:v for k,v in list(resources.items())+[(var_nm, x)]
                })
            )
    else:
        caller, args = tree[0], tree[1:]
        partial = evaluate_tree(caller, resources)
        for arg in args:
            partial = partial(evaluate_tree(arg, resources))
        return partial

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
def lt(a, b):
    return a<b
def eq(a, b):
    return a==b 
def negate(a):
    return not a
def displace(cell, offset):
    return tuple(cell[i]+offset[i] for i in range(2))
def columns(grid): return list(range(grid.W))
def rows(grid): return list(range(grid.H))
def split_shape(v, f): return f(v)
def split_int(v, f): return f(v)
def split_grid(v, f): return f(v)
def split_color(v, f): return f(v)
def split_cell(v, f): return f(v)
def split_intcolorpairs(v, f): return f(v)
def split_ptdgrid(v, f): return f(v)
def pair(x,y): return (x,y)
def fst(z):
    return z[0]
def snd(z): return z[1]
def cond(c,t,f):
    return t if c else f
def repeat(n,i,f):
    for _ in range(n):
        i = f(i)
    return i
def fold(collection,i,f):
    for c in collection:
        i = f(c)(i)
def moomap(collection, f):
    return [f(c) for c in collection]
def uniq(collection):
    return list(set(collection))
def moolen(collection):
    return len(collection) 

def silouhette(grid):
    return np.array([
        [1 if el!='K' else 0 for el in row]
        for row in grid.colors
    ])
def reserve_shape(grid, shape, cell_in_shape):
    new_grid = grid.copy()
    cell = new_grid.reserve_shape(shape, cell_in_shape, spacious=True)
    return (new_grid, cell) 
def paint_sprite(field, sprite, cell_in_field, cell_in_sprite):
    new_grid = field.copy()
    new_grid.paint_sprite(sprite, cell_in_field, cell_in_sprite)
    return new_grid
def reserve_and_paint_sprite(field, sprite, cell_in_sprite):
    new_grid, cell_in_field = reserve_shape(grid, silouhette(sprite), cell_in_sprite) 
    return paint_sprite(new_grid, sprite, cell_in_field, cell_in_sprite)

def paint_cell(field, cell_in_field, color):
    new_grid = field.copy()
    new_grid.paint_cell(cell_in_field, color)
    return new_grid

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

resources = {
    # splits
    'split_int':lambda i: lambda f: split_int(i,f),
    'split_grid':lambda i: lambda f: split_grid(i,f),
    'split_color':lambda i: lambda f: split_color(i,f),
    'split_ptdgrid':lambda i: lambda f: split_ptdgrid(i,f),

    # product types
    'pair':lambda x: lambda y: pair(x,y),
    'fst':fst,
    'snd':snd,

    # functionals
    'repeat':lambda n: lambda i: lambda f: repeat(n,i,f),

    # constructors
    'afew':lambda noise:  2+geometric(0.25),
    'many':lambda noise: 10+geometric(2.50),
    'noise':None,
    'new_grid':lambda h: lambda w: Grid(h,w),
    'rainbow':lambda noise: uniform(GENERIC_COLORS),
    'small_plus':small_plus,
    'large_plus':large_plus,
    'small_times':small_times,
    'large_times':large_times,
    'large_square':large_square,

    # rendering operators
    'reserve_shape':lambda g: lambda s: lambda c: reserve_shape(g,s,c),
    'monochrome': lambda s: lambda c: monochrome(s,c),
    'center':center,
    'paint_sprite':lambda g: lambda s: lambda c: lambda cc: paint_sprite(g,s,c,cc),
    'paint_cell':lambda g: lambda cel: lambda col: paint_cell(g,cel, col),
}

if __name__=='__main__':
    P = Parser(code)
    t = P.get_tree()
    print(t)
    while True:
       try:
           x,y = evaluate_tree(t, resources)
           print(CC+str_from_grids([x.colors, y.colors], render_color))
           break
       except InternalError:
           continue
