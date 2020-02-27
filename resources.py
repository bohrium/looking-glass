''' author: samtenka
    change: 2020-02-26
    create: 2019-02-26
    descrp: translate textual code to python function or program tree 
    to use: import
                from resources import primitives
'''

import inspect
import numpy as np

from utils import ARC_path, InternalError
from utils import CC, pre                               # ansi
from utils import secs_endured, megs_alloced            # profiling
from utils import reseed, bernoulli, geometric, uniform # math

from shape import ShapeGen 
from block import GENERIC_COLORS, Block, block_equals
from grid import Grid

#=============================================================================#
#=====  0. ANSI CONTROL FOR RICH OUTPUT TEXT =================================#
#=============================================================================#

sm = staticmethod

class PrimitivesWrapper:
    ''' Organize and expose primitives to be called by a dsl interpreter
    '''
    def __curry__(method):
        ''' Curry the given method so that its signature goes from 
            ((w, x, ..., y) -> z) to (w -> x -> ... -> y -> z).  This helps us
            translate imperatively written subroutines to permit functional
            style calls.

            For example, if the given method is my_show_concat --- 
            hypothetically implemented in this class as
            
                @sm
                def my_show_concat(moo, goo):
                    print(moo+goo)

            --- then (when P has type PrimitivesWrapper) the two expressions

                P.primitives['my_show_concat']('moon')('goon')      # good
                P.primitives['my_show_concat']('moon', 'goon')      # error

            would respectively evaluate to 'moongoon' and raise an error.
        '''
        length = len(inspect.signature(method).parameters)
        text = ''
        for i in range(length):
            text += 'lambda _x{}: '.format(i)
        text += 'method(' 
        for i in range(length):
            text += '_x{}, '.format(i)
        text += ')' 
        return eval(text, {'method':method})

    def __init__(self):
        ''' Package curried versions of all non-magic methods
        '''
        self.primitives = {
            nm: self.__curry__(self.__getattribute__(nm))
            for nm in dir(PrimitivesWrapper) if not nm.startswith('__') 
        }

#=============================================================================#
#=====  1. ANSI CONTROL FOR RICH OUTPUT TEXT =================================#
#=============================================================================#

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #~~~~~~~~~~ 1.0 Generics  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
                
    #-----------------  1.0.0 split  -----------------------------------------#

    @sm
    def split_shape(v, f): return f(v)
    @sm
    def split_int(v, f): return f(v)
    @sm
    def split_grid(v, f): return f(v)
    @sm
    def split_color(v, f): return f(v)
    @sm
    def split_cell(v, f): return f(v)
    @sm
    def split_intcolorpairs(v, f): return f(v)
    @sm
    def split_ptdgrid(v, f): return f(v)

    #-----------------  1.0.1 product types  ---------------------------------#

    @sm
    def pair(x,y): return (x,y)
    @sm
    def fst(z): return z[0]
    @sm
    def snd(z): return z[1]

    #-----------------  1.0.2 iteration  -------------------------------------#

    @sm
    def moomap(collection, f): return [f(c) for c in collection]
    @sm
    def repeat(n,i,f):
        for _ in range(n):
            i = f(i)
        return i
    @sm
    def fold(collection,i,f):
        for c in collection:
            i = f(c)(i)

    @sm
    def uniq(collection): return list(set(collection))
    @sm
    def moolen(collection): return len(collection) 

    #-----------------  1.0.3 logic  -----------------------------------------#

    @sm
    def negate(a): return not a

    @sm
    def cond(c,t,f): return t if c else f

    @sm
    def lt(a, b): return a<b
    @sm
    def eq(a, b): return a==b 
    @sm
    def shape_eq(lhs, rhs):
        return (
            lhs.shape==rhs.shape
            and np.sum(np.abs(lhs-rhs))==0
        )


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #~~~~~~~~~~ 1.1 Concrete Constructors  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #~~~~~~~~~~ 1.2 Parameterized Objects  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    @sm
    def make_plus(side):
        shape = np.zeros((side, side))
        for i in range(side):
            shape[i,side//2] = 1
            shape[side//2,i] = 1
        return shape

    def make_times(side):
        shape = np.zeros((side, side))
        for i in range(side):
            shape[i,i] = 1
            shape[i,side-i] = 1
        return shape

    def make_square(side):
        shape = np.ones((side, side))
        return shape

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #~~~~~~~~~~ 1.3 Measuring and Moving  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    @sm
    def center(shape): return (shape.shape[0]//2, shape.shape[1]//2)
    @sm
    def volume(shape): return np.sum(shape)
    @sm
    def displace(cell, offset):
        return tuple(cell[i]+offset[i] for i in range(2))

    @sm
    def columns(grid): return list(range(grid.W))
    @sm
    def rows(grid): return list(range(grid.H))

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #~~~~~~~~~~ 1.4 Rendering  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    @sm
    def monochrome(shape, color):
        G = Grid(H=shape.shape[0], W=shape.shape[1])
        G.colors = np.array([
            [color if el else 'K' for el in row]
            for row in shape
        ]) 
        return G
    @sm
    def silouhette(grid):
        return np.array([
            [1 if el!='K' else 0 for el in row]
            for row in grid.colors
        ])

    @sm
    def reserve_shape(grid, shape, cell_in_shape):
        new_grid = grid.copy()
        cell = new_grid.reserve_shape(shape, cell_in_shape, spacious=True)
        return (new_grid, cell) 

    @sm
    def paint_sprite(field, sprite, cell_in_field, cell_in_sprite):
        new_grid = field.copy()
        new_grid.paint_sprite(sprite, cell_in_field, cell_in_sprite)
        return new_grid
    @sm
    def paint_cell(field, cell_in_field, color):
        new_grid = field.copy()
        new_grid.paint_cell(cell_in_field, color)
        return new_grid

    @sm
    def reserve_and_paint_sprite(field, sprite, cell_in_sprite):
        new_grid, cell_in_field = reserve_shape(grid, silouhette(sprite), cell_in_sprite) 
        return paint_sprite(new_grid, sprite, cell_in_field, cell_in_sprite)
 

