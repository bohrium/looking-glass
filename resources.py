''' author: samtenka
    change: 2020-02-27
    create: 2019-02-26
    descrp: Organize dsl primitives' implementations. 
    to use: To import and initialize, type
                from resources import PrimitivesWrapper
                P = PrimitivesWrapper()
            Then the dictionary of implementations-and-types-by-name is:
                P.primitives 
            For example, running this file with
                python resources.py
            will print the dictionary item by item.
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

from lg_types import tInt, tCell, tColor, tShape, tBlock, tGrid, tDir, tNoise
from lg_types import tCount_, tFilter_, tArgmax_, tMap_, tRepeat_

def sm(lg_type): 
    def dec(method):
        method.lg_type = lg_type
        method = staticmethod(method)
        return method
    return dec

class PrimitivesWrapper:
    ''' Expose imperatively written subroutines to permit functional style
        calls (by the dsl interpreter).
    '''

    #=========================================================================#
    #=  0. FUNCTIONALIZATION  ================================================#
    #=========================================================================#

    def __curry__(self, method):
        ''' Curry the given method so that its signature goes from 
            ((w, x, ..., y) -> z) to (w -> x -> ... -> y -> z).  Note that in
            the argumentless case, we send (() -> z) to (z).

            For example, if the given method is my_show_concat --- 
            hypothetically implemented in this class as
            
                @sm(tInt)
                def my_show_concat(moo, goo):
                    print(moo+goo)

            --- then (when P has type PrimitivesWrapper) the two expressions

                P.primitives['my_show_concat']('moon')('goon')      # good
                P.primitives['my_show_concat']('moon', 'goon')      # error

            would respectively evaluate to 'moongoon' and raise an error.
        '''
        length = len(inspect.signature(method).parameters)
        if length==0: return method()
        text = ''
        for i in range(length):
            text += 'lambda _x{}: '.format(i)
        text += 'method(' 
        for i in range(length):
            text += '_x{}, '.format(i)
        text += ')' 
        return eval(text, {'method':method})

    def __init__(self):
        ''' Package curried versions of all non-magic methods and attributes
        '''
        self.primitives = {
            nm: (
                self.__curry__(self.__getattribute__(nm)),
                self.__getattribute__(nm).lg_type
            )
            for nm in dir(PrimitivesWrapper) if not nm.startswith('__') 
        }

    #=========================================================================#
    #=  1. IMPLEMENTATIONS  ==================================================#
    #=========================================================================#

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #~~~~~~~~~~ 1.0 Generics  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
                
    #-----------------  1.0.0 split  -----------------------------------------#

    @sm(tShape)
    def split_shape(v, f): return f(v)
    @sm(tInt)
    def split_int(v, f): return f(v)
    @sm(tInt)
    def split_grid(v, f): return f(v)
    @sm(tInt)
    def split_color(v, f): return f(v)
    @sm(tInt)
    def split_cell(v, f): return f(v)
    @sm(tInt)
    def split_intcolorpairs(v, f): return f(v)
    @sm(tInt)
    def split_ptdgrid(v, f): return f(v)

    #-----------------  1.0.1 product types  ---------------------------------#

    @sm(tInt)
    def pair(x,y): return (x,y)
    @sm(tInt)
    def fst(z): return z[0]
    @sm(tInt)
    def snd(z): return z[1]

    #-----------------  1.0.2 iteration  -------------------------------------#

    @sm(tInt)
    def moomap(collection, f): return [f(c) for c in collection]
    @sm(tInt)
    def repeat(n,i,f):
        for _ in range(n):
            i = f(i)
        return i
    @sm(tInt)
    def fold(collection,i,f):
        for c in collection:
            i = f(c)(i)

    @sm(tInt)
    def uniq(collection): return list(set(collection))
    @sm(tInt)
    def moolen(collection): return len(collection) 

    #-----------------  1.0.3 logic  -----------------------------------------#

    @sm(tInt)
    def negate(a): return not a

    @sm(tInt)
    def cond(c,t,f): return t if c else f

    @sm(tInt)
    def lt(a, b): return a<b
    @sm(tInt)
    def eq(a, b): return a==b 
    @sm(tInt)
    def shape_eq(lhs, rhs):
        return (
            lhs.shape==rhs.shape
            and np.sum(np.abs(lhs-rhs))==0
        )


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #~~~~~~~~~~ 1.1 Basic Constructors  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    @sm(tInt)
    def noise(): return None
    @sm(tInt)
    def afew(noise): return  2+geometric(0.25)
    @sm(tInt)
    def many(noise): return 10+geometric(2.50)
    @sm(tInt)
    def rainbow(noise): return uniform(GENERIC_COLORS)
    @sm(tInt)
    def new_grid(h, w): return Grid(h,w)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #~~~~~~~~~~ 1.2 Explicit Shapes and Sprites  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    @sm(tInt)
    def small_square(): return PrimitivesWrapper.make_square(3) 
    @sm(tInt)                                                     
    def small_plus  (): return PrimitivesWrapper.make_plus  (3) 
    @sm(tInt)                                                     
    def small_times (): return PrimitivesWrapper.make_times (3) 
    @sm(tInt)                                                     
    def large_square(): return PrimitivesWrapper.make_square(5) 
    @sm(tInt)                                                     
    def large_plus  (): return PrimitivesWrapper.make_plus  (5) 
    @sm(tInt)                                                     
    def large_times (): return PrimitivesWrapper.make_times (5) 

    @sm(tInt)
    def make_square(side):
        shape = np.ones((side, side))
        return shape
    @sm(tInt)
    def make_plus(side):
        shape = np.zeros((side, side))
        for i in range(side):
            shape[i,side//2] = 1
            shape[side//2,i] = 1
        return shape
    @sm(tInt)
    def make_times(side):
        shape = np.zeros((side, side))
        for i in range(side):
            shape[i,i] = 1
            shape[i,side-1-i] = 1
        return shape

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #~~~~~~~~~~ 1.3 Measuring and Moving  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    @sm(tInt)
    def center(shape): return (shape.shape[0]//2, shape.shape[1]//2)
    @sm(tInt)
    def volume(shape): return np.sum(shape)
    @sm(tInt)
    def displace(cell, offset):
        return tuple(cell[i]+offset[i] for i in range(2))

    @sm(tInt)
    def columns(grid): return list(range(grid.W))
    @sm(tInt)
    def rows(grid): return list(range(grid.H))

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #~~~~~~~~~~ 1.4 Rendering  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    @sm(tInt)
    def monochrome(shape, color):
        G = Grid(H=shape.shape[0], W=shape.shape[1])
        G.colors = np.array([
            [color if el else 'K' for el in row]
            for row in shape
        ]) 
        return G
    @sm(tInt)
    def silouhette(grid):
        return np.array([
            [1 if el!='K' else 0 for el in row]
            for row in grid.colors
        ])

    @sm(tInt)
    def reserve_shape(grid, shape, cell_in_shape):
        new_grid = grid.copy()
        cell = new_grid.reserve_shape(shape, cell_in_shape, spacious=True)
        return (new_grid, cell) 

    @sm(tInt)
    def paint_sprite(field, sprite, cell_in_field, cell_in_sprite):
        new_grid = field.copy()
        new_grid.paint_sprite(sprite, cell_in_field, cell_in_sprite)
        return new_grid
    @sm(tInt)
    def paint_cell(field, cell_in_field, color):
        new_grid = field.copy()
        new_grid.paint_cell(cell_in_field, color)
        return new_grid

    @sm(tInt)
    def reserve_and_paint_sprite(field, sprite, cell_in_sprite):
        new_grid, cell_in_field = reserve_shape(grid, silouhette(sprite), cell_in_sprite) 
        return paint_sprite(new_grid, sprite, cell_in_field, cell_in_sprite)
 
if __name__=='__main__':
    P = PrimitivesWrapper()
    for p_nm, (p_impl, p_type) in P.primitives.items():
        print(CC+'@O {} @D : @P {} @D '.format(p_nm, str(p_type)))
