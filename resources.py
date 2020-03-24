''' author: samtenka
    change: 2020-02-27
    create: 2019-02-26
    descrp: Organize dsl primitives' implementations. 
    to use: To import and initialize, type

                from resources import PrimitivesWrapper
                P = PrimitivesWrapper()

            Then the dictionary of (implementations-and-types)-by-name is:

                P.primitives 

            For example, running this file with

                python resources.py

            will print the dictionary item by item.
'''

import inspect
import numpy as np

from utils import InternalError, internal_assert
from utils import CC, pre                               # ansi
from utils import secs_endured, megs_alloced            # profiling
from utils import reseed, bernoulli, geometric, uniform # math

from shape import ShapeGen 
from block import GENERIC_COLORS, Block, block_equals
from grid import Grid

from lg_types import tInt, tCell, tColor, tShape, tBlock, tGrid, tDir, tNoise
from lg_types import tNmbrdBlock, tClrdCell, tPtdGrid, tGridPair, tNmbrdColor, tNmbrdGrid, tClrdGrid
from lg_types import tCount_, tFilter_, tArgmax_, tMap_, tRepeat_
from lg_types import TS

import functools

def sm(lg_type): 
    ''' decorator
    '''
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
        self.primitives = {}
        common_types = {
            tInt, tCell, tColor, tShape, tGrid, tBlock, tInt.s(),
            tNmbrdGrid, tNmbrdGrid.s(), tShape.s(), tCell.s(),
            tPtdGrid, tGridPair, tNmbrdColor, tNmbrdColor.s(),
            tClrdGrid,
        } 
        for goal in common_types:
            self.__make_nil__(elt=goal)
            self.__make_add__(elt=goal)
            self.__make_len__(elt=goal)

            self.__make_assert__(goal=goal)

            self.__make_filter__(elt=goal)
            self.__make_argmax__(elt=goal)
            self.__make_unwrap_sing__(elt=goal)
            self.__make_repeat__(goal=goal)
            self.__make_uniq__(target=goal)
            self.__make_cond__(goal=goal)
            self.__make_eq__(source=goal)
            for t in common_types:
                self.__make_split__(subgoal=t, goal=goal) 
                self.__make_fold__(contained=t, goal=goal)
                self.__make_map__(source=t, target=goal)

        for prod, (fst, snd) in TS.product_decompositions.items():
            self.__make_pair__(prod=prod, fst=fst, snd=snd)
            self.__make_fst__(prod=prod, fst=fst, snd=snd)
            self.__make_snd__(prod=prod, fst=fst, snd=snd)


        for method_nm in dir(PrimitivesWrapper):
            if method_nm.startswith('__'): continue 
            method = self.__getattribute__(method_nm)
            name = method_nm.replace('_C_', '<').replace('_J_', '>') 

            if name in self.primitives:
                print(CC+'@R overwriting predefined @B {} @D '.format(name))

            self.primitives[name] = (
                self.__curry__(method),
                method.lg_type
            )

    #=========================================================================#
    #=  1. IMPLEMENTATIONS  ==================================================#
    #=========================================================================#

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #~~~~~~~~~~ 1.0 Generics  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
                
    #-----------------  1.0.0 coq tactics  -----------------------------------#

    def __make_split__(self, subgoal, goal):
        name = 'split<{}><{}>'.format(subgoal, goal) 
        lg_type = goal.frm(goal.frm(subgoal)).frm(subgoal)
        impl = lambda v: lambda f: f(v)

        self.primitives[name] = (impl, lg_type)

    #-----------------  1.0.1 product types  ---------------------------------#

    def __make_pair__(self, fst, snd, prod):
        name = 'pair<{}>'.format(prod) 
        lg_type = prod.frm(snd).frm(fst)
        impl = lambda x: lambda y: (x,y)
        self.primitives[name] = (impl, lg_type)
    def __make_fst__(self, fst, snd, prod):
        name = 'fst<{}>'.format(prod) 
        lg_type = fst.frm(prod)
        impl = lambda xy: xy[0]
        self.primitives[name] = (impl, lg_type)
    def __make_snd__(self, fst, snd, prod):
        name = 'snd<{}>'.format(prod) 
        lg_type = snd.frm(prod)
        impl = lambda xy: xy[1]
        self.primitives[name] = (impl, lg_type)

    #-----------------  1.0.2 multiset types  --------------------------------#

    def __make_nil__(self, elt):
        name = 'nil<{}>'.format(elt) 
        lg_type = elt.s()
        impl = ()
        self.primitives[name] = (impl, lg_type)
    def __make_add__(self, elt):
        name = 'add<{}>'.format(elt) 
        lg_type = elt.s().frm(elt).frm(elt.s())
        impl = lambda tail: lambda head: (head,)+tail
        self.primitives[name] = (impl, lg_type)
    def __make_len__(self, elt):
        name = 'len<{}>'.format(elt) 
        lg_type = tInt.frm(elt.s())
        impl = lambda l: len(l)
        self.primitives[name] = (impl, lg_type)

    #-----------------  1.0.3 iteration  -------------------------------------#

    def __make_repeat__(self, goal):
        name = 'repeat<{}>'.format(goal) 
        lg_type = goal.frm(goal.frm(goal)).frm(goal).frm(tInt)
        impl = (
            lambda n: lambda i: lambda f:
            functools.reduce(lambda a,b: f(a), [i] + list(range(n)))
        )
        self.primitives[name] = (impl, lg_type)
    def __make_fold__(self, contained, goal):
        name = 'fold<{}><{}>'.format(contained, goal) 
        lg_type = goal.frm(goal.frm(goal).frm(contained)).frm(goal).frm(contained.s())
        impl = (
            lambda cs: lambda i: lambda f:
            functools.reduce(lambda a,b: f(b)(a), [i] + list(cs))
        )
        self.primitives[name] = (impl, lg_type)
    def __make_map__(self, source, target):
        name = 'map<{}><{}>'.format(source, target)
        lg_type = target.s().frm(target.frm(source)).frm(source.s())
        impl = lambda ss: lambda f: map(f, ss)
        self.primitives[name] = (impl, lg_type)

    def __make_filter__(self, elt):
        name = 'filter<{}>'.format(elt)
        lg_type = elt.s().frm(tInt.frm(elt)).frm(elt.s())
        impl = lambda ss: lambda p: tuple(s for s in ss if p(s)!=0)
        self.primitives[name] = (impl, lg_type)

    def __make_uniq__(self, target):
        name = 'uniq<{}>'.format(target)
        lg_type = target.s().frm(target.s())
        impl = lambda ss: list(set(collection))
        self.primitives[name] = (impl, lg_type)
    def __make_argmax__(self, elt):
        name = 'argmax<{}>'.format(elt)
        lg_type = elt.s().frm(tInt.frm(elt)).frm(elt.s())
        def impl(elts):
            def impl_inner(score):
                if not elts: return elts
                scored_elts = [(score(e), e) for e in elts] 
                m = max(se[0] for se in scored_elts)
                f = tuple(e for s,e in scored_elts if s==m)
                return f
            return impl_inner
        self.primitives[name] = (impl, lg_type)
    def __make_unwrap_sing__(self, elt):
        name = 'unwrap_sing<{}>'.format(elt)
        lg_type = elt.frm(elt.s())
        def impl(elts):
            #internal_assert(len(elts)==1, 'can only unwrap singleton')
            #return elts[0]
            internal_assert(len(elts), 'can only unwrap singleton')
            return elts[np.random.choice(len(elts))]
        self.primitives[name] = (impl, lg_type)

    def __make_assert__(self, goal):
        name = 'assert<{}>'.format(goal)
        lg_type = goal.frm(goal).frm(tInt)
        def impl(cond):
            def inner_impl(x):
                internal_assert(cond, 'assert error')
                return x
            return inner_impl
        self.primitives[name] = (impl, lg_type)

    #-----------------  1.0.4 logic  -----------------------------------------#

    def __make_cond__(self, goal):
        name = 'cond<{}>'.format(goal)
        lg_type = goal.frm(goal).frm(goal).frm(tInt)
        impl = lambda c: lambda t: lambda f: t if c else f
        self.primitives[name] = (impl, lg_type)

    def __make_eq__(self, source):
        name = 'eq<{}>'.format(source)
        lg_type = tInt.frm(source).frm(source)
        impl = lambda a: lambda b: 1 if a==b else 0
        self.primitives[name] = (impl, lg_type)

    @sm(tInt.frm(tInt))
    def negate(a): return 1-a

    @sm(tInt.s().frm(tInt))
    def range(a): return tuple(range(a))



    @sm(tInt.frm(tInt).frm(tInt).frm(tInt))
    def affine(m, x, b): return m*x + b



    @sm(tInt.frm(tInt).frm(tInt))
    def less_than(a, b): return 1 if (a<b) else 0

    @sm(tInt.frm(tInt).frm(tInt))
    def at_most(a, b): return 1 if a<=b else 0

    @sm(tInt.frm(tShape).frm(tShape))
    def eq_C_shape_J_(lhs, rhs):
        return 1 if (
            lhs.shape==rhs.shape
            and np.sum(np.abs(lhs-rhs))==0
        ) else 0

    @sm(tInt.frm(tShape))
    def height_C_shape_J_(s):
        return s.shape[0]
    @sm(tInt.frm(tShape))
    def width_C_shape_J_(s):
        return s.shape[1] if len(s.shape)==2 else 0

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #~~~~~~~~~~ 1.1 Basic Constructors  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    @sm(tNoise)
    def noise(): return None

    @sm(tInt.frm(tNoise))
    def coin(tNoise): return bernoulli(0.5)
    @sm(tInt.frm(tNoise))
    def afew(noise): return  2+geometric(0.25)
    @sm(tInt.frm(tNoise))
    def svrl(noise): return  1+bernoulli(0.8)+bernoulli(0.8)+geometric(0.5)
    @sm(tInt.frm(tNoise))
    def many(noise): return 6+geometric(3.0)
    @sm(tInt)
    def zero(): return 0
    @sm(tInt)
    def one(): return 1
    @sm(tInt)
    def two(): return 2
    @sm(tInt)
    def three(): return 3
    @sm(tInt)
    def four(): return 4

    @sm(tDir)
    def east(): return ( 0,  1)
    @sm(tDir)
    def northeast(): return (-1,  1)
    @sm(tDir)
    def north(): return (-1,  0)
    @sm(tDir)
    def northwest(): return (-1, -1)
    @sm(tDir)
    def west(): return ( 0, -1)
    @sm(tDir)
    def southwest(): return ( 1, -1)
    @sm(tDir)
    def south(): return ( 1,  0)
    @sm(tDir)
    def southeast(): return ( 1,  1)


    @sm(tColor.frm(tNoise))
    def rainbow(noise): return uniform(GENERIC_COLORS)
    @sm(tColor)
    def gray(): return 'A'
    @sm(tColor)
    def red(): return 'R'
    @sm(tColor)
    def cyan(): return 'C'
    @sm(tColor)
    def blue(): return 'B'
    @sm(tColor)
    def brown(): return 'N'
    @sm(tColor)
    def green(): return 'G'
    @sm(tColor)
    def yellow(): return 'Y'
    @sm(tColor)
    def orange(): return 'O'
    @sm(tColor)
    def purple(): return 'P'


 
    @sm(tGrid.frm(tInt).frm(tInt))
    def new_grid(h, w): return Grid(h,w)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #~~~~~~~~~~ 1.2 Shapes and Sprites  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    @sm(tShape.frm(tInt).frm(tNoise))
    def gen_shape(noise, side):
        #internal_assert(1<=side<=6, 'requested shape sidelength illegal')
        if not (1<=side<=6): side=6
        SG = ShapeGen()
        SG.set_side(side)
        shape = SG.search(crop=True) 
        return shape 

    @sm(tShape)
    def small_square(): return PrimitivesWrapper.make_square(3) 
    @sm(tShape)                                                     
    def small_plus  (): return PrimitivesWrapper.make_plus  (3) 
    @sm(tShape)                                                     
    def small_times (): return PrimitivesWrapper.make_times (3) 
    @sm(tShape)                                                     
    def large_square(): return PrimitivesWrapper.make_square(5) 
    @sm(tShape)                                                     
    def large_plus  (): return PrimitivesWrapper.make_plus  (5) 
    @sm(tShape)                                                     
    def large_times (): return PrimitivesWrapper.make_times (5) 

    @sm(tShape.frm(tInt))
    def make_square(side):
        shape = np.ones((side, side))
        return shape
    @sm(tShape.frm(tInt))
    def make_plus(side):
        shape = np.zeros((side, side))
        for i in range(side):
            shape[i,side//2] = 1
            shape[side//2,i] = 1
        return shape
    @sm(tShape.frm(tInt))
    def make_times(side):
        shape = np.zeros((side, side))
        for i in range(side):
            shape[i,i] = 1
            shape[i,side-1-i] = 1
        return shape

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #~~~~~~~~~~ 1.3 Measuring and Moving  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    @sm(tCell.frm(tShape))
    def center(shape):
        return (shape.shape[0]//2, (shape.shape[1] if len(shape.shape)==2 else 0)//2)
    @sm(tInt.frm(tShape))
    def volume(shape): return np.sum(shape)
    @sm(tCell.frm(tDir).frm(tCell))
    def displace(cell, offset):
        return tuple(cell[i]+offset[i] for i in range(2))

    @sm(tCell.s().frm(tShape))
    def cells_of(shape):
        h,w = shape.shape
        return tuple((r,c) for r in range(h) for c in range(w) if shape[r,c])



    @sm(tCell.s().s().frm(tShape))
    def holes_of(shape):
        h, w = shape.shape
        bordered = np.zeros((h+2,w+2))
        bordered[1:h+1,1:w+1] = shape
        offsets = {
            (dr,dc)
            for dr in range(-1,2) for dc in range(-1,2)
            if abs(dr)+abs(dc)==1
        }
        points = {
            (r,c)
            for r in range(h+2) for c in range(w+2)
            if bordered[r,c]==0
        } 
        unaccounted_for = points
        parts = []
        while unaccounted_for:
            # bfs
            seen = set([min(unaccounted_for)])
            frontier = seen # always a subset of seen
            while frontier:
                neighbors = set([
                    (r+dr, c+dc) for (r, c) in frontier for (dr, dc) in offsets
                ]).intersection(points)
                frontier = neighbors.difference(seen)
                seen.update(frontier)
            if (0,0) not in seen:
                parts.append(tuple((r-1,c-1) for r,c in seen))
            unaccounted_for = unaccounted_for.difference(seen)
        return tuple(parts)

    @sm(tShape.s().frm(tShape))
    def components_of(shape):
        offsets = {
            (dr,dc)
            for dr in range(-1,2) for dc in range(-1,2)
            if (dr,dc)!=(0,0)
        }
        h, w = shape.shape
        points = {
            (r,c)
            for r in range(h) for c in range(w)
            if shape[r,c]
        } 
        parts = []
        while points:
            # bfs
            seen = set([min(points)])
            frontier = seen # always a subset of seen
            while frontier:
                neighbors = set([
                    (r+dr, c+dc) for (r, c) in frontier for (dr, dc) in offsets
                ]).intersection(points)
                frontier = neighbors.difference(seen)
                seen.update(frontier)
            parts.append(seen)
            points = points.difference(seen)
        return tuple(parts)

    @sm(tInt.s().frm(tGrid))
    def columns(grid): return list(range(grid.W))

    @sm(tInt.s().frm(tGrid))
    def rows(grid): return list(range(grid.H))

    @sm(tGrid.frm(tInt).frm(tGrid))
    def rotate_grid(grid, nb_rots):
        new_grid = grid.copy()
        new_grid.rotate(nb_rots)
        return new_grid

    @sm(tGrid.frm(tDir).frm(tGrid))
    def reflect_grid(grid, axis_dir):
        new_grid = grid.copy()
        new_grid.reflect(axis_dir)
        return new_grid

    @sm(tCell.frm(tDir).frm(tGrid))
    def corner(grid, direction):
        h,w = direction
        h = (
            0           if h==-1 else 
            grid.H-1    if h==+1 else 
            grid.H//2
        )
        w = (
            0           if w==-1 else 
            grid.W-1    if w==+1 else 
            grid.W//2
        )
        return (h,w)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #~~~~~~~~~~ 1.4 Rendering  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    @sm(tGrid.frm(tColor).frm(tShape))
    def monochrome(shape, color):
        G = Grid(H=shape.shape[0], W=(shape.shape[1] if len(shape.shape)==2 else 0))
        G.colors = np.array([
            [color if el else 'K' for el in row]
            for row in shape
        ]) 
        return G
    @sm(tShape.frm(tGrid))
    def silouhette(grid):
        return np.array([
            [1 if el!='K' else 0 for el in row]
            for row in grid.colors
        ])

    @sm(tCell.frm(tGrid).frm(tNoise))
    def sample_cell(noise, grid):
        internal_assert(grid.H*grid.W != 0, 'cannot sample cell from vacuous grid')
        return (uniform(grid.H), uniform(grid.W)) 


    @sm(tPtdGrid.frm(tCell).frm(tShape).frm(tGrid).frm(tNoise))
    def reserve_shape(noise, grid, shape, cell_in_shape):
        new_grid = grid.copy()
        cell = new_grid.reserve_shape(shape, cell_in_shape, spacious=True)
        return (new_grid, cell) 

    @sm(tGrid.frm(tCell).frm(tGrid).frm(tPtdGrid))
    def paint_sprite(ptd_field, sprite, cell_in_sprite):
        new_grid = ptd_field[0].copy()
        new_grid.paint_sprite(sprite, ptd_field[1], cell_in_sprite)
        return new_grid

    @sm(tGrid.frm(tCell).frm(tShape).frm(tPtdGrid))
    def crop(ptd_field, shape, cell_in_sprite):
        new_grid = ptd_field[0].copy()
        new_grid.crop(shape, ptd_field[1], cell_in_sprite)
        return new_grid



    @sm(tGrid.frm(tColor).frm(tCell).frm(tGrid))
    def paint_cell(field, cell, color):
        internal_assert(0<field.H and 0<field.W,
            'cannot paint vacuous grid!'
        )
        new_grid = field.copy()
        if not (0<=cell[0]<field.H and 0<=cell[1]<field.W):
            return new_grid
        #internal_assert(0<=cell[0]<field.H and 0<=cell[1]<field.W,
        #    'cell out of bounds!'
        #)
        new_grid.paint_cell(cell, color)
        return new_grid
    @sm(tGrid.frm(tColor).frm(tInt).frm(tGrid))
    def paint_row(field, row_nb, color):
        new_grid = field.copy()
        new_grid.paint_row(row_nb, color)
        return new_grid
    @sm(tGrid.frm(tColor).frm(tInt).frm(tGrid))
    def paint_column(field, row_nb, color):
        new_grid = field.copy()
        new_grid.paint_column(row_nb, color)
        return new_grid

    @sm(tGrid.frm(tColor.frm(tNoise)).frm(tGrid).frm(tNoise))
    def sprinkle(noise, field, colorgen):
        new_grid = field.copy()
        for _ in range(5):
            new_grid.noise(colorgen(None), 0.1)
        return new_grid

    @sm(tGrid.frm(tCell).frm(tColor).frm(tGrid))
    def fill(field, color, cell):
        #internal_assert(0<=cell[0]<field.H and 0<=cell[1]<field.W,
        #    'cell out of bounds!'
        #)
        new_grid = field.copy()
        if not (0<field.H and 0<field.W): return new_grid 
        if not (0<=cell[0]<field.H and 0<=cell[1]<field.W): cell = (0,0)
        new_grid.fill(color, cell)
        return new_grid

    #@sm(tGrid.frm(tCell).frm(tGrid).frm(tGrid).frm(tNoise))
    #def reserve_and_paint_sprite(noise, field, sprite, cell_in_sprite):
    #    shape = PrimitivesWrapper.sillouhette(sprite)
    #    center = PrimitivesWrapper.center(shape)
    #    new_grid, cell_in_field = PrimitivesWrapper.reserve_shape(
    #        noise, grid, shape, center
    #    )
    #    return PrimitivesWrapper.paint_sprite(
    #        new_grid, sprite, cell_in_field, center
    #    )
 
if __name__=='__main__':
    NB_LINES_PER_FETCH = 25
    P = PrimitivesWrapper()
    print(CC+'@G here are our arc primitives! @D ')
    for i, (p_nm, (p_impl, p_type)) in enumerate(sorted(P.primitives.items())):
        head, type_params = (lambda _: (p_nm[:_], p_nm[_:]) if _!=-1 else (p_nm, ''))(p_nm.find('<'))
        print(CC+'@O {}@G {}@D : @P {} @D '.format(head, type_params, str(p_type)))
        if (i+1)%NB_LINES_PER_FETCH: continue
        input(CC+'show next {} primitives?'.format(NB_LINES_PER_FETCH))
