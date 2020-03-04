''' author: samtenka
    change: 2020-02-20
    create: 2019-02-16
    descrp:
    to use: 
'''

from utils import internal_assert                       # maybe
from utils import CC, pre                               # ansi
from utils import secs_endured, megs_alloced            # profiling
from utils import reseed, bernoulli, geometric, uniform # math

import numpy as np

from shape import ShapeGen
from block import GENERIC_COLORS, Block

class Grid:
    def __init__(self, H, W):

        self.colors = np.array([['K' for _ in range(W)] for _ in range(H)])
        self.occupd = np.array([[0   for _ in range(W)] for _ in range(H)])

        self.H = H 
        self.W = W

    def __eq__(self, rhs):
        if self.H != rhs.H or self.W != rhs.W: return False
        for h in range(self.H):
            for w in range(self.W):
                if self.colors[h,w] != rhs.colors[h,w]:
                    return False
        return True

    def center(self):
        return (self.H//2, self.W//2)
    def get_border_cell(self, cell, offset):
        internal_assert(self.cell_in_bounds(cell), 'ray needs to start inside grid')
        while True:
            new_cell = tuple(cell[0]+offset[i] for i in range(2))
            if not self.cell_in_bounds(new_cell):
                return cell
            cell = new_cell

    def copy(self):
        G = Grid(self.H, self.W)
        G.colors = np.copy(self.colors)
        G.occupd = np.copy(self.occupd)
        return G

    def rotate(self, rotations):
        internal_assert(self.H*self.W, 'cannot rotate vacuous grid')
        self.colors = np.rot90(self.colors, rotations)
        self.occupd = np.rot90(self.occupd, rotations)
        if rotations%2:
            self.H, self.W = self.W, self.H

    def reflect(self, axis_as_offset):
        rr,cc = axis_as_offset
        internal_assert(
            rr*rr+cc*cc in [1,2],
            'do not know how to reflect across this axis'
        )
        if rr*rr+cc*cc == 1: # orthogonal flip
            transform = lambda arr: arr[::-1,:] if rr else arr[:,::-1]
        else: # diagonal flip
            transform = (
                lambda arr:
                np.transpose(arr) if rr*cc==1 else np.transpose(arr[::-1,:])[::-1,:]
            )
            self.H, self.W = self.W, self.H

        self.colors = transform(self.colors)
        self.occupd = transform(self.occupd)

    def replace_color(self, color_source, color_target):
        self.colors = np.array([
            [color_target if el==color_source else el for el in row]
            for row in self.colors
        ]) 

    def fill(self, color, cell):
        offsets = [
            (dr, dc) for dr in range(-1, 2) for dc in range(-1, 2)
            if abs(dr)+abs(dc)==1
        ] 
        points = set([
            (r, c) for r in range(self.H) for c in range(self.W)
            if self.colors[r,c]==self.colors[cell[0], cell[1]] 
        ]) 

        # Breadth First Search:
        seen = set([cell])
        frontier = seen # always a subset of seen
        while frontier:
            for r,c in frontier:
                self.colors[r,c] = color
            neighbors = set([
                (r+dr, c+dc) for (r, c) in frontier for (dr, dc) in offsets
            ]).intersection(points)
            frontier = neighbors.difference(seen)
            seen.update(frontier)
        return self

    def noise(self, colors, density=0.05):
        for r in range(self.H):
            for c in range(self.W):
                if not bernoulli(density): continue
                self.colors[r,c] = uniform(colors) 
        return self

    def paint_cell(self, cell, color):
        if not self.cell_in_bounds(cell):
            return
        r, c = cell
        self.colors[r, c] = color

    def paint_row(self, row, color):
        if not self.cell_in_bounds((row,0)):
            return
        self.colors[row,:] = color

    def paint_column(self, column, color):
        if not self.cell_in_bounds((0,column)):
            return
        self.colors[:,column] = color

    def paint_sprite(self, sprite, cell, sprite_cell):
        r, c = cell
        h, w = sprite.colors.shape[0], sprite.colors.shape[1]
        rrr,ccc = sprite_cell
        for rr in range(h): 
            if not (0<=r+rr-rrr<self.H): continue
            for cc in range(w): 
                if not (0<=c+cc-ccc<self.W): continue
                color = sprite.colors[rr, cc]
                if color=='K': continue
                self.colors[r+rr-rrr, c+cc-ccc] = color

    def reserve_shape(self, shape, shape_cell, nb_tries=5, spacious=False):
        h, w = shape.shape
        rrr,ccc = shape_cell
        internal_assert(h<=self.H and w<=self.W, 'shape too big to reserve')
        for _ in range(nb_tries):
            r = uniform(range(rrr,self.H-h+rrr+1))
            c = uniform(range(ccc,self.W-w+ccc+1))
            if not self.shape_in_bounds(shape, r-rrr, c-ccc): continue
            if self.shape_overlaps_occupied(shape, r-rrr, c-ccc): continue
            for dr in range(h):
                rr = r+dr-rrr
                if not (0<=rr<self.H): continue
                for dc in range(w):
                    if not shape[dr,dc]: continue
                    cc = c+dc-ccc
                    if not (0<=cc<self.W): continue
                    if spacious:
                        self.occupd[max(0,rr-1):min(rr+2,self.H),
                                    max(0,cc-1):min(cc+2,self.W)] = 1
                    else:
                        self.occupd[rr,cc] = 1
            return r,c
        internal_assert(False, 'failed to reserve shape')

    #def sample_occupd(self):
    #    while True:
    #        r, c = uniform(self.H), uniform(self.W) 
    #        if self.occupd[r,c]: return r,c

    #def sample_cell(self):
    #    return uniform(self.H), uniform(self.W) 

    #def get_ray(self, r, c, dr, dc):
    #    cells = []
    #    for t in range(1, max(self.H, self.W)):
    #        rr, cc = r+t*dr, c+t*dc
    #        if not (0<=rr<self.H and 0<=cc<self.W): break
    #        if self.occupd[rr, cc]: break
    #        cells.append((rr,cc))
    #    return cells

    #def render_ray(self, cells, color):
    #    for rr,cc in cells:
    #        self.colors[rr,cc] = color

    def cell_in_bounds(self, cell):
        r, c = cell
        return (0<=r<self.H and 0<=c<self.W)

    def shape_in_bounds(self, arr, r, c):
        h, w = arr.shape
        for dr in range(h):
            for dc in range(w):
                if not arr[dr,dc]: continue
                if not (0<=r+dr<self.H and 0<=c+dc<self.W):
                    return False
        return True

    def shape_overlaps_occupied(self, arr, r, c): 
        h, w = arr.shape
        for dr in range(h):
            if not (0<=r+dr<self.H): continue
            for dc in range(w):
                if not arr[dr,dc]: continue
                if not (0<=c+dc<self.W): continue
                if self.occupd[r+dr, c+dc]:
                    return True
        return False


