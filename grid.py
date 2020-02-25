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

    def copy(self):
        G = Grid(self.H, self.W)
        G.colors = np.copy(self.colors)
        G.occupd = np.copy(self.occupd)
        return G

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

    def paint_sprite(self, sprite, cell):
        r, c = cell
        h, w = sprite.shape[0], sprite.shape[1]
        for rr in range(h): 
            if not (0<=r+rr-h//2<self.H): continue
            for cc in range(w): 
                if not (0<=c+cc-w//2<self.W): continue
                color = sprite[rr, cc]
                if color=='K': continue
                self.colors[r+rr-h//2, c+cc-w//2] = sprite[rr, cc]

    def reserve_shape(self, shape, nb_tries=5, spacious=False):
        h, w = shape.shape
        internal_assert(h<=self.H and w<=self.W, 'shape too big to reserve')
        for _ in range(nb_tries):
            r = uniform(range(h//2,self.H-h//2))
            c = uniform(range(w//2,self.W-w//2))
            if not self.shape_in_bounds(shape, r, c): continue
            if self.shape_overlaps_occupied(shape, r, c): continue
            for dr in range(h):
                rr = r+dr-h//2
                if not (0<=rr<self.H): continue
                for dc in range(w):
                    if not shape[dr,dc]: continue
                    cc = c+dc-w//2
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

    def shape_in_bounds(self, arr, r, c):
        h, w = arr.shape
        for dr in range(h):
            for dc in range(w):
                if not arr[dr,dc]: continue
                if not (0<=r+dr-h//2<self.H and 0<=c+dc-w//2<self.W):
                    return False
        return True

    def shape_overlaps_occupied(self, arr, r, c): 
        h, w = arr.shape
        for dr in range(h):
            if not (0<=r+dr-h//2<self.H): continue
            for dc in range(w):
                if not arr[dr,dc]: continue
                if not (0<=c+dc-w//2<self.W): continue
                if self.occupd[r+dr-h//2, c+dc-w//2]:
                    return True
        return False


