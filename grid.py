''' author: samtenka
    change: 2020-02-20
    create: 2019-02-16
    descrp:
    to use: 
'''

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
        self.weakoc = np.array([[0   for _ in range(W)] for _ in range(H)])
        self.blkids = np.array([[-1  for _ in range(W)] for _ in range(H)])

        self.H = H 
        self.W = W

    def copy(self):
        G = Grid(self.H, self.W)
        G.colors = self.colors[:,:]
        G.occupd = self.occupd[:,:]
        G.weakoc = self.weakoc[:,:]
        G.blkids = self.blkids[:,:]
        return G

    def fill(self, color, cell_pos):
        offsets = [
            (dr, dc) for dr in range(-1, 2) for dc in range(-1, 2)
            if abs(dr)+abs(dc)==1
        ] 
        points = set([
            (r, c) for r in range(self.H) for c in range(self.W)
            if self.colors[r,c]==self.colors[cell_pos[0], cell_pos[1]] 
        ]) 

        # BFS:
        seen = set([cell_pos])
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

    def noise(self, colors=GENERIC_COLORS):
        for r in range(self.H):
            for c in range(self.W):
                if bernoulli(0.95): continue
                self.colors[r,c] = uniform(colors) 
        return self

    def render_blocks(self, blocks, nb_tries_per_block=5):
        for i, block in enumerate(blocks):
            h, w = block.shape.shape
            for _ in range(nb_tries_per_block):
                #r, c = uniform(self.H-h), uniform(self.W-w)
                r, c = uniform(self.H), uniform(self.W)
                if not self.block_in_bounds(block.shape, r, c): continue
                if self.block_touches(block.shape, r, c): continue
                self.place_block(block.shape, r, c, block.color, block_id=i)
                #self.blocks.append((block.shape, r, c, block.color))
                break
            else: continue
            return None
        return self

    def sample_occupd(self):
        while True:
            r, c = uniform(self.H), uniform(self.W) 
            if self.occupd[r,c]: return r,c

    def get_ray(self, r, c, dr, dc):
        cells = []
        for t in range(1, max(self.H, self.W)):
            rr, cc = r+t*dr, c+t*dc
            if not (0<=rr<self.H and 0<=cc<self.W): break
            if self.occupd[rr, cc]: break
            cells.append((rr,cc))
        return cells

    def render_ray(self, cells, color):
        for rr,cc in cells:
            self.colors[rr,cc] = color

    def block_in_bounds(self, arr, r, c):
        block_h, block_w = arr.shape 
        for dr in range(block_h):
            for dc in range(block_w):
                if not arr[dr,dc]: continue
                if not (0<=r+dr<self.H and 0<=c+dc<self.W):
                    return False
        return True

    def block_touches(self, arr, r, c): 
        block_h, block_w = arr.shape 
        for dr in range(block_h):
            for dc in range(block_w):
                if not arr[dr,dc]: continue
                if not (0<=r+dr<self.H and 0<=c+dc<self.W): continue
                if self.weakoc[r+dr, c+dc]:
                    return True
        return False

    def place_block(self, arr, r, c, col, block_id): 
        block_h, block_w = arr.shape 
        for dr in range(block_h):
            for dc in range(block_w):
                if not arr[dr,dc]: continue
                if not (0<=r+dr<self.H and 0<=c+dc<self.W): continue
                rr, cc = r+dr, c+dc
                self.colors[rr,cc] = col 
                self.blkids[rr,cc] = block_id
                self.occupd[rr,cc] = 1
                self.weakoc[max(0,rr-1):min(self.H,rr+2),
                              max(0,cc-1):min(self.W,cc+2)] = 1 

    def __str__(self):
        '''
        '''
        return str(CC+
            ' {} \n'.format('_'*2*self.W)+
            '\n'.join(
                '|' + ''.join(
                    ('@{} \u2588\u2588@D '.format(self.colors[r,c])
                      if self.colors[r,c]!='K' else '@D  \u00b7@D ')
                     for c in range(self.W)
                ) + '|'
                for r in range(self.H)
            )+
            '\n`{}`\n'.format('`'*2*self.W)
        )

#def concat_multilines(displays):
#    lines = [d.split('\n') for d in displays]
#    heights = [len(ls) for ls in lines] 
#    pre(heights == sorted(heights, reverse=True), '!')
#    return '\n'.join(
#        ' '.join(
#            ls[h] if h<len(ls) else ''
#            for ls in lines
#        )
#        for h in range(max(heights))
#    )
#
#if __name__=='__main__':
#    nb_examples = 4
#    for _ in range(nb_examples):
#        size = 12+geometric(1)
#        S = Scene(size, size)
#        #S.noise(colors=['A'])
#        S.sample_blocks(5)
#    
#        rays = [] 
#        while len(rays) < 2:
#            rays = []
#            r,c = S.sample_occupd()
#            for dr in [-1, 0, 1]: 
#                for dc in [-1, 0, 1]: 
#                    if abs(dr)+abs(dc)!=1: continue
#                    ray = S.get_ray(r,c, dr, dc)
#                    if len(ray) <= 1: continue
#                    rays.append(ray)
#        for ray in rays:
#            S.render_ray(ray, 'A')
#
#        block = S.blocks[S.block_ids[r,c]]
#        block, col = block[0], block[3]
#        Q = Scene(block.shape[0], block.shape[1])
#        Q.place_block(block, 0,0, col, 0)
#    
#        print(CC+concat_multilines([str(S), str(Q)]))
