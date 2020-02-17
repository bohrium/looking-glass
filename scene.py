''' author: samtenka
    change: 2020-02-16
    create: 2019-02-16
    descrp: visualize terms
    to use: 
'''

from utils import CC, pre                               # ansi
from utils import secs_endured, megs_alloced            # profiling
from utils import reseed, bernoulli, geometric, uniform # math

import numpy as np

from block import Block

COLORS = 'KBRGYAPOCN'
SPECIAL_COLORS = 'KA'
GENERIC_COLORS = 'BRGYPOCN'

class Scene:
    def __init__(self, H, W):

        self.colors = np.array([['K' for _ in range(W)] for _ in range(H)])
        self.occupied = np.array([[0 for _ in range(W)] for _ in range(H)])
        self.weak_occ = np.array([[0 for _ in range(W)] for _ in range(H)])
        self.block_ids = np.array([[-1 for _ in range(W)] for _ in range(H)])

        self.H = H 
        self.W = W

    def sample_blocks(self, nb_blocks, sides=[2,3,4,5], colors=GENERIC_COLORS):
        B = Block()
        self.blocks = []
        while len(self.blocks) != nb_blocks:
            side = uniform(sides)
            B.set_side(side)
            block = B.search() 
            color = uniform(colors)
            for _ in range(5):
                r, c = uniform(self.H-side), uniform(self.W-side)
                if not S.block_in_bounds(block, r, c): continue
                if S.block_touches(block, r, c): continue
                S.place_block(block, r, c, color, block_id=len(self.blocks))
                self.blocks.append((block, r, c, color))
                break

    def noise(self, colors=GENERIC_COLORS):
        for r in range(self.H):
            for c in range(self.W):
                if bernoulli(0.95): continue
                self.colors[r,c] = uniform(colors) 

    def sample_occupied(self):
        while True:
            r, c = uniform(self.H), uniform(self.W) 
            if self.occupied[r,c]: return r,c

    def get_ray(self, r, c, dr, dc):
        cells = []
        for t in range(1, max(self.H, self.W)):
            rr, cc = r+t*dr, c+t*dc
            if not (0<=rr<self.H and 0<=cc<self.W): break
            if self.occupied[rr, cc]: break
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
                if self.weak_occ[r+dr, c+dc]:
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
                self.block_ids[rr,cc] = block_id
                self.occupied[rr,cc] = 1
                self.weak_occ[max(0,rr-1):min(self.H,rr+2),
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
    for _ in range(2):
        size = 12+geometric(3)
        print(size)
        S = Scene(size, size)
        S.noise(colors=['A'])
        S.sample_blocks(6)
    
        rays = [] 
        while len(rays) < 2:
            rays = []
            r,c = S.sample_occupied()
            for dr in [-1, 0, 1]: 
                for dc in [-1, 0, 1]: 
                    if abs(dr)+abs(dc)!=1: continue
                    ray = S.get_ray(r,c, dr, dc)
                    if len(ray) <= 1: continue
                    rays.append(ray)
        for ray in rays:
            S.render_ray(ray, 'A')

        block = S.blocks[S.block_ids[r,c]]
        block, col = block[0], block[3]
        Q = Scene(block.shape[0], block.shape[1])
        Q.place_block(block, 0,0, col, 0)
    
        print(CC+concat_multilines([str(S), str(Q)]))
