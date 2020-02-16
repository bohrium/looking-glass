''' author: samtenka
    change: 2020-02-16
    create: 2019-02-15
    descrp: visualize terms
    to use: 
'''

from utils import CC, pre                               # ansi
from utils import secs_endured, megs_alloced            # profiling
from utils import prod, reseed, bernoulli, geometric    # math

import numpy as np

#=============================================================================#
#       0. VISUALIZE OBJECTS                                                  #
#=============================================================================#

#-----------------------------------------------------------------------------#
#               0.0 define ansi command abbreviations                         #
#-----------------------------------------------------------------------------#

'''
    Block Language:
        Constraints:
            Reflection Symmetries [A]:
                1/2  chance enforce vertical axis 
                1/4  chance enforce horizontal axis 
                1/16 chance enforce slash axis 
                1/16 chance enforce backslash axis 
            Geometry: 
                1/8  chance contains central pixel(s)
                1/8  chance contains border
                1/8  chance ms-paint convex
                1/16 chance is a product of sets
            Topology:
                1/2  chance enforce king connectedness
                1/4  chance enforce ferz connectedness [B]
                1/4  chance enforce simple connectedness
                1/8  chance enforce non-(simple connectedness)

    [A] Together generate 4 rotations, too.  But not every D4 subgroup arises.
    [B] See `en.wikipedia.org/wiki/Fairy_chess_piece`
'''
class Block:
    def __init__(self):
        self.constraints = {
            'sym-axes': {
                'verti': bernoulli(1.0/2),
                'horiz': bernoulli(1.0/4),
                'slash': bernoulli(1.0/16),
                'blash': bernoulli(1.0/16),
            },
            'geometry': {
                'brdrd': bernoulli(1.0/8),
                'cntrd': bernoulli(1.0/8),
                'cnvex': bernoulli(1.0/8),
                'prdct': bernoulli(1.0/16),
            },
            'topology': {
                'kconn': bernoulli(1.0/2),
                'fconn': bernoulli(1.0/4),
                'simpl': bernoulli(1.0/8),
                'nsimp': bernoulli(2.0/8), # 2.0 to counter overruling by others
            },
        }

        self.side = 2 + sum(bernoulli(0.25) for _ in range(6))

        if (self.side<=2 or
            (self.side<=4 and self.constraints['geometry']['cntrd']) or
            self.constraints['geometry']['cnvex'] or
            self.constraints['geometry']['prdct'] or
            self.constraints['topology']['simpl']):
            self.constraints['topology']['nsimp'] = 0


    def check_sides(self, arr): 
        pre(arr.shape == (self.side, self.side), 'expected side x side array') 

    transfs_by_axis = {
        'verti': (lambda arr: arr[:, ::-1]                             ),
        'horiz': (lambda arr: arr[::-1, :]                             ),
        'slash': (lambda arr: np.transpose(arr[::-1])[::-1]            ),
        'blash': (lambda arr: np.transpose(arr)                        ),
    }
    def check_sym_pred_by_nm(self, arr, axis): 
        self.check_sides(arr)
        return np.array_equal(arr, Block.transfs_by_axis[axis](arr))
    def make_more_sym(self, arr, axis):
        if self.check_sym_pred_by_nm(arr, axis):
            return
        arr[:] = np.maximum(arr, Block.transfs_by_axis[axis](arr))  
    def passes_sym_req(self, arr, axis):
        return (
            not self.constraints['sym-axes'][axis] or
            self.check_sym_pred_by_nm(arr, axis)
        )
    def passes_all_sym_reqs(self, arr): 
        for axis in Block.transfs_by_axis:
            if not self.passes_sym_req(arr, axis):
                return False
        return True

    def check_brdrd(self, arr):
        inhab_rows = np.nonzero(np.sum(arr, axis=1)) 
        inhab_cols = np.nonzero(np.sum(arr, axis=0))
        rmin, rmax = np.amin(inhab_rows), np.amax(inhab_rows)
        cmin, cmax = np.amin(inhab_cols), np.amax(inhab_cols)

        for edge in (arr[rmin:rmax+1,cmin], arr[rmin,cmin:cmax+1]):
            for hasendpt in (np.amin, np.amax):
                if not hasendpt(edge):
                    return False 
        return True

    def make_more_brdrd(self, arr):
        if self.check_brdrd(arr):
            return
        inhab_rows = np.nonzero(np.sum(arr, axis=1)) 
        inhab_cols = np.nonzero(np.sum(arr, axis=0))
        rmin, rmax = np.amin(inhab_rows), np.amax(inhab_rows)
        cmin, cmax = np.amin(inhab_cols), np.amax(inhab_cols)
        arr[rmin:rmax+1,cmin] = 1
        arr[rmin:rmax+1,cmax] = 1
        arr[rmin,cmin:cmax+1] = 1
        arr[rmax,cmin:cmax+1] = 1

    def check_cntrd(self, arr):
        half = self.side//2
        if self.side%2==0 and np.min(arr[half-1:half+1, half-1:half+1])!=1:
            return False
        elif self.side%2==1 and arr[half, half]!=1:
            return False
        return True

    def make_more_cntrd(self, arr):
        if self.check_cntrd(arr):
            return

        half = self.side//2
        if self.side%2==0:
            arr[half-1:half+1, half-1:half+1] = 1
        else:
            arr[half, half] = 1

    def check_cnvex(self, arr):
        points = set((r,c) for r in range(self.side)
                     for c in range(self.side) if arr[r,c])
        for p in points:
            for q in points:
                if p <= q: continue 
                mid = ((p[0]+q[0])/2.0, (p[1]+q[1])/2.0)
                nb_hits = sum(
                    arr[int(rround(mid[0])), int(cround(mid[1]))]
                    for rround in (np.ceil, np.floor)
                    for cround in (np.ceil, np.floor)
                )
                if nb_hits==0:
                    return False
        return True

    def make_more_cnvex(self, arr):
        if self.check_cnvex(arr):
            return

        for r in range(self.side):
            for c in range(self.side):
                if arr[r,c]: continue
                if sum(arr[:r, c])*sum(arr[r:,c ]):
                    arr[max(0,r-1):min(self.side,r+1), c] = 1
                if sum(arr[ r,:c])*sum(arr[r ,c:]):
                    arr[r, max(0,c-1):min(self.side,c+1)] = 1

    def check_prdct(self, arr):
        proj_by_row = np.sum(arr, axis=0) 
        proj_by_col = np.sum(arr, axis=1) 
        reconstruction = np.minimum(1, np.outer(proj_by_row, proj_by_col))
        if not np.array_equal(arr, reconstruction):
            return False
        return True

    def make_more_prdct(self, arr):
        if self.check_prdct(arr):
            return
        proj_by_row = np.sum(arr, axis=0) 
        proj_by_col = np.sum(arr, axis=1)
        arr[:] = np.minimum(1, np.outer(proj_by_row, proj_by_col))

    geo_preds_by_nm = {
        'brdrd': check_brdrd,
        'cntrd': check_cntrd,
        'cnvex': check_cnvex,
        'prdct': check_prdct,
    }
    def passes_geo_req(self, arr, pred_nm):
        return (
            not self.constraints['geometry'][pred_nm] or
            Block.geo_preds_by_nm[pred_nm](self, arr)
        )
    def passes_all_geo_reqs(self, arr): 
        for pred_nm in Block.geo_preds_by_nm:
            if not self.passes_geo_req(arr, pred_nm):
                return False
        return True

    offset_preds_by_top_pred_nm = {
        'kconn': lambda dr, dc: max(abs(dr),abs(dc))==1,
        'fconn': lambda dr, dc: abs(dr)+abs(dc)==1,     
        'simpl': lambda dr, dc: abs(dr)+abs(dc)==1,
        'nsimp': lambda dr, dc: None,
    }
    def check_connected(self, arr, offset_pred):
        offsets = [
            (dr,dc) for dr in range(-1,2) for dc in range(-1,2)
            if offset_pred(dr, dc)
        ] 

        points = set((r,c) for r in range(arr.shape[0])
                     for c in range(arr.shape[1]) if arr[r,c])
        if not points:
            return True

        # check connectedness by breadth first search

        seen = set([min(points)])
        frontier = seen # always a subset of seen
        while frontier:
            neighbors = set([(r+dr,c+dc) for (r,c) in frontier
                             for (dr, dc) in offsets]).intersection(points)
            frontier = neighbors.difference(seen)
            seen.update(frontier)

        return len(seen) == len(points)

    def check_top_req(self, arr, nm):
        if nm=='nsimp':
            return not self.check_top_req(arr, 'simpl')
        elif nm=='simpl':
            # border with 0's
            new_arr = np.ones((self.side+2, self.side+2))
            new_arr[1:self.side+1, 1:self.side+1] = 1-arr
            arr = new_arr
        return self.check_connected(arr, Block.offset_preds_by_top_pred_nm[nm])

    def make_more_connected(self, arr):
        if self.check_top_req(arr, 'fconn'):
            return

        offsets = [
            (dr,dc) for dr in range(-1,2) for dc in range(-1,2)
            if Block.offset_preds_by_top_pred_nm['fconn'](dr, dc)
        ] 

        is_neighbor = lambda r,c,dr,dc: min(bernoulli(1.0/2), 
            1 if ( 
                0<=r+dr<self.side and 0<=c+dc<self.side
                and arr[r+dr,c+dc]
            ) else 0
        )
        for r in range(self.side):
            for c in range(self.side):
                arr[r,c] = max(arr[r, c], min(1, 
                    sum(is_neighbor(r,c,dr,dc) for dr,dc in offsets)
                ))

    def passes_top_req(self, arr, pred_nm):
        return (
            not self.constraints['topology'][pred_nm] or
            self.check_top_req(arr, pred_nm)
        )
    def passes_all_top_reqs(self, arr): 
        for pred_nm in Block.offset_preds_by_top_pred_nm:
            if not self.passes_top_req(arr, pred_nm):
                return False
        return True

    def propose(self):
        base_prob = 1.0 / (
              2.0
            * 2.5       ** sum(self.constraints['sym-axes'].values())
            * 4.0       **     self.constraints['geometry']['brdrd']
            * self.side **     self.constraints['geometry']['cnvex']
            * self.side ** (   self.constraints['geometry']['prdct'])
            * 6.0       ** max(self.constraints['topology']['kconn'],
                               self.constraints['topology']['fconn'])
            * self.side **     self.constraints['topology']['simpl']
        )  
        arr = np.random.binomial(1, base_prob, (self.side, self.side))
        arr[np.random.randint(self.side), np.random.randint(self.side)] = 1

        for axis in Block.transfs_by_axis:
            if not self.passes_sym_req(arr, axis):
                self.make_more_sym(arr, axis)

        if not self.passes_geo_req(arr, 'brdrd'): self.make_more_brdrd(arr)
        if not self.passes_geo_req(arr, 'cntrd'): self.make_more_cntrd(arr)
        if not self.passes_geo_req(arr, 'cnvex'): self.make_more_cnvex(arr)
        if not self.passes_geo_req(arr, 'prdct'): self.make_more_prdct(arr)

        if not (self.passes_top_req(arr, 'kconn') and self.passes_top_req(arr, 'fconn')):
            self.make_more_connected(arr)
        if not self.passes_top_req(arr, 'simpl'):
            self.make_more_cnvex(arr)

        return arr

    def search(self):
        while True:
            arr = self.propose()

            # retry if not big enough 
            if np.sum(arr) == 0: continue
            inhab_rows = np.nonzero(np.sum(arr, axis=0)) 
            inhab_cols = np.nonzero(np.sum(arr, axis=1))
            rmin, rmax = np.amin(inhab_rows), np.amax(inhab_rows)
            cmin, cmax = np.amin(inhab_cols), np.amax(inhab_cols)
            if ((rmin, rmax)!=(0, self.side-1) and
                (cmin, cmax)!=(0, self.side-1)):
                continue

            if not self.passes_all_sym_reqs(arr): continue
            if not self.passes_all_geo_reqs(arr): continue
            if not self.passes_all_top_reqs(arr): continue
            return arr

    def colored(self, arrs):
        return str(CC + '\n'.join(
            ' '.join(
                '|' +
                ''.join('@M []@D ' if arr[r,c] else '@W   @D ' for c in range(self.side)) +
                '|'
                for arr in arrs
            )
            for r in range(self.side)
        ))

#=============================================================================#
#       3. ILLUSTRATE UTILITIES                                               #
#=============================================================================#

for _ in range(5): 
    B = Block()
    print(CC + '@Y {} @D '.format(str(B.constraints)))
    print(CC + '@D {} @D '.format(str(B.side)))
    print(B.colored([B.search() for _ in range(120//(3+2*B.side))]))
    print()
