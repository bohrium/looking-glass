''' author: samtenka
    change: 2020-02-15
    create: 2019-02-15
    descrp: visualize terms
    to use: 
'''

from utils import CC, pre                               # ansi
from utils import secs_endured, megs_alloced            # profiling
from utils import prod, reseed, bernoulli, geometric    #math

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
                1/4  chance enforce vertical axis 
                1/4  chance enforce horizontal axis 
                1/16 chance enforce slash axis 
                1/16 chance enforce backslash axis 
            Geometry: 
                1/4  chance contains central pixel(s)
                1/4  chance contains border
                1/4  chance ms-paint convex
                1/16 chance is a product of sets
            Topology:
                1/2  chance enforce king connectedness
                1/4  chance enforce ferz connectedness [B]
                1/8  chance enforce simple connectedness

    [A] Together generate 4 rotations, too.  But not every D4 subgroup arises.
    [B] See `en.wikipedia.org/wiki/Fairy_chess_piece`
'''
class Block:
    def __init__(self):
        self.constraints = {
            'sym-axes': {
                'verti': bernoulli(1.0/4),
                'horiz': bernoulli(1.0/4),
                'slash': bernoulli(1.0/16),
                'blash': bernoulli(1.0/16),
            },
            'geometry': {
                'brdrd': bernoulli(1.0/4),
                'cntrd': bernoulli(1.0/4),
                'cnvex': bernoulli(1.0/4),
                'prdct': bernoulli(1.0/16),
            },
            'topology': {
                'kconn': bernoulli(1.0/2),
                'fconn': bernoulli(1.0/4),
                'simpl': bernoulli(1.0/8),
            },
        }
        self.side = 5#2 + sum(geometric(0.25) for _ in range(4))

    def check_sides(self, arr): 
        pre(arr.shape == (self.side, self.side), 'expected side x side array') 

    transfs_by_axis = {
        'verti': (lambda arr: arr[:, ::-1]                          ),
        'horiz': (lambda arr: arr[::-1, :]                          ),
        'slash': (lambda arr: np.rot90(np.transpose(np.rot90(arr))) ),
        'blash': (lambda arr: np.transpose(arr)                     ),
    }
    def obeys_sym_constraints(self, arr): 
        self.check_sides(arr)
        for axis, transf in Block.transfs_by_axis.items():
            if not self.constraints['sym-axes'][axis]:
                continue
            if not np.array_equal(arr, transf(arr)):
                return False
        return True

    def obeys_geo_constraints(self, arr): 
        if self.constraints['geometry']['brdrd']:
            inhab_rows = np.nonzero(np.sum(arr, axis=1)) 
            inhab_cols = np.nonzero(np.sum(arr, axis=0))
            rmin, rmax = np.amin(inhab_rows), np.amax(inhab_rows)
            cmin, cmax = np.amin(inhab_cols), np.amax(inhab_cols)

            if ((np.amin(arr[rmin:rmax+1,cmin]), np.amin(arr[rmin:rmax+1,cmax]), np.amin(arr[rmin,cmin:cmax+1]), np.amin(arr[rmax,cmin:cmax+1])) != 
                (1,1,1,1)):
                return False

        if self.constraints['geometry']['cntrd']:
            half = self.side//2
            if (self.side%2==0 and np.sum(arr[half-1:half+1, half-1:half+1])!=4 or
                self.side%2==1 and arr[half, half]!=1):
                return False

        if self.constraints['geometry']['cnvex']:
            points = set((r,c) for r in range(self.side)
                         for c in range(self.side) if arr[r,c])
            for p in points:
                for q in points:
                    if p <= q: continue 
                    mid = (int((p[0]+q[0])/2.0), int((p[1]+q[1])/2.0))
                    if not arr[mid[0], mid[1]]:
                        return False

        if self.constraints['geometry']['prdct']:
            proj_by_row = np.sum(arr, axis=0) 
            proj_by_col = np.sum(arr, axis=1) 
            reconstruction = np.minimum(1, np.outer(proj_by_row, proj_by_col))
            if not np.array_equal(arr, reconstruction):
                return False

        return True
 
    impls_by_conntype = {
        'kconn': {'offset_pred':lambda dr, dc: max(abs(dr),abs(dc))==1, 'compl':False},
        'fconn': {'offset_pred':lambda dr, dc: abs(dr)+abs(dc)==1,      'compl':False},
        'simpl': {'offset_pred':lambda dr, dc: abs(dr)+abs(dc)==1,      'compl':True },
    }
    def obeys_top_constraints(self, arr): 
        for conntype, impl in Block.impls_by_conntype.items():
            if not self.constraints['topology'][conntype]: continue
            offsets = [(dr,dc) for dr in range(-1,2) for dc in range(-1,2)
                       if impl['offset_pred'](dr, dc)] 


            if impl['compl']: 
                new_arr = np.zeros((self.side+2, self.side+2))
                new_arr[1:self.side+1, 1:self.side+1] = 1-arr
                arr = new_arr
            points = set((r,c) for r in range(self.side)
                         for c in range(self.side) if arr[r,c])
            if not points: continue 

            # check connectedness by breadth first search

            seen = set([min(points)])
            frontier = seen # always a subset of seen
            while frontier:
                neighbors = set([(r+dr,c+dc) for (r,c) in frontier
                                 for (dr, dc) in offsets]).intersection(points)
                frontier = neighbors.difference(seen)
                seen.update(frontier)
            if len(seen) != len(points):
                return False 

        return True
    
    def propose(self):
        base_prob = 1.0 / min(self.side*self.side, 
              2.0
            * 2.0       ** sum(self.constraints['sym-axes'].values())
            * 4.0       **     self.constraints['geometry']['brdrd']
            * self.side **     self.constraints['geometry']['cnvex']
            * self.side ** (2* self.constraints['geometry']['prdct'])
            * 6.0       ** max(self.constraints['topology']['kconn'],
                               self.constraints['topology']['fconn'])
        )  
        arr = np.random.binomial(1, base_prob, (self.side, self.side))
        arr[np.random.randint(self.side), np.random.randint(self.side)] = 1

        for axis, transf in Block.transfs_by_axis.items():
            if not self.constraints['sym-axes'][axis]: continue
            arr = np.maximum(arr, Block.transfs_by_axis[axis](arr))  

        #inhabited_rows = np.array([0])
        #inhabited_cols = np.array([0])
        #while ((np.amin(inhabited_rows), np.amax(inhabited_rows))!=(0, self.side-1) and
        #       (np.amin(inhabited_cols), np.amax(inhabited_cols))!=(0, self.side-1)):
        #    inhabited_rows = np.nonzero(np.sum(arr, axis=0)) 
        #    inhabited_cols = np.nonzero(np.sum(arr, axis=1))
        #    arr = np.maximum(arr, np.random.binomial(1, base_prob, (self.side, self.side)))

        if not self.obeys_geo_constraints(arr):
            if self.constraints['geometry']['brdrd']:
                inhab_rows = np.nonzero(np.sum(arr, axis=1)) 
                inhab_cols = np.nonzero(np.sum(arr, axis=0))
                rmin, rmax = np.amin(inhab_rows), np.amax(inhab_rows)
                cmin, cmax = np.amin(inhab_cols), np.amax(inhab_cols)
                arr[rmin:rmax+1,cmin] = 1
                arr[rmin:rmax+1,cmax] = 1
                arr[rmin,cmin:cmax+1] = 1
                arr[rmax,cmin:cmax+1] = 1

            if self.constraints['geometry']['cntrd']:
                half = self.side//2
                if self.side%2==0:
                    arr[half-1:half+1, half-1:half+1] = 1
                else:
                    arr[half, half] = 1

            if self.constraints['geometry']['cnvex']:
                for r in range(self.side):
                    for c in range(self.side):
                        if arr[r,c]: continue
                        if sum(arr[:r, c])*sum(arr[r:,c ]):
                            arr[max(0,r-1):min(self.side,r+1), c] = 1
                        if sum(arr[ r,:c])*sum(arr[r ,c:]):
                            arr[r, max(0,c-1):min(self.side,c+1)] = 1

            if self.constraints['geometry']['prdct']:
                proj_by_row = np.sum(arr, axis=0) 
                proj_by_col = np.sum(arr, axis=1)
                arr = np.minimum(1, np.outer(proj_by_row, proj_by_col))

        if not self.obeys_top_constraints(arr):
            if (self.constraints['topology']['kconn'] or
                self.constraints['topology']['fconn']):

                offsets = [(dr,dc) for dr in range(-1,2) for dc in range(-1,2)
                           if Block.impls_by_conntype['fconn']['offset_pred'](dr, dc)] 

                is_neighbor = lambda r,c,dr,dc: min(bernoulli(1.0/2), 
                    1 if ( 
                        0<=r+dr<self.side and 0<=c+dc<self.side
                        and arr[r+dr,c+dc]
                    ) else 0
                )
                arr = np.array([[
                    1 if (
                        arr[r,c] or
                        sum(is_neighbor(r,c,dr,dc) for dr,dc in offsets)!=0
                    ) else 0
                for c in range(self.side)] for r in range(self.side)])

        return arr

    def search(self):
        while True:
            arr = self.propose()

            if np.sum(arr) == 0: continue
            inhabited_rows = np.nonzero(np.sum(arr, axis=0)) 
            inhabited_cols = np.nonzero(np.sum(arr, axis=1))
            if ((np.amin(inhabited_rows), np.amax(inhabited_rows))!=(0, self.side-1) and
                (np.amin(inhabited_cols), np.amax(inhabited_cols))!=(0, self.side-1)):
                    continue

            #if not self.obeys_sym_constraints(arr): continue
            if not self.obeys_geo_constraints(arr): continue
            #if not self.obeys_top_constraints(arr): continue
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
