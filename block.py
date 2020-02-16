''' author: samtenka
    change: 2020-02-16
    create: 2019-02-15
    descrp: visualize terms
    to use: 
'''

from utils import CC, pre                       # ansi
from utils import secs_endured, megs_alloced    # profiling
from utils import reseed, bernoulli, geometric  # math

import numpy as np

#=============================================================================#
#=====  0. GENERATIVE MODEL FOR BLOCKS, HIGH LEVEL  ==========================#
#=============================================================================#

class Block:
    '''
        Blocks are subject to a randomly sampled set of constraints:

            Reflection Symmetries [A]:
                1/4  chance enforce vertical axis 
                1/4  chance enforce horizontal axis 
                1/16 chance enforce slash axis 
                1/16 chance enforce backslash axis 
            Geometry: 
                1/8  chance contains border
                1/8  chance ms-paint convex
                1/8  chance not (ms-paint convex) [B]
                1/16 chance is a product of sets
            Topology:
                3/4  chance enforce king connectedness
                1/4  chance enforce ferz connectedness [C]
                1/4  chance enforce simple connectedness [D]
                1/8  chance enforce non-(simple connectedness) [B, D]

        [A] These reflections together generate rotations, too.  But not every
            D4 subgroup arises.
        [B] Since we allow conflicting constraints such as convexity to
            overrule these constraints, the sampling probability 1/8 is not
            implemented exactly. 
        [C] Ferzs are pieces that move in size-1 *orthogonal* steps.
        [D] An empty cell surrounded only orthogonally counts as a hole.
    '''

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #~~~~~~~~~  0.0 Sample Parameters of Block Type  ~~~~~~~~~~~~~~~~~~~~~~~~~#

    def __init__(self):
        '''
        '''
        #-------------  0.0.0 sample side length  ----------------------------#

        self.side = 2 + sum(bernoulli(1.0/4) for _ in range(6))

        #-------------  0.0.1 sample constraint set  -------------------------#

        self.constraints = {
            'sym': {
                'verti': bernoulli(1.0/4),
                'horiz': bernoulli(1.0/4),
                'slash': bernoulli(1.0/16),
                'blash': bernoulli(1.0/16),
            },
            'geo': {
                'brdrd': bernoulli(1.0/8),
                'cnvex': bernoulli(1.0/8),
                'ncnvx': bernoulli(2.0/8), # 2.0 counters overruling by others
                'prdct': bernoulli(1.0/16),
            },
            'top': {
                'kconn': bernoulli(3.0/4),
                'fconn': bernoulli(1.0/4),
                'simpl': bernoulli(1.0/8),
                'nsmpl': bernoulli(2.0/8), # 2.0 counters overruling by others
            },
        }

        #-------------  0.0.2 prevent constraint conflicts  ------------------#

        if (self.side<=2 or 
            self.side<=3 and self.constraints['top']['nsmpl'] or
            self.constraints['geo']['cnvex'] or 
            sum(int(self.constraints[con][nm])*w for con, nm, w in
                (('geo', 'brdrd', 1.0),
                 ('geo', 'prdct', 1.6),
                 ('top', 'kconn', 0.4),
                 ('top', 'fconn', 0.4),
                 ('top', 'simpl', 1.0),)) >= 2.0
            ):
            self.constraints['geo']['ncnvx'] = 0

        if (self.side<=2 or
            self.constraints['geo']['cnvex'] or
            self.constraints['geo']['prdct'] or
            self.constraints['top']['simpl']):
            self.constraints['top']['nsmpl'] = 0

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #~~~~~~~~~  0.1 Rejection-Sample Blocks (Approximately)  ~~~~~~~~~~~~~~~~~#

    def search(self):
        '''
        '''
        while True:
            arr = self.propose()

            #---------  0.1.0 retry if doesn't span height or width  ---------#

            inhab_rows = np.nonzero(np.sum(arr, axis=0)) 
            inhab_cols = np.nonzero(np.sum(arr, axis=1))
            rmin, rmax = np.amin(inhab_rows), np.amax(inhab_rows)
            cmin, cmax = np.amin(inhab_cols), np.amax(inhab_cols)
            if ((rmin, rmax)!=(0, self.side-1) and
                (cmin, cmax)!=(0, self.side-1)):
                continue

            #---------  0.1.1 retry if doesn't meet specifications  ----------#

            if not self.passes_all_sym_reqs(arr): continue
            if not self.passes_all_geo_reqs(arr): continue
            if not self.passes_all_top_reqs(arr): continue

            return arr

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #~~~~~~~~~  0.2 Constraint-Guided Block Proposal  ~~~~~~~~~~~~~~~~~~~~~~~~#

    def propose(self):
        '''
        '''
        #-------------  0.2.0 sample block, sparsely if constrained  ---------# 

        base_prob = 1.0 / (
              2.0
            * 2.5       ** sum(self.constraints['sym'].values())
            * 4.0       **     self.constraints['geo']['brdrd']
            * self.side **     self.constraints['geo']['cnvex']
            * self.side ** (   self.constraints['geo']['prdct'])
            * 6.0       ** max(self.constraints['top']['kconn'],
                             self.constraints['top']['fconn'])
            * self.side **     self.constraints['top']['simpl']
        )  
        arr = np.random.binomial(1, base_prob, (self.side, self.side))

        #-------------  0.2.1 ensure block is nonempty  ----------------------#

        arr[np.random.randint(self.side), np.random.randint(self.side)] = 1

        #-------------  0.2.2 adjust toward symmetry requirements  -----------#

        for axis in Block.transfs_by_axis:
            if not self.passes_sym_req(arr, axis):
                self.make_more_sym(arr, axis)

        #-------------  0.2.3 adjust toward geo requirements  -----------#

        if not self.passes_geo_req(arr, 'brdrd'): self.make_more_brdrd(arr)
        if not self.passes_geo_req(arr, 'cnvex'): self.make_more_cnvex(arr)
        if not self.passes_geo_req(arr, 'prdct'): self.make_more_prdct(arr)

        #-------------  0.2.4 adjust toward top requirements  -----------#

        if not (self.passes_top_req(arr, 'kconn') and
                self.passes_top_req(arr, 'fconn')):
            self.make_more_connected(arr)
        if not self.passes_top_req(arr, 'simpl'):
            self.make_more_cnvex(arr)

        return arr

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #~~~~~~~~~  0.3 Pre- and Post- Processing of Sampled Arrays  ~~~~~~~~~~~~~#

    def check_sides(self, arr): 
        '''
        '''
        pre(arr.shape == (self.side, self.side), 'expected side x side array') 

    def colored(self, arrs):
        '''
        '''
        return str(CC + '\n'.join(
            ' '.join(
                '|' + ''.join(
                    ('@M []@D ' if arr[r,c] else '@W   @D ')
                    for c in range(self.side)
                ) + '|'
                for arr in arrs
            )
            for r in range(self.side)
        ))

    #=========================================================================#
    #=  1. IMPLEMENT SYMMETRY CONCEPTS  ======================================#
    #=========================================================================#

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #~~~~~~~~~  1.0 Check Symmetry Predicates  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    transfs_by_axis = {
        'verti': (lambda arr: arr[:, ::-1]                 ),
        'horiz': (lambda arr: arr[::-1, :]                 ),
        'slash': (lambda arr: np.transpose(arr[::-1])[::-1]),
        'blash': (lambda arr: np.transpose(arr)            ),
    }

    def check_sym_pred_by_nm(self, arr, axis): 
        '''
        '''
        self.check_sides(arr)
        return np.array_equal(arr, Block.transfs_by_axis[axis](arr))

    def passes_sym_req(self, arr, axis):
        '''
        '''
        return (
            not self.constraints['sym'][axis] or
            self.check_sym_pred_by_nm(arr, axis)
        )

    def passes_all_sym_reqs(self, arr): 
        '''
        '''
        for axis in Block.transfs_by_axis:
            if not self.passes_sym_req(arr, axis):
                return False
        return True

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #~~~~~~~~~  1.1 Ensure Symmetry Predicates are Satisfied  ~~~~~~~~~~~~~~~~#

    def make_more_sym(self, arr, axis):
        '''
        '''
        if self.check_sym_pred_by_nm(arr, axis):
            return
        arr[:] = np.maximum(arr, Block.transfs_by_axis[axis](arr))  

    #=========================================================================#
    #=  2. IMPLEMENT GEOMETRY CONCEPTS  ======================================#
    #=========================================================================#

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #~~~~~~~~~  2.0 Check Geometry Predicates  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    def check_brdrd(self, arr):
        '''
        '''
        inhab_rows = np.nonzero(np.sum(arr, axis=1)) 
        inhab_cols = np.nonzero(np.sum(arr, axis=0))
        rmin, rmax = np.amin(inhab_rows), np.amax(inhab_rows)
        cmin, cmax = np.amin(inhab_cols), np.amax(inhab_cols)

        for edge in (arr[rmin:rmax+1,cmin], arr[rmin,cmin:cmax+1]):
            for hasendpt in (np.amin, np.amax):
                if not hasendpt(edge):
                    return False 
        return True

    def check_cnvex(self, arr):
        '''
        '''
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

    def check_ncnvx(self, arr):
        '''
        '''
        return not self.check_cnvex(arr)  

    def check_prdct(self, arr):
        '''
        '''
        proj_by_row = np.sum(arr, axis=0) 
        proj_by_col = np.sum(arr, axis=1) 
        reconstruction = np.minimum(1, np.outer(proj_by_row, proj_by_col))
        if not np.array_equal(arr, reconstruction):
            return False
        return True

    geo_preds_by_nm = {
        'brdrd': check_brdrd,
        'cnvex': check_cnvex,
        'ncnvx': check_ncnvx,
        'prdct': check_prdct,
    }

    def passes_geo_req(self, arr, pred_nm):
        '''
        '''
        return (
            not self.constraints['geo'][pred_nm] or
            Block.geo_preds_by_nm[pred_nm](self, arr)
        )

    def passes_all_geo_reqs(self, arr): 
        '''
        '''
        for pred_nm in Block.geo_preds_by_nm:
            if not self.passes_geo_req(arr, pred_nm):
                return False
        return True

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #~~~~~~~~~  2.1 Make Geometry Predicates More Likely ~~~~~~~~~~~~~~~~~~~~~#

    def make_more_brdrd(self, arr):
        '''
        '''
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

    def make_more_cnvex(self, arr):
        '''
            Mutate array by introducing all rook-trapping cells, thus ensuring
            convexity with respect to rook moves but not necessarily convexity
            overall.
        '''
        if self.check_cnvex(arr):
            return

        for r in range(self.side):
            for c in range(self.side):
                if arr[r,c]: continue
                if sum(arr[:r, c])*sum(arr[r:,c ]):
                    arr[max(0,r-1):min(self.side,r+1), c] = 1
                if sum(arr[ r,:c])*sum(arr[r ,c:]):
                    arr[r, max(0,c-1):min(self.side,c+1)] = 1

    def make_more_prdct(self, arr):
        '''
        '''
        if self.check_prdct(arr):
            return
        proj_by_row = np.sum(arr, axis=0) 
        proj_by_col = np.sum(arr, axis=1)
        arr[:] = np.minimum(1, np.outer(proj_by_row, proj_by_col))

    #=========================================================================#
    #=  3. IMPLEMENT TOPOLOGY CONCEPTS  ======================================#
    #=========================================================================#

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #~~~~~~~~~  3.0 Define Parameterized Notion of Connectedness  ~~~~~~~~~~~~#

    #-----------------  3.0.0 when is a displacement neighborly?  ------------#

    offset_preds_by_top_pred_nm = {
        'kconn': lambda dr, dc: max(abs(dr),abs(dc))==1,
        'fconn': lambda dr, dc: abs(dr)+abs(dc)==1,     
        'simpl': lambda dr, dc: abs(dr)+abs(dc)==1,
        'nsmpl': lambda dr, dc: None,
    }

    def check_connected(self, arr, offset_pred):
        '''
            Check whether the cells in the block described, with two connected
            when their offset obeys the offset predicate, yields a connected
            graph.
        '''
        #-------------  3.0.1 prepare points  --------------------------------#

        points = set((r,c) for r in range(arr.shape[0])
                     for c in range(arr.shape[1]) if arr[r,c])
        if not points:
            return True

        #-------------  3.0.2 prepare notion of neighbor  --------------------#

        offsets = [
            (dr,dc) for dr in range(-1,2) for dc in range(-1,2)
            if offset_pred(dr, dc)
        ] 

        #-------------  3.0.3 check connectedness by BFS  --------------------#

        seen = set([min(points)])
        frontier = seen # always a subset of seen
        while frontier:
            neighbors = set([(r+dr,c+dc) for (r,c) in frontier
                             for (dr, dc) in offsets]).intersection(points)
            frontier = neighbors.difference(seen)
            seen.update(frontier)

        return len(seen) == len(points)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #~~~~~~~~~  3.1 Check Topology Predicates  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    def check_top_req(self, arr, nm):
        '''
        '''
        if nm=='nsmpl':
            return not self.check_top_req(arr, 'simpl')
        elif nm=='simpl':

            #---------  3.1.0 simply connected iff S2 complement connected  --#

            new_arr = np.ones((self.side+2, self.side+2))
            new_arr[1:self.side+1, 1:self.side+1] = 1-arr
            arr = new_arr
        return self.check_connected(arr, Block.offset_preds_by_top_pred_nm[nm])

    def passes_top_req(self, arr, pred_nm):
        '''
        '''
        return (
            not self.constraints['top'][pred_nm] or
            self.check_top_req(arr, pred_nm)
        )

    def passes_all_top_reqs(self, arr): 
        '''
        '''
        for pred_nm in Block.offset_preds_by_top_pred_nm:
            if not self.passes_top_req(arr, pred_nm):
                return False
        return True

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #~~~~~~~~~  3.2 Make Topology Predicates More Likely ~~~~~~~~~~~~~~~~~~~~~#

    def make_more_connected(self, arr):
        '''
            Mutate array by adding ferz neighbors, thus potentially merging
            connected components and improving connectedness.
        '''
        if self.check_top_req(arr, 'fconn'):
            return

        #-------------  3.2.0 prepare random notion of neighbor  -------------#

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

        #-------------  3.2.1 add random neighbors to block  -----------------#

        for r in range(self.side):
            for c in range(self.side):
                arr[r,c] = max(arr[r, c], min(1, 
                    sum(is_neighbor(r,c,dr,dc) for dr,dc in offsets)
                ))

#=============================================================================#
#=====  4. SAMPLE and DISPLAY BLOCK TYPES  ===================================#
#=============================================================================#

nb_classes = 5
text_width = 120

for _ in range(nb_classes): 
    B = Block()
    print(CC + '@Y {} @D '.format(str(B.constraints)))
    print(CC + '@D {} @D '.format(str(B.side)))
    print(B.colored([B.search() for _ in range(text_width//(3+2*B.side))]))
    print()
