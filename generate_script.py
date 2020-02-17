''' author: samtenka
    change: 2020-02-16
    create: 2020-02-16
    descrp: generate a script of type Grid x Grid <--()
    to use: import
                from lg_types import LGType, tUnit, tInt, tPos, tBool,
                                     tColor, tGrid
'''

from utils import CC, pre                               # ansi
from utils import secs_endured, megs_alloced            # profiling
from utils import reseed, bernoulli, geometric, uniform # math

from lg_types import tInt, tCell, tColor, tBlock, tGrid, tDir
from lg_types import tCount_, tFilter_, tArgmax_, tMap_, tRepeat_

#=============================================================================#
#=====  0. PROVER  ===========================================================#
#=============================================================================#

signatures_by_nm = {
    # basic samplers:
    'make_sml': tInt,
    'make_big': tInt,
    'make_cel': tCell,
    'make_col': tColor,
    'make_dir': tDir,
    'make_blk': tBlock.frm(tInt),
    # render a grid:
    'blnk_grd': tGrid.frm(tInt).frm(tInt),
    'rndr_blk': tGrid.frm(tBlock).frm(tCell).frm(tColor).frm(tGrid),
    'rndr_blks': tGrid.frm(tBlock.s()).frm(tCell.s()).frm(tColor.s()).frm(tGrid),
    # numerical concepts:
    'size_blks': tCount_(tBlock),
    'size_blk': tInt.frm(tBlock),
    'amax_blks': tArgmax_(tBlock),
    # list helpers:
    'sing_cols': tColor.s().frm(tColor),
    'cons_cols': tColor.s().frm(tColor).frm(tColor.s()),
    'sing_cels': tCell.s().frm(tCell),
    'cons_cels': tCell.s().frm(tCell).frm(tCell.s()),
    'sing_blks': tBlock.s().frm(tBlock),
    'cons_blks': tBlock.s().frm(tBlock).frm(tBlock.s()),
}

var_count = 0
def get_fresh():
    global var_count
    var_count += 1
    return str(var_count)

verbose = False

def construct(goal, resources):
    if verbose:
        print('analyzing {}'.format(str(goal)))

    if bernoulli(0.001):
        pre(False, 'timeout')

    if bernoulli(0.1):
        split = uniform([tInt, tCell, tColor, tBlock, tGrid])
        if bernoulli(0.5):
            split = split.s()
        arg  = construct(split, resources)

        var_nm = get_fresh()
        resources = {k:v for k,v in resources.items()}
        resources[var_nm] = split

        body = construct(goal, resources)
        return '(({}->{})({}))'.format(var_nm, body, arg)

    elif goal.kind=='prod':
        fst = construct(goal.fst, resources)
        snd = construct(goal.snd, resources)
        return '({}; {})'.format(fst, snd)
    else:
        matches = [
            (nm,sig,hypoths) for nm,sig in resources.items()
            for conseqs, hypoths in sig.conseq_hypoth_pairs()
            if goal in conseqs
        ] 
        if matches:
            nm, sig, hypoths = uniform(matches) 
            if verbose:
                print('matched {} with {}'.format(goal, nm))
            hypoths = [' '+construct(h, resources) for h in hypoths]
            return '({}{})'.format(
                nm, ''.join(hypoths)
            )
        else:
            pre(False, 'no match found')

def tenacious_construct(goal):
    global var_count
    while True:
        try:
            code = construct(goal, resources=signatures_by_nm)
            pre(50 <= len(code) <= 500, 'code too long or short')
            return code
        except:
            var_count=0
            continue

print(tenacious_construct(tGrid.times(tGrid)))
