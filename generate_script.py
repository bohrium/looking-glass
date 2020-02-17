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

from lg_types import tInt, tCell, tColor, tShape, tBlock, tGrid, tDir, tNoise
from lg_types import tCount_, tFilter_, tArgmax_, tMap_, tRepeat_

#=============================================================================#
#=====  0. PROVER  ===========================================================#
#=============================================================================#

sigs_by_nm = {
    # basic samplers:
    #'one'     : tInt.frm(tNoise),                     # [1]
    'gen_some': tInt.frm(tNoise),                      # [1,2,3]
    'gen_svrl': tInt.frm(tNoise),                      # [3,4,5]
    'gen_many': tInt.frm(tNoise),                      # [15...30] 
    'gen_cell': tCell.frm(tGrid).frm(tNoise),          # uniform cell
    'gray'    : tColor,                                # 'A'
    'gen_rain': tColor.frm(tNoise),                    # GENERIC_COLORS
    'gen_shap': tShape.frm(tInt).frm(tNoise),          # get random shape 
    # product types:
    'blok_cons': tBlock.frm(tShape).frm(tColor),
    'shap_blok': tShape.frm(tBlock),
    'colr_blok': tColor.frm(tBlock),
    # render a grid:
    'fill_grd':  tGrid.frm(tCell).frm(tColor),
    'noise_grd': tGrid.frm(tColor).frm(tNoise).frm(tGrid),      # sparse noise
    'blnk_grd':  tGrid.frm(tInt).frm(tInt),                     # black with given height, width
    'rndr_blks': tGrid.frm(tBlock.s()).frm(tGrid).frm(tNoise),
    'rndr_blk':  tGrid.frm(tBlock).frm(tGrid),                  # put block in upper right corner
    ## numerical concepts:
    'size_shap':  tInt.frm(tShape),
    'amax_blocks': tArgmax_(tBlock),
    #'amax_shapes': tArgmax_(tShape),
    ## list helpers:
    #'sing_cols': tColor.s().frm(tColor),
    #'cons_cols': tColor.s().frm(tColor).frm(tColor.s()),
    'gen_blks': tBlock.s().frm(tNoise).frm(tBlock.frm(tNoise)).frm(tInt),
    'cons_blks': tBlock.s().frm(tBlock).frm(tBlock.s()),
}

var_count = 0
def get_fresh():
    global var_count
    var_count += 1
    return str(var_count)

verbose = False
lines = 0 

def construct(goal, resources):
    global lines
    if verbose:
        print('analyzing {}'.format(str(goal)))
        lines += 1

    if bernoulli(1e-4):
        pre(False, 'timeout')

    if bernoulli(0.1): # split
        split = uniform([
            tInt, tColor, tShape, tShapes, tBlock, tBlocks, tGrid
        ])
        arg  = construct(split, resources)

        var_nm = get_fresh()
        resources = {k:v for k,v in resources.items()}
        resources[var_nm] = split

        body = construct(goal, resources)
        return '(({}->{})({}))'.format(var_nm, body, arg)
    elif goal.kind=='from' and bernoulli(0.8): # destruct
        var_nm = get_fresh()
        resources = {k:v for k,v in resources.items()}
        resources[var_nm] = goal.arg
        body = construct(goal.out, resources)
        return '({}->{})'.format(var_nm, body)
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
                lines += 1
            hypoths = [' '+construct(h, resources) for h in hypoths]
            return (
                nm if not hypoths else
                '({}{})'.format(nm, ''.join(hypoths))
            )
        else:
            pre(False, 'no match found')

def tenacious_construct(goal, add_resources={}):
    global var_count, lines

    resources = {k:v for k,v in sigs_by_nm.items()}
    for k,v in add_resources.items():
        resources[k] = v

    for i in range(10**3):
        if verbose:
            print()
            print()
        try:
            code = construct(goal, resources)
            #pre(50 <= len(code) <= 500, 'code too long or short')
            return code
        except:
            var_count=0
            #print(CC+('@^ '+' '*100)*(lines+3))
            lines = 0 
            continue

print('blocks:'  , tenacious_construct(tBlock.s(), {'noise':tNoise,                                                  }))
print('getblock:', tenacious_construct(tBlock,     {                'blocks':tBlock.s()                              }))
print('grid:'    , tenacious_construct(tGrid,      {'noise':tNoise, 'blocks':tBlock.s(), 'block':tBlock,             }))
print('X:'       , tenacious_construct(tGrid,      {'noise':tNoise,                      'block':tBlock, 'grid':tGrid}))
print('Y:'       , tenacious_construct(tGrid,      {                                     'block':tBlock, 'grid':tGrid}))

# future:
#'fit_shap': tCell.frm().frm(tGrid).frm(tNoise),    # get fitting location for shape 
#'rndr_blk': tGrid.frm(tBlock).frm(tCell).frm(tGrid),
#'gen_crnr': tCell.frm(tGrid)                       # upper left cell
#'rndr_blk': tGrid.frm(tBlock).frm(tGrid),
#'size_blks': tCount_(tBlock),
#'sing_cels': tCell.s().frm(tCell),
#'cons_cels': tCell.s().frm(tCell).frm(tCell.s()),
#'gen_card': tDir.frm(tNoise),                      # [ferz directions] 
#'black'   : tColor,                                # 'K'

#elif goal.kind=='prod':
#    fst = construct(goal.fst, resources)
#    snd = construct(goal.snd, resources)
#    return '({}; {})'.format(fst, snd)

