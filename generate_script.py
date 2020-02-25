''' author: samtenka
    change: 2020-02-20
    create: 2020-02-16
    descrp: generate a script of type grid x grid <-- noise, factored into
            components of type 
                Z_s : blocks <-- noise
                Z_b : block  <-- blocks
                Z_g : grid   <-- block <-- blocks <-- noise
                X   : grid   <-- grid <-- block <-- noise 
                Y   : grid   <-- grid <-- block             
            The overall map is then defined as
                lambda n:
                    (lambda s:
                        (lambda b:
                            (lambda g:
                                (X n b g, Y b g)
                            )(Z_g n b s)
                        )(Z_b s)
                    )(Z_s n)
    to use: call  
                get_script()
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
    'gen_some': tInt.frm(tNoise),                      # [1,2,3]
    'gen_svrl': tInt.frm(tNoise),                      # [3,4,5]
    #'gen_many': tInt.frm(tNoise),                      # [15...30] 
    'gen_cell': tCell.frm(tGrid).frm(tNoise),          # uniform cell
    'gen_crnr': tCell,                                 # upper left cell
    'gray'    : tColor,                                # 'A'
    'gen_rain': tColor.frm(tNoise),                    # GENERIC_COLORS
    'gen_shap': tShape.frm(tInt).frm(tNoise),          # get random shape 
    # product types:
    'blok_cons': tBlock.frm(tShape).frm(tColor),
    'shap_blok': tShape.frm(tBlock),
    'colr_blok': tColor.frm(tBlock),
    # render a grid:
    'fill_grd':  tGrid.frm(tCell).frm(tColor).frm(tGrid),
    'noise_grd': tGrid.frm(tColor).frm(tNoise).frm(tGrid),      # sparse noise
    'blnk_grd':  tGrid.frm(tInt).frm(tInt),                     # black with given height, width
    'rndr_blks': tGrid.frm(tBlock.s()).frm(tGrid).frm(tNoise),
    'rndr_blk':  tGrid.frm(tBlock).frm(tGrid),                  # put block in upper right corner
    ## numerical concepts:
    'volume_shap':  tInt.frm(tShape),
    'height_shap':  tInt.frm(tShape),
    'width_shap':  tInt.frm(tShape),
    'amax_blocks': tArgmax_(tBlock),
    #'amax_shapes': tArgmax_(tShape),
    ## list helpers:
    #'sing_cols': tColor.s().frm(tColor),
    #'cons_cols': tColor.s().frm(tColor).frm(tColor.s()),
    'sing_blks': tBlock.s().frm(tBlock),
    'gen_blks': tBlock.s().frm(tNoise).frm(tBlock.frm(tNoise)).frm(tInt),
    'cons_blks': tBlock.s().frm(tBlock).frm(tBlock.s()),
    'uniq_blks': tBlock.s().frm(tBlock.s()),
}

var_count = 0
def get_fresh():
    global var_count
    var_count += 1
    return 'x'+str(var_count)

verbose = False

def construct(goal, resources):
    if verbose:
        print('analyzing {}'.format(str(goal)))

    if bernoulli(1e-5):
        # random timeout
        return

    if bernoulli(0.1): # split
        split = uniform([
            tInt, tColor, tShape, tShapes, tBlock, tBlocks, tGrid
        ])
        arg  = construct(split, resources)

        var_nm = get_fresh()
        resources = {k:v for k,v in resources.items()}
        resources[var_nm] = split

        body = construct(goal, resources)
        return {
            'text':'{}(({}->{})({}))'.format(var_nm, body['text'], arg['text']),
            'pyth': '(lambda {}: {})({})'.format(var_nm, body['pyth'], arg['pyth'])
        }
    elif goal.kind=='from' and bernoulli(0.8): # destruct
        var_nm = get_fresh()
        resources = {k:v for k,v in resources.items()}
        resources[var_nm] = goal.arg
        body = construct(goal.out, resources)
        return {
            'text': '({}->{})'.format(var_nm, body['text']),
            'pyth': '(lambda {}: {})'.format(var_nm, body['pyth'])
        }
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
            hypoths = [construct(h, resources) for h in hypoths]
            pretty_nm = str(CC+'@D '+nm+'@P ')

            applied = nm
            for h in hypoths[::-1]:
                applied = '{}({})'.format(applied, h['pyth'])

            return {
                'text': (
                    pretty_nm if not hypoths else
                    '({} {})'.format(pretty_nm, ' '.join(h['text'] for h in hypoths[::-1]))
                ),
                'pyth': applied, 
            }
        else:
            # no match found
            return

def tenacious_construct(goal, add_resources={}):
    global var_count

    resources = {k:v for k,v in sigs_by_nm.items()}
    for k,v in add_resources.items():
        resources[k] = v

    for i in range(10**4):
        if verbose:
            print()
            print()
        try:
            code = construct(goal, resources)
            pre(code is not None, '')
            return code
        except:
            var_count=0
            continue

def get_script(): 
    blocks  = tenacious_construct(tBlock.s(), {'noise':tNoise})
    getblock= tenacious_construct(tBlock,     {'blocks':tBlock.s()})
    Y       = tenacious_construct(tGrid,      {'block':tBlock})
    blocks  ['pyth'] = 'lambda noise: {}'.format(blocks  ['pyth'])
    getblock['pyth'] = 'lambda blocks: {}'.format(getblock['pyth'])
    Y       ['pyth'] = 'lambda block: {}'.format(Y['pyth'])

    X = {'pyth':'', 'text':''}

    X['pyth'] = 'lambda noise: lambda blocks: rndr_blks(noise)(blnk_grd(10)(10))(blocks)'
    X['text'] = '(rndr_blks noise (blnk_grd 10 10) blocks)'

    print(str(CC+'@O blocks = @P '), blocks  ['text'])
    print(str(CC+'@O block  = @P '), getblock['text'] + ' block')
    print(str(CC+'@O X      = @P '), X       ['text'])
    print(str(CC+'@O Y      = @P '), Y       ['text'])

    return blocks, getblock, X, Y

if __name__=='__main__':
    get_script()

#    #X       = tenacious_construct(tGrid,      {'noise':tNoise, 'blocks':tBlock.s(),                })
#    #Y       = tenacious_construct(tGrid,      {                                     'block':tBlock,})
#    #X       ['pyth'] = 'lambda noise: lambda blocks: {}              '.format(X       ['pyth'])
#    #Y       ['pyth'] = '                             lambda block: {}'.format(Y       ['pyth'])


# future:
#'one'     : tInt,                                  # 1 
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
