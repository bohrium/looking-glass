''' author: samtenka
    change: 2020-03-04
    create: 2019-02-26
    descrp: translate textual code to python function or program tree 
    to use: 
'''

from utils import InternalError                         # maybe
from utils import CC, pre                               # ansi
from utils import secs_endured, megs_alloced            # profiling

from parse import Parser, str_from_tree_flat, str_from_tree
from fit_weights import WeightLearner
from resources import PrimitivesWrapper
from vis import str_from_grids, render_color


where = ''
def evaluate_tree(tree, resources, depth=0):
    global where 
    tab = '|  '*depth
    if type(tree)==type(''):
        rtrn = resources[tree][0] # get implementation instead of type
        if type(rtrn)==type('') and (rtrn in resources):
            where += '{}resource @B {} \n@D '.format(tab, str_from_tree_flat(tree))
            rtrn = evaluate_tree(rtrn, resources, depth+1)
        return rtrn
    elif type(tree)==type({}):
        where += '{}lambda @G {} \n@D '.format(tab, str_from_tree_flat(tree))
        for (var_nm, var_type), body in tree.items():
            return (lambda x:
                evaluate_tree(body, {
                    k:v for k,v in list(resources.items())+[(var_nm, (x, var_type))]
                }, depth+1)
            )
    else:
        caller, args = tree[0], tree[1:]
        where += '{}caller @P {} \n@D '.format(tab, str_from_tree_flat(caller))
        partial = evaluate_tree(caller, resources, depth+1)
        for arg in args:
            where += '{}arg @O {} \n@D '.format(tab, str_from_tree_flat(arg))
            aa = evaluate_tree(arg, resources, depth+1)
            #where += '{}appl @R \n@D '.format(tab)
            partial = partial(aa)
        return partial

if __name__=='__main__':
    #CODE_FILE_NM = 'manual.003.arcdsl'
    #CODE_FILE_NM = 'manual.006.arcdsl'
    #CODE_FILE_NM = 'manual.007.arcdsl'
    #CODE_FILE_NM = 'manual.008.arcdsl'
    #CODE_FILE_NM = 'manual.016.arcdsl'
    #CODE_FILE_NM = 'manual.022.arcdsl'
    #CODE_FILE_NM = 'manual.023.arcdsl'
    #CODE_FILE_NM = 'manual.032.arcdsl'
    CODE_FILE_NM = 'manual.034.arcdsl'
    with open(CODE_FILE_NM) as f:
        code = f.read()
    print(CC+'parsing @P {}@D ...'.format(CODE_FILE_NM))
    P = Parser(code)
    t = P.get_tree()
    print(CC+'@P {} @D '.format(str_from_tree(t)))

    print(CC+'sampling from @P {}@D ...'.format(CODE_FILE_NM))
    primitives = PrimitivesWrapper().primitives
    for I in range(100):
       try:
           where = ''
           x,y = evaluate_tree(t, primitives)
           print(I)
           print(CC+str_from_grids([x.colors, y.colors], render_color))
           break
       except InternalError:
           continue
       except TypeError as e:
           print(CC+where)
           print(e)
           break

