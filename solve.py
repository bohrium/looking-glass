''' author: samtenka
    change: 2020-02-26
    create: 2019-02-26
    descrp: translate textual code to python function or program tree 
    to use: 
'''

from utils import InternalError                         # maybe
from utils import CC, pre                               # ansi
from utils import secs_endured, megs_alloced            # profiling

from parse import Parser
from fit_weights import WeightLearner
from resources import PrimitivesWrapper
from vis import str_from_grids, render_color


def evaluate_tree(tree, resources):
    if type(tree)==type(''):
        rtrn = resources[tree][0] # get implementation instead of type
        if type(rtrn)==type('') and (rtrn in resources):
            rtrn = evaluate_tree(rtrn, resources)
        return rtrn
    elif type(tree)==type({}):
        for (var_nm, var_type), body in tree.items():
            return (lambda x:
                evaluate_tree(body, {
                    k:v for k,v in list(resources.items())+[(var_nm, (x, var_type))]
                })
            )
    else:
        caller, args = tree[0], tree[1:]
        partial = evaluate_tree(caller, resources)
        for arg in args:
            partial = partial(evaluate_tree(arg, resources))
        return partial

if __name__=='__main__':
    CODE_FILE_NM = 'manual.003.arcdsl'
    CODE_FILE_NM = 'hello.arcdsl'
    with open(CODE_FILE_NM) as f:
        code = f.read()
    print(CC+'parsing @P {}@D ...'.format(CODE_FILE_NM))
    P = Parser(code)
    t = P.get_tree()

    print(CC+'sampling from @P {}@D ...'.format(CODE_FILE_NM))
    primitives = PrimitivesWrapper().primitives
    while True:
       try:
           x,y = evaluate_tree(t, primitives)
           print(CC+str_from_grids([x.colors, y.colors], render_color))
           break
       except InternalError:
           continue

    WL = WeightLearner()
    WL.observe_tree(t)
    WL.compute_weights()
    predictions = WL.predict('root', set([]))
    for k,v in sorted(predictions.items(), reverse=True, key=lambda xy: xy[1]):
        print(CC+'@R {} @G {}'.format(k,v))


