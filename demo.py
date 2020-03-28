''' author: samtenka
    change: 2020-03-11
    create: 2019-02-26
    descrp: 
    to use: 
'''

import glob
import numpy as np

from utils import InternalError                         # maybe
from utils import CC, pre                               # ansi
from utils import secs_endured, megs_alloced            # profiling
from utils import paths                                 # paths    

from parse import Parser, str_from_tree_flat, str_from_tree
from grid import Grid
from resources import PrimitivesWrapper
from vis import str_from_grids, render_color

def print_grids(grids):
    print(CC+str_from_grids([
        z.colors for z in grids
    ], render_color))

where = ''
samples = {}

def evaluate_tree(tree, resources, depth=0, show_exec=False):
    global where, samples
    if not show_exec:
        where = ''
        samples={}

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
        caller, arg_trees = tree[0], tree[1:]
        where += '{}caller @P {} \n@D '.format(tab, str_from_tree_flat(caller))
        partial = evaluate_tree(caller, resources, depth+1)
        arg_vals = [] 
        for at in arg_trees :
            where += '{}arg @O {} \n@D '.format(tab, str_from_tree_flat(at))
            arg_vals.append(evaluate_tree(at, resources, depth+1))
        if caller.startswith('split') and type(arg_trees[1])==dict:
            name = list(arg_trees[1].keys())[0][0]
            samples[name] = arg_vals[0]
        for av in arg_vals:
            partial = partial(av)
        return partial

def demonstrate(file_nm, print_text=True, nb_rows=1, nb_cols=1, nb_tries=10, show_exec=True):
    global where, samples
    pre(nb_cols==1 or not show_exec, 'when showing exec, need nb_cols to be 1')
    with open(file_nm) as f:
        code = f.read()
    print(CC+'parsing @P {}@D ...'.format(file_nm))
    P = Parser(code)
    t = P.get_tree()

    if print_text:
        print(CC+'@P {} @D '.format(str_from_tree(t)))

    print(CC+'sampling from @P {}@D ...'.format(file_nm))
    primitives = PrimitivesWrapper().primitives

    for _ in range(nb_rows):
        xys = [] 
        for _ in range(nb_cols):
            for _ in range(nb_tries):
                where = ''
                samples = {}
                try:
                    xys += list(evaluate_tree(t, primitives, show_exec=True))
                    xys += [primitives['new_grid'][0](0)(0)]
                    break
                except InternalError:
                    continue
                except Exception as e:
                    print(CC+'@R error@D ')
                    print(CC+where)
                    print(e)
                    break

            if show_exec:
                for nm, val in samples.items():
                    if type(val)==Grid:
                        print(CC+'@O {}@D : {} : '.format(nm, type(val)))
                        print_grids([val])
                    elif type(val)==np.array:
                        print(CC+'@O {}@D : {} : '.format(nm, type(val)))
                        print_grids([primitives['monochrome'](val, 'P')])
                    elif type(val)==str:
                        print(CC+'@O {}@D : {} : '.format(nm, type(val)))
                        grid = Grid(1, 1)
                        grid.colors[0][0] = val
                        print_grids([grid])
                    elif type(val)==int:
                        print(CC+'@O {}@D : {} : '.format(nm, type(val)))
                        print(val)

        print_grids(xys)

if __name__=='__main__':
    file_nms = paths('manual')
    for fnm in file_nms:
        print(fnm)
        demonstrate(fnm, print_text=False, show_exec=False, nb_rows=2, nb_cols=2)
        input(CC+'@O next?@D ')

