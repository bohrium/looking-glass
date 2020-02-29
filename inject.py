''' author: samtenka
    change: 2020-02-28
    create: 2019-02-28
    descrp: translate textual code to python function or program tree 
    to use: 
            To demonstrate verbosely on some artificial trees, run
                python inject.py -v 
            Remove that flag for non-verbose testing. 
'''

import sys
import types

from utils import InternalError                         # maybe
from utils import CC, pre                               # ansi
from utils import secs_endured, megs_alloced            # profiling

from parse import Parser
from fit_weights import WeightLearner
from resources import PrimitivesWrapper
from vis import str_from_grids, render_color

class DependencyAnalyzer:
    def __init__(self, sensitives, verbose=False):
        self.sensitives = sensitives
        self.verbose = verbose

    def report(self, tag, dst, src, env, depth):
        ''' Print the small-step substitution specified in arguments. '''
        if not self.verbose: return
        print(CC + '  '*depth +
            ('@D {} @O {} @D from @P {} @D in environ @B {} @D '.format(
                tag, dst, src, env
            )
        ))

    def abstract_eval(self, tree, environ={}, depth=0): 
        '''
            Abstractly interpret `tree` to find subset of leaves in
            self.sensitives that could affect tree's value.
        '''
        if type(tree) in (set, types.FunctionType):
            return tree
        elif type(tree) == str:
            if tree in sensitives:
                return {tree}
            elif tree in environ:
                impl = environ[tree][0]
                return self.abstract_eval(impl, environ, depth+1)
            else:
                return set()
        elif type(tree) == dict:
            for (var_nm, var_type), body in tree.items():
                return (lambda x:
                    self.abstract_eval(body, {
                        k:v for k,v in
                        list(environ.items())+[(var_nm, (x, var_type))]
                    }, depth+1)
                )
        elif type(tree) == list:
            head, args = tree[0], tree[1:]

            if head=='pair':
                return (
                    self.abstract_eval(args[0], environ, depth+1),
                    self.abstract_eval(args[1], environ, depth+1)
                )
            elif head=='fst':
                return self.abstract_eval(args[0], environ, depth+1)[0]
            elif head=='snd':                                          
                return self.abstract_eval(args[0], environ, depth+1)[1]
            else:
                partial = self.abstract_eval(head, environ, depth+1)
                self.report('|cal', partial, head, environ, depth)
                for arg in args:
                    aa = self.abstract_eval(arg, environ, depth+1)
                    self.report('|arg', aa, arg, environ, depth)
                    partial = (
                        partial.union(aa) if type(partial)==set else
                        partial(aa) if type(partial)==types.FunctionType 
                        else pre(False, 'argument should be set or function!')
                    )
                self.report(':ans', partial, tree, environ, depth)
                return partial
        else:
            pre(False, 'tree is of unrecognized type!')

if __name__=='__main__':
    verbose = (len(sys.argv)==2 and sys.argv[1] == '-v')

    environ = {}
    sensitives={'NOISE', 'MORE_NOISE'}
    IA = DependencyAnalyzer(sensitives, verbose)
        
    tree_depset_pairs = [
        #(
        #    ['pair', 'MORE_NOISE', 'NOISE'],
        #    (set(['MORE_NOISE']), set(['NOISE']))
        #),
        (
            ['fst', ['pair', 'MORE_NOISE', 'NOISE']],
            set(['MORE_NOISE'])
        ),
        #(
        #    ['MORE_NOISE'],
        #    set(['MORE_NOISE'])
        #),
        #(
        #    ['rac', [{('x', 0): ['cow']}, 'MORE_NOISE'], 'NOISE', 'cow'],
        #    set(['NOISE'])
        #),
        #(
        #    [
        #        {('x', None): {('y', None): ['cow', 'x', 'y']}},
        #        'MORE_NOISE', 'NOISE'
        #    ],
        #    set(['NOISE', 'MORE_NOISE'])
        #),
        #(
        #    [
        #        {('x', None): {('y', None): ['y', 'x']}},
        #        [{('x', None): {('y', None): 'x'}}, 'NOISE', 'MORE_NOISE'],
        #        {('x', None): 'rac'}
        #    ],
        #    set([])
        #)
    ]
    for tree, dependency_set in tree_depset_pairs:
        print(CC+'testing on @P {}@D '.format(tree))
        print(CC+'expected @O {}@D '.format(dependency_set))
        predicted = IA.abstract_eval(tree, environ)
        pre(predicted==dependency_set, 'failed test!')
        print(CC+'@G passed!@D \n')
    
    print('all tests executed!\n')
