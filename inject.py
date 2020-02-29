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

class DependencyValue:
    def __init__(self, kind, **kwargs):
        pre(kind in ['base', 'from', 'pair'], 'unknown kind `{}`'.format(kind)) 

        self.kind = kind

        if self.kind=='base':
            self.members = kwargs['members']
        elif self.kind == 'pair':
            self.fst = kwargs['fst'] 
            self.snd = kwargs['snd'] 
        elif self.kind=='from':
            self.call_on = kwargs['call_on']

    def pair(self, rhs):
        return DependencyValue('pair',
            fst=self,
            snd=rhs
        )

    def __repr__(self):
        if self.kind=='base':
            return str(sorted(self.members))
        elif self.kind == 'pair':
            return '({},{})'.format(repr(self.fst), repr(self.snd))
        elif self.kind=='from':
            return 'FUNC'

    def __eq__(self, rhs):
        return repr(self)==repr(rhs)

    def union(self, rhs):
        if self.kind=='base':
            return DependencyValue(self.kind,
                members=self.members.union(rhs.members)
            )
        elif self.kind=='pair':
            return DependencyValue(self.kind,
                self.fst.union(rhs.fst),
                self.snd.union(rhs.snd)
            )
        elif self.kind=='from':
            return DependencyValue(self.kind,
                call_on = lambda x:
                    self.call_on(x).union(rhs.call_on(x)),
            )

def base(*args):
    return DependencyValue('base', members=set(args))

class DependencyAnalyzer:
    def __init__(self, sensitives, sigs_by_nm, verbose=False):
        self.sensitives = sensitives
        self.sigs_by_name = sigs_by_nm
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
        #if type(tree) in (set, types.FunctionType):
        #    return tree
        if type(tree) == DependencyValue:
            return tree

        elif type(tree) == str:
            if tree in sensitives:
                return DependencyValue('base', members={tree})
            elif tree in environ:
                impl = environ[tree][0]
                return self.abstract_eval(impl, environ, depth+1)
            else:
                return DependencyValue('base', members=set())
        elif type(tree) == dict:
            for (var_nm, var_type), body in tree.items():
                return DependencyValue('from', call_on=(lambda x:
                    self.abstract_eval(body, {
                        k:v for k,v in
                        list(environ.items())+[(var_nm, (x, var_type))]
                    }, depth+1)
                ))
        elif type(tree) == list:
            head, args = tree[0], tree[1:]

            if head=='pair':
                return ( 
                    self.abstract_eval(args[0], environ, depth+1).pair(
                    self.abstract_eval(args[1], environ, depth+1)
                ))
            elif head=='fst':
                return self.abstract_eval(args[0], environ, depth+1).fst
            elif head=='snd':                                       
                return self.abstract_eval(args[0], environ, depth+1).snd
            else:
                partial = self.abstract_eval(head, environ, depth+1)
                self.report('|cal', partial, head, environ, depth)
                for arg in args:
                    aa = self.abstract_eval(arg, environ, depth+1)
                    self.report('|arg', aa, arg, environ, depth)
                    partial = (
                        partial.call_on(aa)
                        if partial.kind=='from' else
                        partial.union(aa)
                    )
                self.report(':ans', partial, tree, environ, depth)
                return partial
        else:
            pre(False, 'tree is of unrecognized type!')

if __name__=='__main__':
    verbose = (len(sys.argv)==2 and sys.argv[1] == '-v')

    sensitives = {'NOISE', 'MORE_NOISE', 'EVEN_MORE_NOISE'}
    sigs_by_nm = {}
    #sigs_by_nm = {
    #    'mix': (base.pair(base)).frm(base.pair(base)).frm(base.pair(base))
    #}
    IA = DependencyAnalyzer(sensitives, sigs_by_nm, verbose=verbose)
        
    tree_depset_pairs = [
        (
            ['pair', 'MORE_NOISE', 'NOISE'],
            base('MORE_NOISE').pair(base('NOISE'))
        ),
        (
            ['fst', ['pair', 'MORE_NOISE', 'NOISE']],
            base('MORE_NOISE')
        ),
        #(
        #    ['mix',
        #        ['pair', 'NOISE', 'NOISE'],
        #        ['pair', 'MORE_NOISE', 'NOISE']
        #    ],
        #    (set(['NOISE', 'MORE_NOISE']), set(['NOISE', 'MORE_NOISE']))
        #),
        (
            ['MORE_NOISE'],
            base('MORE_NOISE')
        ),
        (
            ['rac', [{('x', 0): ['cow']}, 'MORE_NOISE'], 'NOISE', 'cow'],
            base('NOISE')
        ),
        (
            [
                {('x', None): {('y', None): ['cow', 'x', 'y']}},
                'MORE_NOISE', 'NOISE'
            ],
            base('NOISE', 'MORE_NOISE')
        ),
        (
            [
                {('x', None): {('y', None): ['y', 'x']}},
                [{('x', None): {('y', None): 'x'}}, 'NOISE', 'MORE_NOISE'],
                {('x', None): 'rac'}
            ],
            base()
        )
    ]
    for tree, dependency_set in tree_depset_pairs:
        print(CC+'testing on @P {}@D '.format(tree))
        print(CC+'expected @O {}@D '.format(dependency_set))
        predicted = IA.abstract_eval(tree, {})
        pre(predicted==dependency_set, 'failed test!')
        print(CC+'@G passed!@D \n')
    
    print('all tests executed!\n')
