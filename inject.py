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

class DepType:
    def __init__(self, kind, **kwargs):
        pre(kind in ['base', 'from', 'pair'], 'unknown kind `{}`'.format(kind)) 
        self.kind = kind
        if self.kind=='base':
            pass
        elif self.kind=='from':
            self.out = kwargs['out'] 
            self.arg = kwargs['arg'] 
        elif self.kind == 'pair':
            self.fst = kwargs['fst'] 
            self.snd = kwargs['snd'] 
    def frm(self, rhs):
        return DepType('from', out=self, arg=rhs)
    def pair(self, rhs):
        return DepType('pair', fst=self, snd=rhs)

    def __repr__(self):
        if self.kind=='base': return 'BASE'
        elif self.kind == 'pair':
            return '({},{})'.format(repr(self.fst), repr(self.snd))
        elif self.kind=='from':
            return '({} <- {})'.format(repr(self.out), repr(self.arg))



    def build(self, leaves=set()):
        '''
            Return value of this type maximally sensitive to given set of
            ints
        '''
        if self.kind=='base':
            return DepValue('base',
                members = leaves
            )
        elif self.kind == 'pair':
            return DepValue('pair',
                fst = self.fst.build(leaves),
                snd = self.snd.build(leaves)
            )
        elif self.kind=='from':
            return DepValue('from',
                call_on = lambda x: self.out.build(leaves.union(x.squash())),
                default = self.out.build(leaves)
            )


class DepValue:

    def squash(self):
        '''
            Return set of ints on which, in isolation, this depvalue depends
        '''
        if self.kind=='base':
            return self.members
        elif self.kind == 'pair':
            return self.fst.squash().union(self.snd.squash())
        elif self.kind=='from':
            return self.default.squash() 

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
            self.default = kwargs['default']

    def pair(self, rhs):
        return DepValue('pair',
            fst=self,
            snd=rhs
        )

    def __repr__(self):
        if self.kind=='base':
            return str(sorted(self.members))
        elif self.kind == 'pair':
            return '({},{})'.format(repr(self.fst), repr(self.snd))
        elif self.kind=='from':
            return 'F{}'.format(repr(self.default))

    def __eq__(self, rhs):
        return repr(self)==repr(rhs)

tBase = DepType('base')

def base(*args):
    return DepValue('base', members=set(args))

class DepAnalyzer:
    def __init__(self, sensitives, environ, verbose=False):
        self.sensitives = sensitives
        self.environ = environ
        self.verbose = verbose
    
    def report(self, tag, dst, src, env, depth):
        ''' Print the small-step substitution specified in arguments. '''
        if not self.verbose: return
        print(CC + '  '*depth +
            ('@D {} @O {} @D from @P {} @D in environ @B {} @D '.format(
                tag, dst, src, env
            )
        ))
    
    def abstract_eval(self, tree, environ=None, depth=0): 
        '''
            Abstractly interpret `tree` to find subset of leaves in
            self.sensitives that could affect tree's value.
        '''
        if environ is None:
            environ = self.environ

        if type(tree) == DepValue:
            return tree
        elif type(tree) == str:
            if tree in sensitives: return base(tree)
            elif tree in environ: return self.abstract_eval(environ[tree], environ, depth+1)
            else: return base()
        elif type(tree) == dict:
            for (var_nm, var_type), body in tree.items():
                return DepValue('from',
                    call_on = lambda x:
                        self.abstract_eval(body, {
                            k:v for k,v in
                            list(environ.items())+[(var_nm, x)]
                        }, depth+1),
                    default =  
                        self.abstract_eval(body, {
                            k:v for k,v in
                            list(environ.items())+[(var_nm, var_type.build())]
                        }, depth+1)
                )
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
                    partial = partial.call_on(aa)
                self.report(':ans', partial, tree, environ, depth)
                return partial
        else:
            pre(False, 'tree is of unrecognized type!')

if __name__=='__main__':
    verbose = (len(sys.argv)==2 and sys.argv[1] == '-v')

    sensitives = {'NOISE', 'MORE_NOISE', 'EVEN_MORE_NOISE'}
    sigs_by_nm = {
        'rac3': tBase.frm(tBase).frm(tBase).frm(tBase),
        'rac2': tBase.frm(tBase).frm(tBase),
        'rac1': tBase.frm(tBase),
        'cow': tBase,
        'mix': tBase.frm(tBase.pair(tBase)),
        'mmix': (tBase.pair(tBase)).frm(tBase.pair(tBase)).frm(tBase.pair(tBase)),
        'mymap': (tBase.pair(tBase)).frm(tBase.frm(tBase)).frm(tBase.pair(tBase)),
    }
    environ = {
        k: v.build()
        for k,v in sigs_by_nm.items()
    }
    IA = DepAnalyzer(sensitives, environ, verbose=verbose)
        
    tree_depset_pairs = {
        'basic': [ 
            (
                ['MORE_NOISE'],
                base('MORE_NOISE')
            ),
            (
                [
                    {('y', tBase.frm(tBase)): ['y', 'NOISE']},
                    {('x', tBase): 'cow'}
                ],
                base()
            ),
            (
                [
                    {('x', tBase): {('y', tBase.frm(tBase)): ['y', 'x']}},
                    [{('x', tBase): {('y', tBase): 'x'}}, 'NOISE', 'MORE_NOISE'],
                    {('x', tBase): 'cow'}
                ],
                base()
            )
        ],
        'product types': [
            (
                ['pair', 'MORE_NOISE', 'NOISE'],
                base('MORE_NOISE').pair(base('NOISE'))
            ),
            (
                ['fst', ['pair', 'MORE_NOISE', 'NOISE']],
                base('MORE_NOISE')
            ),
            (
                [
                    {('f',tBase.frm(tBase)):
                        {('p',tBase.pair(tBase)):
                            ['pair', 
                                ['f', ['fst', 'p']],
                                ['f', ['snd', 'p']],
                            ]
                        }
                    },
                    {('x', tBase): 'NOISE'},
                    ['pair', 'MORE_NOISE', 'cow']
                ],
                base('NOISE').pair(base('NOISE'))
            ),
            (
                [
                    ['fst', ['pair',
                        {('x', tBase): 'NOISE'},
                        {('x', tBase): 'MORE_NOISE'},
                    ]],
                    ['MORE_NOISE']
                ],
                base('NOISE')
            ),
            (
                [
                    [
                        ['snd', ['pair',
                            {('x', tBase.pair(tBase)): ['fst', 'x']},
                            {('x', tBase.pair(tBase)): ['snd', 'x']},
                        ]],
                        ['pair',
                            {('x', tBase): {('y', tBase): ['pair', 'NOISE'     , 'cow']}},
                            {('x', tBase): {('y', tBase): ['pair', 'MORE_NOISE', 'cow']}},
                        ]
                    ],
                    'cow',
                    'cow'
                ],
                base('MORE_NOISE').pair(base())
            ),
        ],
        'blackbox primitives': [
            (
                [
                    'rac3',
                    [{('x', tBase): 'cow'}, 'MORE_NOISE'],
                    'NOISE',
                    'cow'
                ],
                base('NOISE')
            ),
            (
                [
                    {('x', tBase): {('y', tBase): ['rac2', 'x', 'y']}},
                    'MORE_NOISE', 'NOISE'
                ],
                base('NOISE', 'MORE_NOISE')
            ),
            (
                ['mix', ['pair', 'NOISE', 'NOISE']],
                base('NOISE')
            ),
            (
                ['mmix',
                    ['pair', 'NOISE', 'NOISE'],
                    ['pair', 'MORE_NOISE', 'NOISE']
                ],
                base('NOISE', 'MORE_NOISE').pair(base('NOISE', 'MORE_NOISE'))
            ),
            (
                ['mymap',
                    ['pair', 'NOISE', 'MORE_NOISE'],
                    'rac1',
                ],
                base('NOISE', 'MORE_NOISE').pair(base('NOISE', 'MORE_NOISE'))
            ),
        ],
    }

    print()
    for test_nm, pairs in tree_depset_pairs.items():
        print(CC+'@R --- TESTING {} ---@D '.format(test_nm))
        for tree, dependency_set in pairs:
            print(CC+'analyze the tree @P {}@D '.format(tree))
            print(CC+'expect dependencies @O {}@D '.format(dependency_set))
            predicted = IA.abstract_eval(tree, environ)
            pre(predicted==dependency_set, 'failed test!')
            print(CC+'@G passed!@D \n')
    
    print('all tests executed!\n')
