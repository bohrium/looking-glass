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

ROLLOUT = 4
# TODO: convergence check so that rollout unneeded (subtle: need to check
# convergence across all tree nodes!!

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
        selfr, rhsr = repr(self), repr(rhs)
        pre('F' not in selfr+rhsr, 'cannot compare function types!')
        return selfr==rhsr

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

    def union_leaf(self, intset):
        if self.kind=='base':
            return DepValue('base',
                members = self.members.union(intset)
            )
        elif self.kind == 'pair':
            return DepValue('pair',
                fst = self.fst.union_leaf(intset),
                snd = self.snd.union_leaf(intset)
            )
        elif self.kind=='from':
            return DepValue('from',
                call_on = lambda x: self.call_on(x).union_leaf(intset),
                default = self.default.union_leaf(intset)
            )

    def union(self, rhs):
        if self.kind=='base': 
            return DepValue('base', members = self.members.union(rhs.members))
        elif self.kind=='pair':
            return DepValue('pair',
                fst = self.fst.union(rhs.fst),
                snd = self.snd.union(rhs.snd)
            )
        elif self.kind=='from':
            return DepValue('from',
                call_on = lambda x: self.call_on(x).union(rhs.call_on(x)),
                default = self.default.union(rhs.default)
            )

    # TODO: refactor:
    @staticmethod
    def nested_union(init, seq):
        '''
        '''
        if init.kind=='base': 
            members = init.members
            for elt in seq():
                new_members = members.union(elt.members)
                members = new_members
            return DepValue('base',
                members = members
            )
        elif init.kind=='pair':
            return DepValue('pair',
                fst = DepValue.nested_union(init.fst, lambda:(elt.fst for elt in seq())),
                snd = DepValue.nested_union(init.snd, lambda:(elt.snd for elt in seq())),
            )
        elif init.kind=='from':
            return DepValue('from',
                call_on = lambda x: DepValue.nested_union(init.call_on(x), lambda:(elt.call_on(x) for elt in seq())), 
                default = DepValue.nested_union(init.default, lambda:(elt.default for elt in seq())), 
            )

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

            eval_child = lambda t: self.abstract_eval(t, environ, depth+1)
    
            if head=='pair':
                fst, snd = map(eval_child, args)
                return fst.pair(snd)
            elif head=='fst':
                pair, = map(eval_child, args)
                return pair.fst
            elif head=='snd':
                pair, = map(eval_child, args)
                return pair.snd
            elif head=='cond':
                cond, if_tru, if_fls = map(eval_child, args)
                return if_tru.union(if_fls).union_leaf(cond.squash())
            elif head=='map':
                elts, func = map(eval_child, args)
                return func.call_on(elts)
            elif head=='filter':
                pred, elts = map(eval_child, args)
                return elts.union_leaf(pred.squash())
            elif head=='repeat':
                reps, init, func = map(eval_child, args)
                def rep_appl_generator():
                    x = init
                    for _ in range(ROLLOUT):
                        x = func.call_on(x)
                        yield x
                return DepValue.nested_union(
                    init = init,
                    seq = rep_appl_generator
                ).union_leaf(reps.squash())
            elif head=='fold':
                elts, init, func = map(eval_child, args)
                action = func.call_on(elts)
                def rep_appl_generator():
                    x = init
                    for _ in range(ROLLOUT):
                        x = action.call_on(x)
                        yield x
                return DepValue.nested_union(
                    init = init,
                    seq = rep_appl_generator
                ).union_leaf(elts.squash())
            else:
                partial = eval_child(head)
                self.report('|cal', partial, head, environ, depth)
                for arg in args:
                    aa = eval_child(arg)
                    self.report('|arg', aa, arg, environ, depth)
                    partial = partial.call_on(aa)
                self.report(':ans', partial, tree, environ, depth)
                return partial
        else:
            pre(False, 'tree is of unrecognized type!')

if __name__=='__main__':
    verbose = (len(sys.argv)==2 and sys.argv[1] == '-v')

    sensitives = {'NOISE', 'DIN', 'SOUND'}
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
        'plain lambda calculus': [ 
            (
                ['cow'],
                base()
            ),
            (
                ['NOISE'],
                base('NOISE')
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
                    [{('x', tBase): {('y', tBase): 'x'}}, 'NOISE', 'DIN'],
                    {('x', tBase): 'cow'}
                ],
                base()
            ),
            (
                [
                    {('x', tBase): {('y', tBase):
                        {('z', tBase.frm(tBase).frm(tBase)): ['z', 'x', 'y']}
                    }},
                    'DIN',
                    'NOISE',
                    {('x', tBase): {('y', tBase): 'x'}},
                ],
                base('DIN')
            ),
        ],
        'product types': [
            (
                ['pair', 'DIN', 'NOISE'],
                base('DIN').pair(base('NOISE'))
            ),
            (
                ['fst', ['pair', 'DIN', 'NOISE']],
                base('DIN')
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
                    ['pair', 'DIN', 'cow']
                ],
                base('NOISE').pair(base('NOISE'))
            ),
            (
                [
                    ['fst', ['pair',
                        {('x', tBase): 'NOISE'},
                        {('x', tBase): 'DIN'},
                    ]],
                    ['DIN']
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
                            {('x', tBase): {('y', tBase): ['pair', 'DIN', 'cow']}},
                        ]
                    ],
                    'cow',
                    'cow'
                ],
                base('DIN').pair(base())
            ),
        ],
        'blackbox primitives': [
            (
                [
                    'rac3',
                    [{('x', tBase): 'cow'}, 'DIN'],
                    'NOISE',
                    'cow'
                ],
                base('NOISE')
            ),
            (
                [
                    {('x', tBase): {('y', tBase):
                        {('z', tBase.frm(tBase).frm(tBase)): ['z', 'x', 'y']}
                    }},
                    'DIN',
                    'NOISE',
                    'rac2',
                ],
                base('NOISE', 'DIN')
            ),
            (
                ['mix', ['pair', 'NOISE', 'NOISE']],
                base('NOISE')
            ),
            (
                ['mmix',
                    ['pair', 'NOISE', 'NOISE'],
                    ['pair', 'DIN', 'NOISE']
                ],
                base('NOISE', 'DIN').pair(base('NOISE', 'DIN'))
            ),
            (
                ['mymap',
                    ['pair', 'NOISE', 'DIN'],
                    'rac1',
                ],
                base('NOISE', 'DIN').pair(base('NOISE', 'DIN'))
            ),
        ],
        'refined operators': [
            (
                ['cond',
                    ['rac2', 'cow', 'DIN'],
                    ['pair', 'SOUND', 'cow'],
                    ['pair', 'NOISE', 'cow'],
                ],
                base('DIN', 'SOUND', 'NOISE').pair(base('DIN'))
            ),
            (
                [
                    'map',
                    'SOUND',
                    {('x', tBase): ['pair', ['pair', 'x', 'cow'], 'NOISE']},
                ],
                (base('SOUND').pair(base())).pair(base('NOISE'))
            ),
            (
                ['repeat',
                    'SOUND',
                    ['pair', 'NOISE', ['pair', 'DIN', 'cow']],
                    {('x', tBase.pair(tBase.pair(tBase))):
                        ['pair',
                            ['fst', ['snd', 'x']],
                            ['pair', ['fst', ['snd', 'x']], ['fst', 'x']],
                        ]
                    }
                ],
                base('NOISE', 'DIN', 'SOUND').pair(
                    base('DIN', 'SOUND').pair(
                    base('NOISE', 'DIN', 'SOUND')
                ))
            ),
            (
                [
                    ['fold',
                        'NOISE',
                        {('x',tBase): ['pair', 'x', 'DIN']},
                        {('elt',tBase): {('f', (tBase.pair(tBase)).frm(tBase)):
                            {('x',tBase):
                                ['pair', ['fst', ['f', 'x']], 'elt']
                            }
                        }}
                    ],
                    'SOUND'
                ],
                base('NOISE', 'SOUND').pair(base('DIN', 'NOISE'))
            ),
            (
                ['filter',
                    {('x',tBase): 'SOUND'},
                    ['pair', 'DIN', 'cow']
                ],
                base('DIN', 'SOUND').pair(base('SOUND'))
            ),
        ],
    }

    print()
    for test_nm, pairs in tree_depset_pairs.items():
        print(CC+'@R --- TESTING {} ---@D '.format(test_nm))
        for tree, dependency_set in pairs:
            print(CC+'analyze tree @P {}@D : expect dependencies @O {}@D '.format(
                tree, dependency_set
            ))
            predicted = IA.abstract_eval(tree, environ)
            pre(predicted==dependency_set, 'failed test!')
            print(CC+'@G passed!@D ')
    
    print('all tests executed!\n')
