''' author: samtenka
    change: 2020-02-29
    create: 2019-02-28
    descrp: Bound potential dependencies of output on specified noise leaves in
            a given program tree.  We use abstract interpretation, thus
            obtaining enough precision to find that
                (fst (pair a b))    does not depend on  b
            even though
                (pair a b)          depends on          b
            and
                (fst (pair a b))    depends on          (pair a b)
            In other words, we outperform naive transitive closure. 
    to use: To analyze a program tree, write

                from depend import DepAnalyzer
                DA = DepAnalyzer(sensitives, sigs_by_nm)
                dependency_value = DA.abstract_eval(tree)
                is_dependent = dependency_value.query('my_noise_6') 

            Above, `sensitives` is a set of strings representing the names of
            leaves on which our output might depend, and `sigs_by_nm` is a
            dictionary that sends leaf names to their abstract types (see
            DepType).  (Every non-noise leaf name must be mentioned in
            `sigs_by_nm` and every noise leaf name must be mentioned in
            `sensitives`.)  Here, the set is assumed to contain 'my_noise_6',
            and the last line above finds out whether or not the value can
            depend on that leaf after all.
                
            Often, the output type of the program tree will be structured as
            a pair; we may query the resulting fine-grained dependencies as
            follows: 

                fst_is_dependent = dependency_value.fst.query('my_noise_6') 
                snd_is_dependent = dependency_value.snd.query('my_noise_6') 


            To demonstrate verbosely on some artificial trees, run

                python depend.py -v 

            Remove that flag for non-verbose testing. 
'''

import sys
import types

from utils import CC, pre       # ansi

#=============================================================================#
#=====  0. TYPE OF DEPENDENCY VALUES  ========================================#
#=============================================================================#

class DepType:

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #~~~~~~~~~  0.0 Constructors  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

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

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #~~~~~~~~~  0.1 Build an Abstract Value from its Type  ~~~~~~~~~~~~~~~~~~~#

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

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #~~~~~~~~~  0.2 Display  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    def __repr__(self):
        if self.kind=='base': return 'BASE'
        elif self.kind == 'pair':
            return '({},{})'.format(repr(self.fst), repr(self.snd))
        elif self.kind=='from':
            return '({} <- {})'.format(repr(self.out), repr(self.arg))

#=============================================================================#
#=====  1. DEPENDENCY VALUES  ================================================#
#=============================================================================#

class DepValue:

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #~~~~~~~~~  1.0 Constructors  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

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

    def query(self, leaf):
        return leaf in self.squash().members 

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #~~~~~~~~~  1.1 Recursive Union Operators  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    #-----------------  1.1.0 union of children: yields collapse  ------------#

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

    #-----------------  1.1.1 broadcasted union with set of ints  ------------#

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

    #-----------------  1.1.2 union with another DepValue  -------------------#

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

    #-----------------  1.1.3 union with a sequence of DepValues  ------------#

    def nested_union(self, seq):
        '''
            `seq` should, upon being called with no arguments, return a
            generator instance of DepValue instances compatible in type with
            self.
        '''
        if self.kind=='base': 
            members = self.members
            for elt in seq(): members = members.union(elt.members)
            return DepValue('base', members = members)
        elif self.kind=='pair':
            return DepValue('pair',
                fst = self.fst.nested_union(lambda:(elt.fst for elt in seq())),
                snd = self.snd.nested_union(lambda:(elt.snd for elt in seq())),
            )
        elif self.kind=='from':
            return DepValue('from',
                call_on = (lambda x: self.call_on(x).nested_union(
                    lambda:(elt.call_on(x) for elt in seq())
                )), 
                default = self.default.nested_union(
                    lambda:(elt.default for elt in seq())
                ), 
            )

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #~~~~~~~~~  1.2 Basic Traversals for Display and Comparison  ~~~~~~~~~~~~~#

    def __repr__(self):
        if self.kind=='base':
            return str(sorted(self.members))
        elif self.kind == 'pair':
            return '({},{})'.format(repr(self.fst), repr(self.snd))
        elif self.kind=='from':
            return 'F{}'.format(repr(self.default))

    def __eq__(self, rhs):
        if self.kind=='base': return self.members==rhs.members
        elif self.kind == 'pair':
            return self.fst==rhs.fst and self.snd==rhs.snd
        elif self.kind=='from': pre(False, 'cannot compare function types!')



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~  1.3 Unwrapped Constructor (for convenience)  ~~~~~~~~~~~~~~~~~#

def base(*args):
    return DepValue('base', members=set(args))

#=============================================================================#
#=====  2. ABSTRACT INTERPRETER  =============================================#
#=============================================================================#

class DepAnalyzer:

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #~~~~~~~~~  2.0 Compile Environment and Sensitivities  ~~~~~~~~~`~~~~~~~~~#
 
    def __init__(self, sensitives, sigs_by_nm, verbose=False, rollout=4):
        '''
        '''

        self.sensitives = sensitives
        self.verbose = verbose
        self.rollout=4

        #-------------  2.0.0 compute worst case implementations from types  -#

        self.environ = {
            k: v.build()
            for k,v in sigs_by_nm.items()
        }

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #~~~~~~~~~  2.1 Main Recursion  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    
    def abstract_eval(self, tree, environ=None, depth=0): 
        ''' Abstractly interpret the given tree to find a subset of leaves in
            self.sensitives that could affect the tree's value.
        '''
        if environ is None:
            environ = self.environ

        if type(tree) == DepValue:
            return tree

        #-------------  2.1.0 tree is a leaf  --------------------------------#

        elif type(tree) == str:
            if tree in sensitives:
                return base(tree)
            elif tree in environ:
                return self.abstract_eval(environ[tree], environ, depth+1)
            else:
                return base()

        #-------------  2.1.1 tree is a lambda  ------------------------------#

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
    
            #---------  2.1.2 type-refined operators for products  -----------#

            if head=='pair':
                fst, snd = map(eval_child, args)
                return fst.pair(snd)
            elif head=='fst':
                pair, = map(eval_child, args)
                return pair.fst
            elif head=='snd':
                pair, = map(eval_child, args)
                return pair.snd

            #---------  2.1.3 dependence-refined operators  ------------------#

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
                seq = lambda: self.orbit(init, func.call_on) 
                return init.nested_union(seq).union_leaf(reps.squash())
            elif head=='fold':
                elts, init, func = map(eval_child, args)
                seq = lambda: self.orbit(init, func.call_on(elts).call_on) 
                return init.nested_union(seq).union_leaf(elts.squash())

            #---------  2.1.4 blackbox lambda application chain --------------#

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

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #~~~~~~~~~  2.2 Helpers  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    
    def report(self, tag, dst, src, env, depth):
        ''' Print the small-step substitution specified in arguments. '''
        if not self.verbose: return
        print(CC + '  '*depth +
            ('@D {} @O {} @D from @P {} @D in environ @B {} @D '.format(
                tag, dst, src, env
            )
        ))

    def orbit(self, x, f):
        ''' Return a generator yielding each successive f^i(x), starting with
            i=1 instead of i=0, and that stops after self.rollout many yields.
        '''
        for _ in range(self.rollout):
            x = f(x)
            yield x


#=============================================================================#
#=====  3. THOROUGH TESTING  =================================================#
#=============================================================================#

if __name__=='__main__':
    verbose = (len(sys.argv)==2 and sys.argv[1] == '-v')

    tBase = DepType('base')
    tPair = tBase.pair(tBase) 

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #~~~~~~~~~  3.0 Specify Environment and Sensitive Leaves  ~~~~~~~~~~~~~~~~#

    sensitives = {'NOISE', 'DIN', 'SOUND'}
    sigs_by_nm = {
        'rac3': tBase.frm(tBase).frm(tBase).frm(tBase),
        'rac2': tBase.frm(tBase).frm(tBase),
        'rac1': tBase.frm(tBase),
        'cow': tBase,
        'mix': tBase.frm(tPair),
        'mymix': tPair.frm(tPair).frm(tPair),
        'mymap': tPair.frm(tBase.frm(tBase)).frm(tPair),
    }

    test_collections_by_theme = {

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        #~~~~~  3.1 Lambda Calculus Ground Truths  ~~~~~~~~~~~~~~~~~~~~~~~~~~~#
 
        'plain lambda calculus': [ 

            #---------  3.1.0 inductive base cases  --------------------------#

            (
                ['cow'],
                base()
            ),
            (
                ['NOISE'],
                base('NOISE')
            ),

            #---------  3.1.1 higher order functions  ------------------------#

            (
                [
                    {('f',tBase.frm(tBase)): ['f', 'NOISE']},
                    {('x',tBase): 'cow'}
                ],
                base()
            ),
            (
                [
                    {('x',tBase): {('f',tBase.frm(tBase)): ['f', 'x']}},
                    [{('x',tBase): {('y',tBase): 'x'}}, 'NOISE', 'DIN'],
                    {('x',tBase): 'cow'}
                ],
                base()
            ),

            #---------  3.1.2 simulated pair type: explicit selector  --------#

            (
                [
                    {('x',tBase): {('y',tBase):
                        {('s',tBase.frm(tBase).frm(tBase)): ['s', 'x', 'y']}
                    }},
                    'DIN',
                    'NOISE',
                    {('x',tBase): {('y',tBase): 'x'}},
                ],
                base('DIN')
            ),
        ],

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        #~~~~~  3.2 Product Type Ground Truths  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
 
        'product types': [

            #---------  3.2.0 simple products  -------------------------------#

            (
                ['pair', 'DIN', 'NOISE'],
                base('DIN').pair(base('NOISE'))
            ),

            #---------  3.2.1 nontransitivity of dependence  -----------------#

            (
                ['fst', ['pair', 'DIN', 'NOISE']],
                base('DIN')
            ),

            #---------  3.2.2 functions of products  -------------------------#

            (
                [
                    {('f',tBase.frm(tBase)):
                        {('p',tPair):
                            ['pair', 
                                ['f', ['fst', 'p']],
                                ['f', ['snd', 'p']],
                            ]
                        }
                    },
                    {('x',tBase): 'NOISE'},
                    ['pair', 'DIN', 'cow']
                ],
                base('NOISE').pair(base('NOISE'))
            ),

            #---------  3.2.3 products of functions  -------------------------#

            (
                [
                    ['fst', ['pair',
                        {('x',tBase): 'NOISE'},
                        {('x',tBase): 'DIN'},
                    ]],
                    ['DIN']
                ],
                base('NOISE')
            ),
            (
                [
                    ['snd', ['pair',
                        {('p',tPair): ['fst', 'p']},
                        {('p',tPair): ['snd', 'p']},
                    ]],
                    ['pair',
                        {('x',tBase): {('y',tBase): ['pair', 'NOISE', 'cow']}},
                        {('x',tBase): {('y',tBase): ['pair', 'DIN', 'cow']}},
                    ],
                    'cow',
                    'cow'
                ],
                base('DIN').pair(base())
            ),
        ],

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        #~~~~~  3.3 Blackbox Primitive Ground Truths  ~~~~~~~~~~~~~~~~~~~~~~~~#
 
        'blackbox primitives': [

            #---------  3.3.0 dependency obeys blackbox's type  --------------#

            (
                [
                    'rac3',
                    [{('x',tBase): 'cow'}, 'DIN'],
                    'NOISE',
                    'cow'
                ],
                base('NOISE')
            ),

            #---------  3.3.1 simulated pair type: blackbox selector  --------#

            (
                [
                    {('x',tBase): {('y',tBase):
                        {('s',tBase.frm(tBase).frm(tBase)): ['s', 'x', 'y']}
                    }},
                    'DIN',
                    'NOISE',
                    'rac2',
                ],
                base('NOISE', 'DIN')
            ),

            #---------  3.3.2 maximal mixing of pairs  -----------------------#

            (
                ['mix', ['pair', 'NOISE', 'NOISE']],
                base('NOISE')
            ),
            (
                ['mymix',
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

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        #~~~~~  3.4 Refined Operator Ground Truths  ~~~~~~~~~~~~~~~~~~~~~~~~~~#
 
        'refined operators': [

            #---------  3.4.0 conditional does not mix products  -------------#

            (
                ['cond',
                    ['rac2', 'cow', 'DIN'],
                    ['pair', 'SOUND', 'cow'],
                    ['pair', 'NOISE', 'cow'],
                ],
                base('DIN', 'SOUND', 'NOISE').pair(base('DIN'))
            ),

            #---------  3.4.1 short dep chains; no product mixing  -----------#

            (
                [
                    'map',
                    'SOUND',
                    {('x',tBase): ['pair', ['pair', 'x', 'cow'], 'NOISE']},
                ],
                (base('SOUND').pair(base())).pair(base('NOISE'))
            ),
            (
                ['filter',
                    {('p',tBase): 'SOUND'},
                    ['pair', 'DIN', 'cow']
                ],
                base('DIN', 'SOUND').pair(base('SOUND'))
            ),

            #---------  3.4.2 long dep chains; sensitive to index set  -------#

            (
                ['repeat',
                    'SOUND',
                    ['pair', 'NOISE', ['pair', 'DIN', 'cow']],
                    {('p',tBase.pair(tPair)):
                        ['pair',
                            ['fst', ['snd', 'p']],
                            ['pair', ['fst', ['snd', 'p']], ['fst', 'p']],
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
                        {('elt',tBase): {('f',tPair.frm(tBase)):
                            {('x',tBase):
                                ['pair', ['fst', ['f', 'x']], 'elt']
                            }
                        }}
                    ],
                    'SOUND'
                ],
                base('NOISE', 'SOUND').pair(base('DIN', 'NOISE'))
            ),
        ],
    }

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #~~~~~~~~~  3.5 Analysis-and-Display Loop  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    DA = DepAnalyzer(sensitives, sigs_by_nm, verbose=verbose)

    print()
    for test_nm, pairs in test_collections_by_theme.items():
        print(CC+'@R --- TESTING {} ---@D '.format(test_nm))
        for tree, dependency_value in pairs:
            print(CC+'expect deps @O {} @D from tree @P {}@D '.format(
                dependency_value, tree
            ))
            predicted = DA.abstract_eval(tree)
            pre(predicted==dependency_value, 'failed test!')
            print(CC+'@G passed!@D ')
    
    print('all tests executed!\n')
