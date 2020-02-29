''' author: samtenka
    change: 2020-02-28
    create: 2019-02-28
    descrp: translate textual code to python function or program tree 
    to use: 
'''

import types

from utils import InternalError                         # maybe
from utils import CC, pre                               # ansi
from utils import secs_endured, megs_alloced            # profiling

from parse import Parser
from fit_weights import WeightLearner
from resources import PrimitivesWrapper
from vis import str_from_grids, render_color



#class DependencyType:
#
#    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#    #~~~~~~~~~  0.0 Constructors  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#
#    def __init__(self, kind, **kwargs):
#        pre(kind in ['base', 'from', 'pair'], 'unknown kind `{}`'.format(kind)) 
#        self.kind = kind
#        if self.kind=='base':
#            pass
#        elif self.kind=='from':
#            self.out = kwargs['out'] 
#            self.arg = kwargs['arg'] 
#            pre(not self.arg.has_arrow(), 'function-type arguments forbidden!')
#        elif self.kind == 'pair':
#            self.fst = kwargs['fst'] 
#            self.snd = kwargs['snd'] 
#
#    def frm(self, rhs):
#        return DependencyType('from', out=self, arg=rhs)
#
#    def pair(self, rhs):
#        return DependencyType('pair', fst=self, snd=rhs)
#
#    def __eq__(self, rhs):
#        return repr(self)==repr(rhs)
#
#    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#    #~~~~~~~~~  0.1 Analysis  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#
#    def squash(self):
#        ''' remove dependency from all arguments '''
#        if self.kind=='base': return self
#        elif self.kind=='from': return self.out
#        elif self.kind == 'pair': return (self.fst.squash(), self.snd.squash())
#
#    def has_arrow(self):
#        if self.kind=='base': return False 
#        elif self.kind=='from': return True
#        elif self.kind == 'pair':
#            return (
#                self.fst.has_arrow() or
#                self.snd.has_arrow()
#            )
#
#    def bottom(self, nb_ints=3):
#        if self.kind=='base':
#            return tuple(0 for _ in range(nb_ints))
#        elif self.kind=='from':
#            pre(False, 'no notion of bottom for `from` type!')
#        elif self.kind == 'pair':
#            return (self.fst.bottom(), self.snd.bottom())
#
#    def generators(self, nb_ints=3):
#        if self.kind=='base':
#            return set(
#                tuple(1 if j==i else 0 for j in range(nb_ints))
#                for i in range(nb_ints)
#            )
#        elif self.kind=='from':
#            pre(False, 'no notion of generators for `from` type!')
#        elif self.kind == 'pair':
#            return set(
#                [(g0, self.snd.bottom()) for g0 in self.fst.generators()] +
#                [(self.fst.bottom(), g1) for g1 in self.snd.generators()]  
#            )
#
#    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#    #~~~~~~~~~  0.2 Display  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#
#    def __repr__(self):
#        if self.kind=='base':
#            return '{int}'
#        elif self.kind=='from':
#            return '{} <- {}'.format(
#                ('({})' if self.out.kind=='pair' else '{}').format(
#                    repr(self.out)
#                ),
#                ('({})' if self.arg.kind in ('from', 'pair') else '{}').format(
#                    repr(self.arg)
#                )
#            )
#        elif self.kind=='pair':
#            return '{} x {}'.format(
#                ('({})' if self.fst.kind in ('from', 'pair') else '{}').format(
#                    repr(self.fst)
#                ),
#                ('({})' if self.snd.kind in ('from', 'pair') else '{}').format(
#                    repr(self.snd)
#                )
#            )
#
#    def __hash__(self):
#        return hash(repr(self))
#
##hey = DependencyType('base')
##heyhey = hey.pair(hey)
##heyfhey = hey.frm(hey)
##for h in (hey, heyhey):
##    print(repr(h), h.generators())
#
#class DependencyValue:
#    '''
#        if type is base, then data is set of ints
#        if type is pair, then data is pair of datas of children
#        if type is from, then data is a map map
#    '''
#
#    def __init__(self, dep_type, data): 
#        self.type = dep_type 
#        self.data = data
#
#    def union(self, rhs):
#        if   self.type.kind=='pair' and self.type.kind=='pair':
#            return DependencyValue(self.type, (
#                (self.fst.union(rhs.fst)).data,
#                (self.snd.union(rhs.snd)).data
#            ))
#        elif self.kind=='from' and rhs.kind=='from':
#            return DependencyValue(self.type,
#                lambda x: (self(x).union(rhs(x))).data
#            )
#        else:
#            return DependencyValue(self.type,
#                self.data.union(rhs.data)
#            )
#
#    def __call__(self, arg):
#        pre(self.type.kind=='from' and arg.type==self.type.arg,
#            'type mismatch during attempt to apply function!'
#        )
#        return DependencyValue(
#            self.type.out,
#            self.data(arg.data)
#        )


tree = [
    'hello',
    'noise',
    'cow'
]
resources = {
    'hello':(lambda a:lambda b: b, None),
    'cow':(5, None),
}
sensitives={
    'noise',
    'morenoise'
}

def abs_eval(tree, resources):
    '''
        resources gives (impl, type) pairs by name
    '''
    if type(tree)==type(''):
        if tree in sensitives:
            return set([tree])

        rtrn = resources[tree][0] # implementation, potentially itself unevaluated
        if type(rtrn)==type('') and (rtrn in resources):
            rtrn = abs_eval(rtrn, resources)

        if rtrn in sensitives:
            return set([rtrn])
        else:
            return set([])

    elif type(tree)==type({}):
        for (var_nm, var_type), body in tree.items():
            return (lambda x:
                abs_eval(body, {
                    k:v for k,v in list(resources.items())+[(var_nm, (x, var_type))]
                })
            )
    caller, args = tree[0], tree[1:]
    caller_head = caller if '<' not in caller else caller[:caller.find('<')] 
    if caller_head=='map':
        collection, func = map(abs_eval, args)
        return func, abs_eval(collection)
    elif caller_head=='fold':
        collection, init, func = map(abs_eval, args)
        ans = init 
        new_ans = None
        while ans != new_ans:
            ans = new_ans
            new_ans = ans.union(func(collection)(ans))
        return ans
    elif caller_head=='repeat':
        reps, init, func = map(abs_eval, args)
        ans = init 
        new_ans = None
        while ans != new_ans:
            ans = new_ans
            new_ans = ans.union(func(ans))
        return ans.union(reps)
    elif caller_head=='pair': return tuple(map(abs_eval, args))
    elif caller_head=='fst': return abs_eval(args[0])
    elif caller_head=='snd': return abs_eval(args[1])
    else:
        partial = abs_eval(caller, resources)
        if type(partial) == set:
            for arg in args:
                partial = partial.union(abs_eval(arg, resources))
        elif type(partial) == types.FunctionType:
            for arg in args:
                partial = partial(abs_eval(arg, resources))
        return partial

print(abs_eval(tree, resources))


#def abs_eval(tree, resources):
#    if type(tree)==type(''):
#        rtrn = resources[tree][0] # get implementation instead of type
#        if type(rtrn)==type('') and (rtrn in resources):
#            rtrn = abs_eval(rtrn, resources)
#        rtrn = {} # FLESH OUT BASE CASE
#        return rtrn
#    elif type(tree)==type({}):
#        for (var_nm, var_type), body in tree.items():
#            return (lambda x:
#                evaluate_tree(body, {
#                    k:v for k,v in list(resources.items())+[(var_nm, (x, var_type))]
#                })
#            )
#    caller, args = tree[0], tree[1:]
#    caller_head = caller if '<' not in caller else caller[:caller.find('<')] 
#    if caller_head=='map':
#        collection, func = map(abs_eval, args)
#        return func, abs_eval(collection)
#    elif caller_head=='fold':
#        collection, init, func = map(abs_eval, args)
#        ans = init 
#        new_ans = None
#        while ans != new_ans:
#            ans = new_ans
#            new_ans = ans UNION func(collection)(ans)
#        return ans
#    elif caller_head=='repeat':
#        reps, init, func = map(abs_eval, args)
#        ans = init 
#        new_ans = None
#        while ans != new_ans:
#            ans = new_ans
#            new_ans = ans UNION func(ans)
#        return ans UNION reps
#    elif caller_head=='pair': return tuple(map(abs_eval, args))
#    elif caller_head=='fst': return abs_eval(args[0])
#    elif caller_head=='snd': return abs_eval(args[1])
#    else:
#        partial = abs_eval(caller, resources)
#        for arg in args:
#            partial = partial(abs_eval(arg, resources))
#        return partial
