''' author: samtenka
    change: 2020-03-01
    create: 2019-02-25
    descrp: learn dsl generation weights from observed trees 
    to use: To train on some trees then sample from a child of 'root' with one
            resource of type tInt, type:
                from fit_weights import WeightLearner
                WL = WeightLearner()
                for t in trees:
                    WL.observe_tree(t)
                WL.compute_weights()
                WL.predict('root', {tInt})
'''

import numpy as np

from utils import ARC_path
from utils import CC, pre                       # ansi
from utils import secs_endured, megs_alloced    # profiling
from utils import reseed, bernoulli, geometric  # math

from lg_types import tInt, tCell, tColor, tBlock, tGrid 

from collections import namedtuple

Datapoint = namedtuple('Datapoint', [
    'action',
    'parent',
    'grandp',
    'vailresources',
    'lastres',
    'depth',
]) 

normalize = (lambda a:
    (lambda s: {k:v/s for k,v in a.items()})
    (sum(a.values()))
)

# NOTE: no need for typed roots, since type information will already be in goal
#       and type matching with goal is already enforced via hard constraint

class Index:
    def __init__(self, items=set()):
        self.indices_by_elt = {
            v:i for i,v in enumerate(set(items))
        }
    def as_dict(self):
        return self.indices_by_elt
    def __str__(self):
        return str(self.indices_by_elt)
    def add(self, elt):
        if elt in self.indices_by_elt: return
        self.indices_by_elt[elt] = len(self.indices_by_elt) 
    def __len__(self):
        return len(self.indices_by_elt)
    def idx(self, elt):
        return self.indices_by_elt[elt]

class WeightLearner: 
    def __init__(self):
        self.train_set = []

        self.actions = Index({'root', 'resource'})
        self.parents = Index()
        self.types   = Index({None})

    # TODO: update nomenclature in this method
    def observe_datapoint(
        self, atom, parent, grandparent, resources, lastres, depth
    ):
        atom = 'resource' if atom in resources else atom
        self.actions.add(atom)
        self.parents.add(parent)
        self.parents.add(grandparent)

        self.train_set.append(Datapoint(
            atom,
            parent,
            grandparent,
            set(resources.values()),
            lastres,
            depth,
        ))

    def observe_tree(
        self, tree, parent='root', grandparent='root',
        resources={}, lastres=None, depth=0
    ): 
        if type(tree) == list:
            pre(type(tree[0]) == str,
                'program not in normal form due to caller {}'.format(tree[0])
            )

        head = (
            tree        if type(tree) == str else
            'root'      if type(tree) == dict else
            tree[0]     if type(tree) == list else
            pre(False, 'observe_tree: unrecognized type for {}'.format(tree))
        )

        self.observe_datapoint(
            head            ,
            parent          ,
            grandparent     ,
            resources       ,
            lastres         ,
            depth           ,
        )

        if type(tree) == str:
            pass
        elif type(tree) == dict:
            for (var_nm, var_type), body in tree.items():
                self.types.add(var_type)
                new_resources = {k:v for k,v in resources.items()}
                new_resources[var_nm] = var_type
                self.observe_tree(
                    tree            =   body            ,
                    parent          =   'root'          ,
                    grandparent     =   parent          ,
                    resources       =   new_resources   ,
                    lastres         =   var_type        ,
                    depth           =   depth+1         ,
                )
        elif type(tree) == list:
            caller, args = tree[0], tree[1:]
            new_parent = lambda i: (
                (caller, i) if type(caller)==str else
                pre(False, 'expected {}'.format(caller))
            )
            for i, arg in enumerate(args):
                self.observe_tree(
                    tree            =   arg             ,
                    parent          =   new_parent(i)   ,
                    grandparent     =   parent          ,
                    resources       =   resources       ,
                    lastres         =   lastres         ,
                    depth           =   depth+1         ,
                )

    def initialize_weights(self):
        out_dim = len(self.actions)
        par_dim = len(self.parents) 
        typ_dim = len(self.types) 

        self.w_unigram   = np.full(          out_dim,  0.0)
        self.w_parent    = np.full((par_dim, out_dim), 0.0)
        self.w_grandp    = np.full((par_dim, out_dim), 0.0)
        self.w_vailres   = np.full((typ_dim, out_dim), 0.0)
        self.w_lastres   = np.full((typ_dim, out_dim), 0.0)
        self.w_depth     = np.full(          out_dim,  0.0)

    def predict_logits(
        self, parent, grandp, vailresources, lastres, depth
    ):
        parent_idx = self.parents.idx(parent) 
        grandp_idx = self.parents.idx(grandp) 
        vailres_indices = [self.types.idx(res) for res in vailresources]
        lastres_idx = self.types.idx(lastres)

        as_array = self.predict_logits_by_indices(
            parent_idx, grandp_idx, vailres_indices, lastres_idx, depth
        )
        return {
            a : as_array[i]
            for a, i in self.actions.as_dict().items()
        }

    def predict_logits_by_indices(
        self, parent_idx, grandp_idx, vailres_indices, lastres_idx, depth
    ):
        logits = (
                 self.w_unigram
            +    self.w_parent [parent_idx]
            +    self.w_grandp [grandp_idx]
            +sum(self.w_vailres[        idx] for idx in vailres_indices)
            +    self.w_lastres[lastres_idx]
            +    self.w_depth * depth / 10.0
        )
        clipped = np.maximum(logits - np.amax(logits), -10.0)
        return clipped 

    def grad_update(
        self,
        action_idx,
        parent_idx, grandp_idx, vailres_indices, lastres_idx, depth,
        learning_rate, regularizer=0.1
    ):
        unnormalized_probs = np.exp(self.predict_logits_by_indices(
            parent_idx, grandp_idx, vailres_indices, lastres_idx, depth
        ))
        diffs = unnormalized_probs / np.sum(unnormalized_probs)
        loss = -np.log(diffs[action_idx]) 
        diffs[action_idx] -= 1.0
        # diffs times data gives loss

        self.w_unigram              -= learning_rate * (diffs                                                )  
        self.w_parent [parent_idx]  -= learning_rate * (diffs         + regularizer * np.sign(self.w_parent[parent_idx]))
        self.w_grandp [grandp_idx]  -= learning_rate * (diffs         + regularizer * np.sign(self.w_grandp[grandp_idx]))
        for idx in vailres_indices:
            self.w_vailres[idx]     -= learning_rate * (diffs         + regularizer * np.sign(self.w_vailres[idx]))
        self.w_lastres[lastres_idx] -= learning_rate * (diffs         + regularizer * np.sign(self.w_lastres[lastres_idx]))
        self.w_depth                -= learning_rate * (diffs * depth / 10.0                               )

        return loss

    def compute_weights(self, schedule=[(10,0.5),(10,0.1),(10,0.02),(10,0.004),]):
        '''
            Fit a model
                P(atom | parent,resources) ~
                    exp(w_atom)
                    exp(w_(atom,parent))
                    product_resources of
                        exp(w_(atom,resource))
        '''
        self.initialize_weights()

        total_T = 0
        for T, eta in schedule:
            sum_loss = 0.0 
            for _ in range(T):
                train = list(self.train_set)
                np.random.shuffle(train) 
                for action, parent, grandp, vailresources, lastres, depth in train:
                    action_idx = self.actions.idx(action) 
                    parent_idx = self.parents.idx(parent) 
                    grandp_idx = self.parents.idx(grandp) 

                    vailres_indices = [self.types.idx(res) for res in vailresources]
                    lastres_idx = self.types.idx(lastres)

                    sum_loss += self.grad_update(
                        action_idx,
                        parent_idx, grandp_idx, vailres_indices, lastres_idx, depth,
                        learning_rate = eta
                    )
            total_T += T
            print(CC + 'perplexity @R {:.2f} @D after @G {} @D epochs'.format(
                np.exp(sum_loss/(T*len(self.train_set))), total_T
            ))

if __name__=='__main__':
    tree = [
        'hello_prim',
        {('moo_varnm', tInt):
            ['moovarnm', 'coon_prim']
        },
    ]
    tree = ['i', ['snd', ['pair',
                        {('p','tPair'): ['fst', 'p']},
                        {('p','tPair'): ['snd', 'p']},
                    ]],
                    ['pair',
                        {('x','tBase'): {('y','tBase'): ['pair', 'NOISE', 'cow']}},
                        {('x','tBase'): {('y','tBase'): ['pair', 'DIN', 'cow']}},
                    ],
                    'cow',
                    'cow'
                ]
    
    WL = WeightLearner()
    WL.observe_tree(tree)
    print(CC+'@R actions: @P {} @D '.format(WL.actions))
    print(CC+'@R parents: @P {} @D '.format(WL.parents))
    print(CC+'@R types: @P {} @D '.format(WL.types))
    WL.compute_weights()
