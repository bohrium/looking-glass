''' author: samtenka
    change: 2020-02-25
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

Datapoint = namedtuple('Datapoint', ['atom', 'parent', 'resources']) 

normalize = (lambda a:
    (lambda s: {k:v/s for k,v in a.items()})
    (sum(a.values()))
)

# NOTE: no need for typed roots, since type information will already be in goal
#       and type matching with goal is already enforced via hard constraint

class WeightLearner: 
    def __init__(self):
        self.train_set = []
        self.types = set() 
        self.atoms = {'root', 'resource'}

    def observe_datapoint(self, head, parent, resources):
        if head in resources:
            self.train_set.append(Datapoint('resource', parent, resources))
        else:
            self.train_set.append(Datapoint(head, parent, resources))
            self.atoms.add(head)

    def observe_tree(self, tree, parent='root', resources={}): 
        if type(tree)==type(''):
            self.observe_tree([tree], parent, resources)
        elif type(tree)==type({}):
            for (var_nm, var_type), body in tree.items():
                self.types.add(var_type)
                new_resources = {k:v for k,v in resources.items()}
                new_resources[var_nm] = var_type
                self.observe_datapoint('root', parent, resources)
                self.observe_tree(body, 'root', new_resources) 
        else:
            caller, args = tree[0], tree[1:]
            self.observe_datapoint(caller, parent, resources)
            for arg in args:
                self.observe_tree(arg, caller, resources)

    def initialize_weights(self, pseudo_unigram=0.1, pseudo_bigram=0.1, pseudo_resource=0.1):
        self.w_unigram = {atom:pseudo_unigram for atom in self.atoms}
        self.w_bigram = {
            parent: {atom:pseudo_bigram for atom in self.atoms}
            for parent in self.atoms
        }
        self.w_resource = {
            r: {atom:pseudo_resource for atom in self.atoms}
            for r in self.types
        }

    def compute_weights(self):
        '''
            Fit a model
                P(atom | parent,resources) ~
                    exp(w_atom)
                    exp(w_(atom,parent))
                    product_resources of
                        exp(w_(atom,resource))
        '''
        self.initialize_weights()

        for atom, parent, resources in self.train_set: 
            self.w_unigram[atom] += 1
            self.w_bigram[parent][atom] += 1
            for r in resources.values():
                self.w_resource[r][atom] += 1
        self.w_unigram = normalize(self.w_unigram)
        self.w_bigram = {p:normalize(v) for p,v in self.w_bigram.items()}
        self.w_resource = {r:normalize(v) for r,v in self.w_resource.items()}

    def predict(self, parent, resources):
        scores = {
            a: 1.0
            for a in self.atoms
        }

        for a,v in self.w_unigram.items():
            scores[a] *= v**(0.0/5)
        for a,v in self.w_bigram[parent].items():
            scores[a] *= v**(5.0/5)
        for r in resources:
            for a,v in self.w_resources[r].items():
                scores[a] *= v**((0.0/5)/len(resources))

        scores = normalize(scores)
        return scores

if __name__=='__main__':
    tree = [
        'hello_prim',
        {('moo_varnm', tInt):
            'coon_prim'
        },
    ]
    
    WL = WeightLearner()
    WL.observe_tree(tree)
    print(WL.types)
    print(WL.atoms)
    WL.compute_weights()
    print(WL.predict('root', set([])))
    print(WL.predict('hello_prim', set([])))
