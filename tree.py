''' author: samtenka
    change: 2020-02-25
    create: 2019-02-25
    descrp:
    to use: 
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

# TODO: typed roots?

class WeightLearner: 
    def __init__(self):
        self.train_set = []
        self.types = set() 
        self.atoms = {'root'}

    def observe_tree(self, tree, parent='root', resources=()): 
        if type(tree)==type(''):
            self.observe_tree([tree], parent, resources)
        elif type(tree)==type({}):
            for introd_type, body in tree.items():
                self.observe_tree(body, 'root', resources+(introd_type,)) 
                self.types.add(introd_type)
        else:
            caller, args = tree[0], tree[1:]
            self.train_set.append(Datapoint(caller, parent, resources))
            self.atoms.add(caller)
            for arg in args:
                self.observe_tree(arg, caller, resources)

    def initialize_weights(self, pseudo_unigram=1.0, pseudo_bigram=1.0, pseudo_resource=1.0):
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
            for r in resources:
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
            scores[a] *= v**1.0/3
        for a,v in self.w_bigram[parent].items():
            scores[a] *= v**1.0/3
        for r in resources:
            for a,v in self.w_resources[r].items():
                scores[a] *= v**((1.0/3)/len(resources))

        scores = normalize(scores)
        return scores

tree = [
    'hello',
    'moo',
    {tInt:
        'coon'
    },
    'goo',
]

WL = WeightLearner()
WL.observe_tree(tree)
print(WL.types)
print(WL.atoms)
WL.compute_weights()
print(WL.predict('root', ()))
