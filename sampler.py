''' author: samtenka
    change: 2020-03-12
    create: 2020-03-12
    descrp: generate a script of type grid x grid <-- noise
    to use: type
                from generate_script import get_script 
                get_script()
'''

from collections import namedtuple
import tqdm
import numpy as np
import glob

from utils import InternalError, internal_assert        # maybe
from utils import CC, pre, status                       # ansi
from utils import secs_endured, megs_alloced            # profiling
from utils import reseed, bernoulli, geometric, uniform # math

from lg_types import tInt, tCell, tColor, tShape, tBlock, tGrid, tDir, tNoise
from lg_types import tNmbrdBlock, tBlock, tClrdCell, tPtdGrid, tGridPair       
from lg_types import tCount_, tFilter_, tArgmax_, tMap_, tRepeat_

from parse import Parser, str_from_tree, nb_nodes 
from fit_weights import WeightLearner, init_edge_cntxt, next_edge_cntxt
from resources import PrimitivesWrapper
from demo import evaluate_tree
from vis import str_from_grids, render_color


Match = namedtuple('Match', ['head', 'subgoals']) 

class ListByKey: 
    def __init__(self):
        self.data = {}

    def add(self, key, val):
        if key not in self.data:
            self.data[key] = []
        self.data[key].append(val)

    def keys(self):
        return self.data.keys()

    def sample(self, key):
        return uniform(self.data[key])

class TreeSampler:
    def __init__(self):
        self.primitives = PrimitivesWrapper().primitives
        self.weights = WeightLearner()
        self.weights.observe_manual()
        self.weights.load_weights('fav.r04')

        self.reset_var_count()

    def reset_var_count(self):
        self.var_count = 0

    def get_fresh(self):
        self.var_count += 1
        return 'x'+str(self.var_count)

    def get_matches(self, goal, ecntxt):
        matches_by_actions = ListByKey()
        for head, (impl, sig) in self.primitives.items():
            for conseqs, subgoals in sig.conseq_hypoth_pairs():
                if goal not in conseqs: continue
                matches_by_actions.add(
                    head, Match(head = head, subgoals = subgoals)
                )

        for name, sig in ecntxt.hypths.items():
            for conseqs, subgoals in sig.conseq_hypoth_pairs():
                if goal not in conseqs: continue
                matches_by_actions.add(
                    'resource', Match(head = name, subgoals = subgoals)
                )

        if goal.kind=='from': 
            matches_by_actions.add(
                'root', Match(head = None, subgoals = [])
            )
        return matches_by_actions

    def sample_tree(self, goal, ecntxt):
        '''
        '''
        #if ecntxt.height == 0:
        #    #print(ecntxt.height, '?')
        #    return '?'

        matches_by_actions = self.get_matches(goal, ecntxt) 
        actions = matches_by_actions.keys()
    
        height = self.weights.sample_height(ecntxt)
        action = self.weights.sample_action(ecntxt, height, actions) 
        match = matches_by_actions.sample(action)

        #print(' '*ecntxt.deepth, end='')
        #status('[{}] [{}] [{}] [{}] -> [{}] ... subgoals: [{}]'.format(
        #    ecntxt.deepth, height, goal,
        #    ' ; '.join(map(str, set(ecntxt.hypths.values()))),
        #    action,
        #    ' ; '.join(map(str, match.subgoals))
        #), end='')
        #input()
    
        if action == 'root':
            var_nm = self.get_fresh()
            body = self.sample_tree(
                goal   = goal.out,
                ecntxt = next_edge_cntxt(
                    action, ecntxt, height, var_nm=var_nm, var_type=goal.arg
                )
            )
            return {(var_nm, goal.arg): body}
        else:
            nbkids = len(match.subgoals)
            if match.subgoals:
                favidx = self.weights.sample_favidx(
                    action=action,
                    nbkids=nbkids
                )
            subtrees = [
                self.sample_tree(
                    goal   = subgoal,
                    ecntxt = next_edge_cntxt(
                        action, ecntxt, height, idx=nbkids-1-i, favidx=favidx
                    )
                )
                for i, subgoal in enumerate(match.subgoals)
            ]
            return (
                ([match.head] + subtrees[::-1])
                if subtrees else match.head
            )

if __name__=='__main__':
    TS = TreeSampler()
    tree = TS.sample_tree(
        goal = tGridPair,
        ecntxt = init_edge_cntxt(height=12)
    )
    print(str_from_tree(tree))

    for _ in range(2):
        xys = [] 
        for _ in range(3):
            for _ in range(10):
                try:
                    xys += list(evaluate_tree(tree, TS.primitives))
                    break
                except InternalError:
                    continue
        if not xys: continue
        print(CC+str_from_grids([
            z.colors for z in xys
        ], render_color))


