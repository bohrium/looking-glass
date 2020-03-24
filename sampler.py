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
from utils import bernoulli, binomial                   # math

from containers import ListByKey  

from lg_types import tInt, tCell, tColor, tShape, tBlock, tGrid, tDir, tNoise
from lg_types import tNmbrdBlock, tBlock, tClrdCell, tPtdGrid, tGridPair       
from lg_types import tCount_, tFilter_, tArgmax_, tMap_, tRepeat_

from parse import Parser, str_from_tree, nb_nodes 
from fit_weights import WeightLearner, init_edge_cntxt, next_edge_cntxt, Match
from resources import PrimitivesWrapper
from demo import evaluate_tree
from vis import str_from_grids, render_color

class TreeSampler:
    def __init__(self, timeout_prob=0.0):
        self.primitives = PrimitivesWrapper().primitives
        self.weights = WeightLearner()
        self.weights.observe_manual()
        self.weights.load_weights('fav.n20.r09')

        self.reset_var_count()
        self.timeout_prob = timeout_prob

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

    def sample_height(self, ecntxt):
        param = self.weights.height_prob_param(ecntxt)
        if type(param)==int:
            return param
        else:
            n, p = param
            return binomial(n, p)

    def sample_action(self, ecntxt, height, actions):
        # TODO : regularize orderedness of `actions`
        actions = list(actions)
        probs = self.weights.action_probs(ecntxt, height, actions)
        return np.random.choice(actions, p=probs) 

    def sample_favidx(self, action, nbkids):
        probs = self.weights.favidx_probs(action, nbkids)  
        idx = np.random.choice(nbkids, p=probs) 
        return idx

    def sample_tree(self, goal, ecntxt):
        '''
        '''
        pre(not bernoulli(self.timeout_prob), 'timeout')

        matches_by_actions = self.get_matches(goal, ecntxt) 
        actions = matches_by_actions.keys()
    
        height = self.sample_height(ecntxt)
        action = self.sample_action(ecntxt, height, actions) 
        match = matches_by_actions.sample(action)

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
                favidx = self.sample_favidx(
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


