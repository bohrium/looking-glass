''' author: samtenka
    change: 2020-03-21
    create: 2019-03-21
    descrp: compute prior on trees from given weights
    to use: 
'''

# TODO: trees should bookkeep heights and nb_nodes
# TODO: use ecntxt_idx throughout (rather continually doing index searches)?
# TODO: use stack for ecntxt resources (rather thann copying huge dicts)?

# TODO: initial prior over total tree heights!

import numpy as np
import tqdm

from utils import CC, pre, status               # ansi
from utils import log_binom_dist                # math
from utils import paths                         # path

from containers import ListByKey  

from parse import Parser, str_from_tree, nb_nodes, get_height 
from lg_types import tGridPair, tColor
from fit_weights import WeightLearner, init_edge_cntxt, next_edge_cntxt, Match
from resources import PrimitivesWrapper

class TreePrior: 
    def __init__(self, weights):
        self.weights = weights
        self.primitives = PrimitivesWrapper().primitives

    def log_prior(self, tree):
        height = get_height(tree)
        return self.log_prior_inner(
            goal   = tGridPair,
            ecntxt = init_edge_cntxt(height),
            height = height,
            tree   = tree,
        )

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

    def log_prior_inner(self, goal, ecntxt, height, tree):
        param = self.weights.height_probs(ecntxt)
        accum = 0.0 if type(param)==int else log_binom_dist(param, height)   

        if type(tree)==str:
            action = 'resource' if tree in ecntxt.hypths else tree
        elif type(tree)==dict:
            action = 'root'
            for (var_nm, var_type), body in tree.items():
                accum += self.log_prior_inner(
                    goal = goal.out,
                    ecntxt = next_edge_cntxt(
                        action   = action,
                        ecntxt   = ecntxt,
                        height   = height,
                        var_nm   = var_nm,
                        var_type = var_type,
                    ),
                    height = height-1,
                    tree = body
                )
        elif type(tree)==list:
            caller, args = tree[0], tree[1:] 
            if caller in ecntxt.hypths: 
                action = 'resource'
                partial_type = ecntxt.hypths[caller]
            else:
                action = caller
                partial_type = self.primitives[caller][1]

            heights = [get_height(a) for a in args]
            favidx = max((h, i) for i,h in enumerate(heights))[1]

            accum += np.log(
                self.weights.favidx_probs(action, len(args))[favidx]
            )

            for i, arg in enumerate(args):
                accum += self.log_prior_inner(
                    goal = partial_type.arg,
                    ecntxt = next_edge_cntxt(
                        action   = action,
                        ecntxt = ecntxt,
                        height = height,
                        idx    = i,
                        favidx = favidx,
                    ),
                    height = heights[i],
                    tree = arg
                )
                partial_type = partial_type.out

        matches_by_action = self.get_matches(goal, ecntxt)
        alternative_actions = list(sorted(matches_by_action.keys()))
        action_idx = alternative_actions.index(action) 
        lp = self.weights.action_logprobs(ecntxt, height, alternative_actions)
        return (
            accum
            + lp[action_idx]
            - np.log(matches_by_action.len_at(action)) # uniform sample 
        )

if __name__=='__main__':
    WL = WeightLearner()
    WL.observe_manual()
    WL.load_weights('fav.n04.r09')

    TP = TreePrior(WL)

    file_nms = paths('manual')
    for fnm in file_nms:
        with open(fnm) as f:
            code = f.read()
        print(CC+'parsing @P {}@D ...'.format(fnm), end='  ')
        P = Parser(code)
        t = P.get_tree()

        lp = TP.log_prior(t)
        status('log prior: [{:8.2f}] for [{:3}] nodes'.format(lp, nb_nodes(t)))
