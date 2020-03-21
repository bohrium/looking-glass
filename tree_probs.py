''' author: samtenka
    change: 2020-03-21
    create: 2019-03-21
    descrp: compute prior on trees from given weights
    to use: 
'''

import numpy as np
import tqdm

from utils import CC, pre, status               # ansi
from utils import secs_endured, megs_alloced    # profiling
from utils import reseed, uniform               # math
from utils import paths                         # path

from parse import Parser, str_from_tree, nb_nodes, get_height 
from lg_types import tGridPair, tColor
from fit_weights import WeightLearner, init_edge_cntxt, next_edge_cntxt
from resources import PrimitivesWrapper
from sampler import ListByKey, Match

# TODO: trees should bookkeep heights and nb_nodes
# TODO: use ecntxt_idx throughout (rather continually doing index searches)?
# TODO: use stack for ecntxt resources (rather thann copying huge dicts)?

# URGENT TODO: account for height sampling in the below, too!! 

class TreePrior: 
    def __init__(self, weights):
        self.weights = weights
        self.primitives = PrimitivesWrapper().primitives

    def log_prior(self, tree):
        height = get_height(tree)
        return self.log_prior_inner(
            goal   = tColor,#tGridPair,
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
        accum = 0.0
        if type(tree)==str:
            action = 'resource' if tree in ecntxt.hypths else tree
        elif type(tree)==dict:
            action = 'root'
            for (var_nm, var_type), body in tree.items():
                accum += self.log_prior_inner(
                    goal = var_type.out,
                    ecntxt = next_edge_cntxt(
                        action   = action,
                        ecntxt   = ecntxt,
                        height   = height-1,
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

            for i, arg in enumerate(args):
                accum += self.log_prior_inner(
                    goal = partial_type.arg,
                    ecntxt = next_edge_cntxt(
                        action   = action,
                        ecntxt = ecntxt,
                        height = heights[i],
                        idx    = i,
                        favidx = favidx,
                    ),
                    height = height-1,
                    tree = arg
                )
                partial_type = partial_type.out

        matches_by_action = self.get_matches(goal, ecntxt)
        alternative_actions = list(matches_by_action.keys())
        action_idx = alternative_actions.index(action) 
        p = self.weights.action_probs(ecntxt, height, alternative_actions)
        return (
            accum
            + np.log(p[action_idx])
            - np.log(matches_by_action.len_at(action)) # uniform sample 
        )

if __name__=='__main__':
    WL = WeightLearner()
    WL.observe_manual()
    WL.load_weights('fav.n20.r04')

    TP = TreePrior(WL)
    lp = TP.log_prior(['rainbow', 'noise'])
    status('perplexity (rainbow noise): [{:5.2f}]'.format(np.exp(lp)))

    lp = TP.log_prior('red')
    status('perplexity (red): [{:5.2f}]'.format(np.exp(lp)))
    lp = TP.log_prior('gray')
    status('perplexity (gray): [{:5.2f}]'.format(np.exp(lp)))
    lp = TP.log_prior('cyan')
    status('perplexity (cyan): [{:5.2f}]'.format(np.exp(lp)))
    lp = TP.log_prior('blue')
    status('perplexity (blue): [{:5.2f}]'.format(np.exp(lp)))
    lp = TP.log_prior('brown')
    status('perplexity (brown): [{:5.2f}]'.format(np.exp(lp)))
    lp = TP.log_prior('purple')
    status('perplexity (purple): [{:5.2f}]'.format(np.exp(lp)))
