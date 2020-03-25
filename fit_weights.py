''' author: samtenka
    change: 2020-03-25
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

 NOTE: no need for typed roots, since type information will already be in goal
       and type matching with goal is already enforced via hard constraint
'''

import cProfile

# TODO: lint all
# TODO: vectorize training set for faster training? 
# TODO: unify tree traversal

from collections import namedtuple
import numpy as np
import tqdm

from utils import CC, pre, status                   # ansi
from utils import uniform, binomial, log_binom_dist # math
from utils import paths                             # path

from containers import Index, ListByKey  

from lg_types import tGridPair 
from parse import Parser, get_height
from resources import PrimitivesWrapper

# TODO: expose this global parameter
max_depth = 30

Match = namedtuple('Match', ['head', 'subgoals']) 

EdgeContext = namedtuple('EdgeContext', [
    'height', # at top of edge 
    'action', # at top of edge
    'parent', # at top of edge
    'favord', # property of edge itself
    'hypths', # at bottom of edge
    'deepth', # at bottom of edge
]) 

Datapoint = namedtuple('Datapoint', [
    'ecntxt', # cntxt at edge to parent node
    'height', # height            --- in Nat      --- sampled based on ecntxt 
    'action', # action            --- in String   --- sampled based on ecntxt 
    'matchs', # potential actions
    'nbkids', # branching factor  
    'favidx', # idx of fav child  --- in MaybeNat --- sampled based on ecntxt 
    'tindex', # index of ambient tree (for book-keeping only)
]) 

def init_edge_cntxt(height):
    return EdgeContext( 
        height = height+1,
        action = 'root'  ,
        parent = 'root'  ,
        favord = 1       , 
        hypths = {}      ,
        deepth = 0       ,
    )

def next_edge_cntxt(
    action, ecntxt, height,
    var_nm=None, var_type=None, idx=None, favidx=None
):
    if action=='root':
        augmented_hypths = ecntxt.hypths.copy()
        augmented_hypths[var_nm] = var_type
        return EdgeContext( 
            height = height          ,
            action = 'root'          ,
            parent = ecntxt.action   ,
            hypths = augmented_hypths,
            favord = 1               , 
            deepth = min(max_depth, ecntxt.deepth+1) ,
        )
    else:
        return EdgeContext(
             height = height                 ,
             action = (action, idx)          ,
             parent = ecntxt.action          ,
             hypths = ecntxt.hypths          ,
             favord = 1 if idx==favidx else 0,
             deepth = min(max_depth, ecntxt.deepth+1) ,
        )

normalize_arr = (lambda a:
    a/np.sum(a)
)

def get_generic(head):
    return head[:head.find('<')] if '<' in head else head

sigmoid = lambda x: 1.0/(1.0+np.exp(-x))

class WeightLearner: 
    def __init__(self, regularizer=None, height_unit=10.0):
        self.train_set = []
        self.tree_sizes = {}
        self.tree_heights = {}
        self.primitives = PrimitivesWrapper().primitives

        self.actions = Index({'root', 'resource'})
        self.parents = Index()
        self.genrics = Index()
        self.hypoths = Index({None})

        self.height_unit = height_unit
        self.branch_factor = 1
        self.regularizer = regularizer
        self.none_val = -100.0

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

    def observe_manual(self):
        for file_nm in paths('manual'):
            status('observing [{}] ... '.format(file_nm), end='')
            with open(file_nm) as f:
                nb_nodes = self.observe_tree(Parser(f.read()).get_tree())
            status('[{:3}] nodes found!'.format(nb_nodes))

    def ecntxt_idx(self, ecntxt):
        return EdgeContext(
            height = ecntxt.height,
            action = self.parents.idx(ecntxt.action),
            parent = self.parents.idx(ecntxt.parent),
            hypths = set(
                self.hypoths.idx(h) for h in ecntxt.hypths.values()
                if h in self.hypoths
            ),
            favord = ecntxt.favord,
            deepth = ecntxt.deepth,
        )

    def top_height_prob_param(self): 
        return self.w_top_height[0]

    def height_prob_param(self, ecntxt):
        if ecntxt.height<=1:
            return 0
        if ecntxt.favord:
            return ecntxt.height-1 
        else:
            ecntxt_idx = self.ecntxt_idx(ecntxt)
            logit = float(self.height_logit(ecntxt_idx))
            p = sigmoid(logit)
            return (ecntxt.height-1, p)

    def action_probs(self, ecntxt, height, actions):
        ecntxt_idx = self.ecntxt_idx(ecntxt)
        logits = self.action_logit(ecntxt_idx, height)
        logits = np.array([
            self.none_val if idx is None else logits[idx] 
            for a in actions
            for idx in [self.actions.idx(a)]
        ])
        probs = normalize_arr(np.exp(logits - np.amax(logits)))
        return probs

    def favidx_probs(self, action, nbkids):
        pre(nbkids <= self.branch_factor, 'unprecedented branch factor!') 
        action_idx = self.actions.idx(action) 
        logits = (
            np.full(nbkids, 0.0)
            if action_idx is None else
            self.favidx_logit(action_idx, nbkids)
        )
        probs = normalize_arr(np.exp(logits - np.amax(logits)))
        return probs

    def observe_datapoint(self, ecntxt, height, head, matchs, nbkids, favidx, tindex):
        action = 'resource' if head in ecntxt.hypths else head

        self.actions.add(action)
        for m in sorted(matchs.keys()):
            self.actions.add(m)

        self.parents.add(ecntxt.parent)
        self.parents.add(ecntxt.action)
        self.genrics.add(get_generic(ecntxt.action))

        self.train_set.append(Datapoint(
            ecntxt = ecntxt,
            height = height,
            action = action,
            matchs = matchs,
            nbkids = nbkids,
            favidx = favidx,
            tindex = tindex,
        ))

    def observe_tree(self, tree): 
        tindex = len(self.tree_sizes)
        old_l = len(self.train_set)

        height = get_height(tree)

        self.tree_heights[tindex] = height

        self.observe_tree_inner(
            goal   = tGridPair              , 
            ecntxt = init_edge_cntxt(height),
            height = height                 , 
            tree   = tree                   ,
            tindex = tindex                 ,
        )

        new_l = len(self.train_set)
        nb_nodes = new_l - old_l
        self.tree_sizes[tindex] = nb_nodes 
        return nb_nodes

    def observe_tree_inner(self, goal, ecntxt, height, tree, tindex):
        if type(tree)==list:
            pre(type(tree[0])==str,
                'program not in normal form due to caller {}'.format(tree[0])
            )

        head = (
            tree        if type(tree) == str else
            'root'      if type(tree) == dict else
            tree[0]     if type(tree) == list else
            pre(False, 'unrecognized type for tree {}'.format(tree))
        )

        if type(tree) == str:
            favidx = None
            nbkids = 0
        elif type(tree) == dict:
            favidx = None
            nbkids = 1
            for (var_nm, var_type), body in tree.items():
                self.hypoths.add(var_type)

                self.observe_tree_inner(
                    goal   = goal.out,
                    ecntxt = next_edge_cntxt(
                        head, ecntxt, height, var_nm=var_nm, var_type=var_type
                    ),
                    height = height-1            , 
                    tree   = body                , 
                    tindex = tindex              ,
                )
        elif type(tree) == list:
            caller, args = tree[0], tree[1:]
            pre(type(caller)==str, 'expected {} to be a string'.format(caller))
            if caller in ecntxt.hypths: 
                action = 'resource'
                partial_type = ecntxt.hypths[caller]
            else:
                action = caller
                partial_type = self.primitives[caller][1]

            nbkids = len(args) 
            pre(nbkids, 'program not in normal form due to redundant parens!')
            self.branch_factor = max(self.branch_factor, nbkids)

            heights = [get_height(arg) for arg in args]
            favidx = max((h,i) for i, h in enumerate(heights))[1] 

            for i, (arg, h) in enumerate(zip(args, heights)):
                self.observe_tree_inner(
                    goal   = partial_type.arg,
                    ecntxt = next_edge_cntxt(
                        head, ecntxt, height, idx=i, favidx=favidx
                    ),
                    height = h                        ,
                    tree   = arg                      ,
                    tindex = tindex                   ,
                )
                partial_type = partial_type.out

        self.observe_datapoint(
            ecntxt = ecntxt,
            height = height,
            head   = head  ,
            matchs = self.get_matches(goal, ecntxt),
            nbkids = nbkids,
            favidx = favidx,
            tindex = tindex,
        )

    def initialize_weights(self):
        out_dim = len(self.actions)
        par_dim = len(self.parents) 
        typ_dim = len(self.hypoths) 

        self.w_favidx_action = np.full((self.branch_factor, out_dim), 0.0)

        self.w_top_height    = np.full(1      ,  0.0)
        self.w_height        = np.full(1      ,  0.0)
        self.w_height_parent = np.full(par_dim,  0.0)
        self.w_height_grandp = np.full(par_dim,  0.0)

        self.w_action         = np.full( out_dim          , 0.0)
        self.w_action_parent  = np.full((out_dim, par_dim), 0.0)
        self.w_action_grandp  = np.full((out_dim, par_dim), 0.0)
        self.w_action_hypths  = np.full((out_dim, typ_dim), 0.0)
        self.w_action_deepth  = np.full( out_dim          , 0.0)
        self.w_action_heightN = np.full( out_dim          , 0.0)
        self.w_action_height0 = np.full( out_dim          , 0.0)
        self.w_action_height1 = np.full( out_dim          , 0.0)
        self.w_action_height2 = np.full( out_dim          , 0.0)

    def get_weights_by_name(self):
        return {
            nm: getattr(self, nm)
            for nm in dir(self) if nm.startswith('w_')
        }

    def save_weights(self, prefix):
        for name, weights in self.get_weights_by_name().items():
            np.save('{}/{}.{}.npy'.format(
                paths('weights'), prefix, name
            ), weights)

    def load_weights(self, prefix):
        self.initialize_weights()
        for name, weights in self.get_weights_by_name().items():
            weights[:] = np.load('{}/{}.{}.npy'.format(
                paths('weights'), prefix, name
            ))

    def favidx_logit(self, action_idx, nbkids):
        return self.w_favidx_action[:nbkids, action_idx]

    def height_logit(self, ecntxt_idx):
        return ( 
            self.w_height                           +
            self.w_height_parent[ecntxt_idx.action] if ecntxt_idx.action is not None else 0 +
            self.w_height_grandp[ecntxt_idx.parent] if ecntxt_idx.parent is not None else 0
        )

    def action_logit(self, ecntxt_idx, height):
        if type(height)!=int:
            print('!!!', height)
            input()
        logits = (
                  self.w_action 
            +    (self.w_action_parent[:,ecntxt_idx.action] if ecntxt_idx.action is not None else 0)
            +    (self.w_action_grandp[:,ecntxt_idx.parent] if ecntxt_idx.parent is not None else 0)
            + sum(self.w_action_hypths[:,       idx       ] for idx in ecntxt_idx.hypths)
            +     self.w_action_deepth * ecntxt_idx.deepth / self.height_unit
            +    (self.w_action_heightN if 3<=height else 0)
            +    (self.w_action_height0 if height<=0 else 0)
            +    (self.w_action_height1 if height<=1 else 0)
            +    (self.w_action_height2 if height<=2 else 0)
        )

        return logits - np.amax(logits)

    def grad_update(self, ecntxt, height, action, matchs, nbkids, favidx, learning_rate):
        ecntxt_idx = self.ecntxt_idx(ecntxt)
        matchs_idxs = sorted(self.actions.idx(a) for a in matchs.keys())
        action_idx = self.actions.idx(action)
        action_i = matchs_idxs.index(action_idx)
        matchs_idxs = np.array(matchs_idxs)

        # update favord predictors
        if 2 <= nbkids:
            probs = normalize_arr(np.exp( self.favidx_logit(action_idx, nbkids) ))
            favidx_loss = -np.log(probs[favidx])

            diffs = probs; diffs[favidx] -= 1.0
            update = learning_rate * diffs

            self.w_favidx_action[:nbkids,action_idx] -= update
        else:
            favidx_loss = 0.0

        # update height predictors
        if not ecntxt.favord and 2<=ecntxt.height:
            prob = sigmoid( self.height_logit(ecntxt_idx)[0] ) 
            height_loss = - log_binom_dist((ecntxt.height-1, prob), height)
            update = learning_rate * (prob - float(height)/ecntxt.height)

            self.w_height                           -= update 
            self.w_height_parent[ecntxt_idx.action] -= update
            self.w_height_grandp[ecntxt_idx.parent] -= update
        else:
            height_loss = 0.0

        # update action predictors
        probs = normalize_arr(np.exp(self.action_logit(ecntxt_idx, height)[matchs_idxs]))
        action_loss = -np.log(probs[action_i]) + np.log(matchs.len_at(action)) 

        diffs = probs; diffs[action_i] -= 1.0
        update = learning_rate * diffs

        self.w_action[matchs_idxs]                          -= update
        self.w_action_parent[matchs_idxs,ecntxt_idx.action] -= update
        self.w_action_grandp[matchs_idxs,ecntxt_idx.parent] -= update
        for idx in ecntxt_idx.hypths:
            self.w_action_hypths[matchs_idxs,idx]           -= update
        self.w_action_deepth[matchs_idxs]                   -= update * ecntxt.deepth / self.height_unit
        self.w_action_heightN[matchs_idxs]                  -= update * (1 if 3<=height else -1)
        self.w_action_height0[matchs_idxs]                  -= update * (1 if height<=0 else -1)
        self.w_action_height1[matchs_idxs]                  -= update * (1 if height<=1 else -1)
        self.w_action_height2[matchs_idxs]                  -= update * (1 if height<=2 else -1)

        # l1 regularization:
        lr_reg = learning_rate * self.regularizer
        for nm, wght in self.get_weights_by_name().items(): 
            wght[np.abs(wght) < lr_reg] = 0.0
            wght -= lr_reg * np.sign(wght) 

        return favidx_loss, height_loss, action_loss

    def compute_weights(self, schedule=[(5, 0.8**i) for i in range(20)]):
        '''
            Fit a model
                P(atom | parent,hypths) ~
                    exp(w_atom)
                    exp(w_(atom,parent))
                    product_hypths of
                        exp(w_(atom,resource))
        '''
        self.initialize_weights()

        total_T = -1
        avg_tree_size = float(sum(self.tree_sizes.values()))/len(self.tree_sizes)
        for T, eta in [(1,0)]+schedule:
            sum_loss_f = 0.0 
            sum_loss_h = 0.0 
            sum_loss_a = 0.0 

            for _ in tqdm.tqdm(range(T)):
                train = list(self.train_set)
                np.random.shuffle(train) 
                for ecntxt, height, action, matchs, nbkids, favidx, tindex in train:
                    favord_loss, height_loss, action_loss = self.grad_update(
                        ecntxt, height, action, matchs, nbkids, favidx,
                        learning_rate = eta * avg_tree_size / self.tree_sizes[tindex]  
                    )
                    sum_loss_f += favord_loss / self.tree_sizes[tindex]
                    sum_loss_h += height_loss / self.tree_sizes[tindex]
                    sum_loss_a += action_loss / self.tree_sizes[tindex]

            total_T += T

            status('log priors f[{:4.2f}] h[{:4.2f}] a[{:6.2f}] '.format(
                sum_loss_f / (len(self.tree_sizes) * T),
                sum_loss_h / (len(self.tree_sizes) * T),
                sum_loss_a / (len(self.tree_sizes) * T),
            ), end='')
            print(CC+'after @G {:3} @D epochs (learning rate @O {:7.1e}@D )'
                .format(total_T, eta)
            )

        self.w_top_height[:] = (
            float(sum(self.tree_heights.values())) / 
            len(self.tree_heights)
        )
        
        return sum_loss_f, sum_loss_h, sum_loss_a

if __name__=='__main__':
    WL = WeightLearner()
    WL.observe_manual()
    for r in [6]:
        WL.regularizer = 10**(-r)
        lf, lh, la = WL.compute_weights()
        WL.save_weights('fav.n20.r{:02d}'.format(r))
        status('r = [{}];   f = [{}];   h = [{}];   a = [{}]'.format(
            r, lf, lh, la
        ))
    print('done!')
