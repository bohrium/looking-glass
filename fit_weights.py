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

 NOTE: no need for typed roots, since type information will already be in goal
       and type matching with goal is already enforced via hard constraint
'''

# TODO: vectorize training set for faster training? 
# TODO: vectorize training set for faster training? 

from collections import namedtuple
import numpy as np
import tqdm

from utils import CC, pre, status               # ansi
from utils import secs_endured, megs_alloced    # profiling
from utils import reseed, uniform               # math
from utils import paths                         # path

from lg_types import tInt, tCell, tColor, tBlock, tGridPair 
from parse import Parser, get_height, str_from_tree

from resources import PrimitivesWrapper

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

    def len_at(self, key):
        return len(self.data[key])



def log_choice(n, a):
    return (
          np.sum(np.log(np.arange((n-a)+1, n+1)))
        - np.sum(np.log(np.arange(1, a+1)))
    )

def log_binomial(n_and_p, obs):
    '''
        return log ( (n choose obs) p^(obs) (1-p)^(n-obs)  )
    '''
    n, p = n_and_p
    return (
          log_choice(n, obs)
        +      obs * np.log(    p)
        + (n - obs)* np.log(1.0-p)
    )




from collections import namedtuple

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

max_depth=20
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

def display_datapoint(dp):
    status('[{:2}] [{:30}] [{:30}] [{:1}] [{:30}] [{:2}] --> '.format(*map(str, dp.ecntxt)), end='')
    status('[{:4}] [{:2}] [{:30}]'.format(*map(str, (dp.favidx, dp.height, dp.action))), mood='forest')

normalize_arr = (lambda a:
    a/np.sum(a)
)

normalize_dict = (lambda a:
    (lambda s: {k:v/s for k,v in a.items()})
    (sum(a.values()))
)

def get_generic(head):
    return head[:head.find('<')] if '<' in head else head

sigmoid = lambda x: 1.0/(1.0+np.exp(-x))

class Index:
    '''
    '''
    def __init__(self, items=set()):
        self.indices_by_elt = { v:i for i,v in enumerate(sorted(items)) }
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
        return self.indices_by_elt[elt] if elt in self.indices_by_elt else None

class WeightLearner: 
    def __init__(self, regularizer=1e-9, height_unit=10.0):
        self.train_set = []

        self.actions = Index({'root', 'resource'})
        self.parents = Index()
        self.genrics = Index()
        self.hypoths = Index({None})

        self.tree_sizes = {}

        self.height_unit = height_unit

        self.branch_factor = 1

        self.regularizer = regularizer

        self.primitives = PrimitivesWrapper().primitives

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
            #status('observing [{}] ... '.format(file_nm), end='')
            with open(file_nm) as f:
                nb_nodes = self.observe_tree(Parser(f.read()).get_tree())
            #status('[{:3}] nodes found!'.format(nb_nodes))

    def ecntxt_idx(self, ecntxt):
        return EdgeContext(
            height = ecntxt.height,
            action = self.parents.idx(ecntxt.action),
            parent = self.parents.idx(ecntxt.parent),
            hypths = set(
                self.hypoths.idx(h) for h in ecntxt.hypths.values()
                if h in self.hypoths.as_dict()
            ),
            favord = ecntxt.favord,
            deepth = ecntxt.deepth,
        )

    def height_probs(self, ecntxt):
        if ecntxt.height<=1:
            return 0
        if ecntxt.favord:
            return ecntxt.height-1 
        else:
            ecntxt_idx = self.ecntxt_idx(ecntxt)
            logit = float(self.height_logit(ecntxt_idx))
            p = sigmoid(logit)
            rtrn = np.random.binomial(n=ecntxt.height-1, p=p)
            return (ecntxt.height-1, p)

    def sample_height(self, ecntxt):
        if ecntxt.height<=0:
            return 0
        if ecntxt.favord:
            rtrn = ecntxt.height-1 
            #status('height [{:2}] [{:2}] ***'.format(ecntxt.height, rtrn))
        else:
            ecntxt_idx = self.ecntxt_idx(ecntxt)
            logit = float(self.height_logit(ecntxt_idx))
            p = sigmoid(logit)
            rtrn = np.random.binomial(n=ecntxt.height-1, p=p)
            #status('height pi[{}] h[{:2}] -> [{:.2f}] -> [{:.2f}] -> h\'[{:2}]'.format(
            #    ecntxt.action[-1], ecntxt.height, logit, p, rtrn
            #))
        return rtrn

    def action_logprobs(self, ecntxt, height, actions, none_val = -100.0):
        ecntxt_idx = self.ecntxt_idx(ecntxt)
        logits = self.action_logit(ecntxt_idx, height)
        logits = np.array([
            none_val if idx is None else logits[idx] 
            for a in actions
            for idx in [self.actions.idx(a)]
        ])
        logits -= np.amax(logits)
        norm = np.log(np.sum(np.exp(logits)))
        return logits - norm

    def sample_action(self, ecntxt, height, actions):
        # TODO: implement in terms of action_logprobs
        actions_by_idx = { 
            idx : a
            for a in actions
            for idx in [self.actions.idx(a)]
            if idx is not None
        }
        if not actions_by_idx:
            action = uniform(list(actions))
            #status('action [{}] ***'.format(height, action), mood='sea')
        else:
            action_indices = np.array(sorted(actions_by_idx.keys()))
            ecntxt_idx = self.ecntxt_idx(ecntxt)
            logits = self.action_logit(ecntxt_idx, height)[action_indices]
            logits -= np.amax(logits)
            probs = normalize_arr(np.exp(logits))
            idx = np.random.choice(action_indices, p=probs) 
            action = actions_by_idx[idx]
            #status('action [{:2}] -> [{}]'.format(height, action), mood='sea')
        return action

    def favidx_probs(self, action, nbkids):
        pre(nbkids <= self.branch_factor, 'unprecedented branch factor!') 
        action_idx = self.actions.idx(action) 
        logits = (
            np.full(nbkids, 1.0/nbkids)
            if action_idx is None else
            self.favidx_logit(action_idx, nbkids)
        )
        logits -= np.amax(logits)
        return normalize_arr(np.exp(logits))

    def sample_favidx(self, action, nbkids):
        pre(nbkids <= self.branch_factor, 'unprecedented branch factor!') 
        action_idx = self.actions.idx(action) 
        logits = (
            np.full(nbkids, 1.0/nbkids)
            if action_idx is None else
            self.favidx_logit(action_idx, nbkids)
        )
        logits -= np.amax(logits)
        probs = normalize_arr(np.exp(logits))
        idx = np.random.choice(nbkids, p=probs) 
        #status('favidx [{}] [{}] -> [{}]'.format(action, nbkids, idx), mood='forest')
        return idx

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
            height_loss = - log_binomial((ecntxt.height-1, prob), height)
            update = learning_rate * (prob - float(height)/ecntxt.height)

            self.w_height                           -= update 
            self.w_height_parent[ecntxt_idx.action] -= update
            self.w_height_grandp[ecntxt_idx.parent] -= update
        else:
            height_loss = 0.0

        # update action predictors
        # only actions that are possible should go here
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
                    sum_loss_f += favord_loss * avg_tree_size / self.tree_sizes[tindex]
                    sum_loss_h += height_loss * avg_tree_size / self.tree_sizes[tindex]
                    sum_loss_a += action_loss * avg_tree_size / self.tree_sizes[tindex]

            total_T += T

            status('log priors f[{:4.2f}] h[{:4.2f}] a[{:6.2f}] '.format(
                sum_loss_f / (len(self.tree_sizes) * T),
                sum_loss_h / (len(self.tree_sizes) * T),
                sum_loss_a / (len(self.tree_sizes) * T),
            ), end='')
            print(CC+'after @G {:3} @D epochs (learning rate @O {:7.1e}@D )'
                .format(total_T, eta)
            )

if __name__=='__main__':
    WL = WeightLearner(regularizer=1e-9)
    WL.observe_manual()

    WL.compute_weights()
    WL.save_weights('fav.n04.r09')
    #WL.load_weights('fav.n20.r04')
    print('done!')

    #print(CC+'@P action-specific weights:@D ')
    #print(CC+'@B {:6} @O {:7} @O {:7} @O {:7} @O {:7} @G {:6}'.format(
    #    'deepth', 'heightN', 'height0', 'height1', 'height2', 'unigrm'
    #))
    #named_weights = [(
    #    WL.w_action_deepth[i],
    #    WL.w_action_heightN[i],
    #    WL.w_action_height0[i],
    #    WL.w_action_height1[i],
    #    WL.w_action_height2[i],
    #    WL.w_action[i],
    #    a
    #) for a, i in WL.actions.as_dict().items()]
    #for i, (w_dp, w_ht, w_ht0, w_ht1, w_ht2, w_un, a) in enumerate(
    #    sorted(named_weights, key=lambda x:
    #        2*x[0]+ x[2]+x[3]+x[4]+x[5]
    #    )
    #):
    #    print(CC+'@B {:+6.2f} @O {:+7.2f} @O {:+7.2f} @O {:+7.2f} @O {:+7.2f} @G {:+6.2f} @D {:+6.2f} {}'.format(
    #        w_dp, w_ht, w_ht0, w_ht1, w_ht2, w_un, 
    #        2*w_dp + w_ht0+w_ht1+w_ht2+w_un,
    #        a
    #    ))
    #    if (i+1)%10==0:
    #        input()
