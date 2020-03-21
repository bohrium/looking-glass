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

from parse import Parser, str_from_tree, nb_nodes, get_height 
from fit_weights import WeightLearner, init_edge_cntxt, next_edge_cntxt
from resources import PrimitivesWrapper
from demo import evaluate_tree
from vis import str_from_grids, render_color
from inject import InjectivityAnalyzer
from grid import Grid

from sampler import TreeSampler

class MetropolisHastingsSampler:
    def __init__(self):
        self.primitives = PrimitivesWrapper().primitives
        self.reset_varcount()
        self.depth_bound = 30

        self.weights = WeightLearner()
        self.weights.observe_manual()
        self.weights.load_weights('fav.r04')

        self.TS = TreeSampler(timeout_prob=1e-2)
        self.C = InjectivityAnalyzer()

    #=========================================================================#
    #=  2. METROPOLIS-HASTINGS STEP  =========================================#
    #=========================================================================#
 
    def sample(self, nb_steps, init_t=None):
        t = self.init_propose() if init_t==None else init_t
        lp = self.log_prior(t) 
        ll = self.log_likelihood(t) 
        for _ in range(nb_steps):
            t, lp, ll = self.mh_step(t, lp, ll) 
        return t, lp, ll

    def mh_step(self, old_tree, old_lp, old_ll):
        self.reset_varcount()
        while True:
            try:
                new_tree = self.propose_from(
                    old_tree, 
                    goal = tGridPair,
                    ecntxt = init_edge_cntxt(get_height(old_tree))
                )
            except AssertionError:
                continue
            break
        new_lp = self.log_prior(new_tree)
        new_ll = self.log_likelihood(new_tree)
        status('log prior [{:6.2f}]; log likelihood [{:6.2f}]'.format(
            new_lp, new_ll
        ))
        #print(CC+'@P {}@D '.format(str_from_tree(new_tree)))

        d_ls = (
            - np.log(nb_nodes(new_tree)) + np.log(nb_nodes(old_tree)) # MH correction
            + new_ll - old_ll
        )

        u = uniform(1.0)
        if u < np.exp(d_ls): 
            print(CC+'@G accept! {:.2f} {:.2f} {:.2f} @D '.format(u, np.exp(d_ls), d_ls))
            return (new_tree, new_lp, new_ll)
        else:
            print(CC+'@R reject! {:.2f} {:.2f} {:.2f} @D '.format(u, np.exp(d_ls), d_ls))
            return (old_tree, old_lp, old_ll)

    #=========================================================================#
    #=  0. SCORING  ==========================================================#
    #=========================================================================#

    def log_likelihood(self, tree,
        hidden_noise = 1.0,
        unused_noise = 0.5,
        runtime_err  = 8.0,
        monochrome   = 5.0, 
        identity     = 5.0, 
    ):
        ll = 0.0

        nb_hidden_noise, nb_unused_noise = self.C.interest_stats(tree)
        ll -= nb_hidden_noise * hidden_noise
        ll -= nb_unused_noise * unused_noise

        x,y = None, None
        try: x, y = evaluate_tree(tree, self.primitives)
        except: ll -= runtime_err

        for g in (x, y):
            if type(g)!=Grid: continue
            colors = set(e for r in g.colors for e in r)
            if len(colors) <= 1: ll -= monochrome / 2

        if type(x)==Grid and type(y)==Grid:
            if np.array_equal(x.colors, y.colors): ll -= identity

        return ll
    
    def log_prior(
        self,
        tree,
        goal = tGridPair,  
        parent='root',
        grandp='root',
        imprimitives={},
        lastres=None,
        depth=0
    ):
        return 0.0
        #if type(tree) == str:
        #    ll = self.log_prior_of_node(
        #        tree,
        #        goal = goal,
        #        parent=parent,
        #        grandp=grandp,
        #        imprimitives=imprimitives,
        #        lastres=lastres,
        #        depth=depth,
        #    )
        #    return ll
        #elif type(tree) == list:
        #    ll = self.log_prior_of_node(
        #        tree[0],
        #        goal = goal,
        #        parent=parent,
        #        grandp=grandp,
        #        imprimitives=imprimitives,
        #        lastres=lastres,
        #        depth=depth,
        #    )
        #    partial_type = self.primitives[tree[0]][1]
        #    for i, arg in enumerate(tree[1:]):
        #        arg_type, partial_type = partial_type.arg, partial_type.out
        #        ll += self.log_prior(arg,
        #            goal            =   arg_type        ,
        #            parent          =   (tree[0], i)    ,
        #            grandp          =   parent          ,
        #            imprimitives    =   imprimitives    ,
        #            lastres         =   lastres         ,
        #            depth           =   depth+1         ,
        #        )
        #    return ll
        #elif type(tree) == dict:
        #    for (var_nm, var_type), body in tree.items():
        #        if var_nm.startswith('x'):
        #            self.var_count = max(self.var_count, int(var_nm[1:])+1)
        #        resources = {k:v for k,v in imprimitives.items()}
        #        resources[var_nm] = var_type
        #        pre(var_type==goal.arg,
        #            'expected arg of type {} but {} has type {}'.format(
        #                goal.arg, var_nm, var_type 
        #        )) 
        #        ll = self.log_prior(body,
        #            goal            =   goal.out,
        #            parent          =   'root'      ,
        #            grandp          =   parent      ,
        #            imprimitives    =   resources   ,
        #            lastres         =   var_type    ,
        #            depth           =   depth+1     ,
        #        )
        #        return ll
        #else:
        #    pre(False, 'unexpected tree type {}'.format(type(tree)))

    def reset_varcount(self):
        self.var_count = 0

    def get_fresh(self):
        self.var_count += 1
        return 'x'+str(self.var_count)

    Match = namedtuple('Match', ['token', 'name', 'hypoths']) 

    def log_prior_of_node(
        self,
        atom,
        goal,
        parent,
        grandp,
        imprimitives,
        lastres,
        depth,
    ):
        '''
            self.primitives should be a dictionary of types by name
            imprimitives    should be a dictionary of types by name
        '''
        matches = [
            MetropolisHastingsSampler.Match(nm, nm, hypoths)
            for nm, (impl, sig) in self.primitives.items()
            for conseqs, hypoths in sig.conseq_hypoth_pairs()
            if goal in conseqs
        ] 
        matches += [
            MetropolisHastingsSampler.Match('resource', nm, hypoths)
            for nm, sig in imprimitives.items()
            for conseqs, hypoths in sig.conseq_hypoth_pairs()
            if goal in conseqs
        ]
        if goal.kind=='from': 
            matches.append(
                MetropolisHastingsSampler.Match('root', None, [])
            )

        logits_by_name = self.weights.predict_logits(
            parent          =   parent                      ,
            grandp          =   grandp                      ,
            vailresources   =   set(imprimitives.values())  ,
            lastres         =   lastres                     ,
            depth           =   depth                       ,
            codepth = None,
        )
        logits_by_name = {
            m.token:(logits_by_name[m.token] if m.token in logits_by_name else -10.0)
            for m in matches
        }

        mm = max(
            logits_by_name[m.token]
            for m in matches
        )
        log_partition = np.log(sum(
            np.exp(logits_by_name[m.token]-mm)
            if m.token in logits_by_name else 0.0
            for m in matches
        ))
        if atom in imprimitives:
            atom = 'resource'
        return (logits_by_name[atom]-mm) - log_partition
            
    #=========================================================================#
    #=  1. PROPOSALS  ========================================================#
    #=========================================================================#
    
    def init_propose(self):
        code = '''
            (
            split<color><gridpair> (rainbow noise) \\cc:color -> ( 
            split<shape><gridpair> (gen_shape noise four) \\ss:shape -> (
            pair<gridpair>
                (monochrome ss cc)
                (rotate_grid (monochrome ss gray) one)
            )))
        ''' 
        return Parser(code).get_tree()
    
    def propose_from(self, tree, goal, ecntxt, targ_idx=None):
        # sample resampling node uniformly
        if targ_idx is None:
            targ_idx = uniform(nb_nodes(tree)) 

        height = get_height(tree)

        if targ_idx == 0:
            while True:
                try:
                    subtree = self.TS.sample_tree(
                        goal = goal,
                        ecntxt = ecntxt,
                    )
                    break
                except InternalError as e:
                    print(e.msg, end='', flush=True)
                    continue
            return subtree

        if type(tree) == list:
            targ_idx -= 1 
            head, args = tree[0], tree[1:]
            partial_type = self.primitives[head][1]
            arg_idx = 0
            for arg in args:
                arg_type, partial_type = partial_type.arg, partial_type.out
                step = nb_nodes(arg) 
                if targ_idx-step < 0: break 
                arg_idx += 1
                targ_idx -= step
            favidx = max(range(len(args)), key=lambda i:get_height(args[i])) 
            return (
                [head] +
                args[:arg_idx] + 
                [
                    self.propose_from(
                        tree = args[arg_idx],
                        goal = arg_type,
                        ecntxt = next_edge_cntxt(
                            'resource' if head in ecntxt.hypths else head,
                            ecntxt,
                            height,
                            idx = arg_idx,
                            favidx = favidx,
                        ),
                        targ_idx = targ_idx
                    )
                ] +
                args[arg_idx+1:] 
            )
        elif type(tree) == dict:
            for (var_nm, var_type), body in tree.items():
                return {(var_nm, var_type):
                    self.propose_from(
                        tree = body,
                        goal = goal.out,
                        ecntxt = next_edge_cntxt(
                            'root',
                            ecntxt,
                            height,
                            var_nm = var_nm,
                            var_type = var_type,
                        ),
                        targ_idx = targ_idx - 1
                )}
        else:
            pre(False, '')

#=============================================================================#
#=====  3. CONSOLE OUTPUT  ===================================================#
#=============================================================================#

def show_task(tree, nb_rows=1, nb_cols=3, nb_tries=10, primitives=None):
    for _ in range(nb_rows):
        xys = [] 
        for _ in range(nb_cols):
            for _ in range(10):
                try:
                    xys += [z.colors for z in evaluate_tree(tree, primitives)]
                    xys.append(np.array([])) 
                    break
                except InternalError:
                    continue
        if not xys: continue
        print(CC+str_from_grids(xys, render_color))

if __name__=='__main__':
    MHS = MetropolisHastingsSampler()
    t, lp, ll = MHS.sample(1)
    for i in range(10000):
        try:
            print(CC+'i=@R {:6d}@D '.format(i))
            if i%100==0:
                input(CC+'@O show? @D ')
                print(str_from_tree(t))
                show_task(t, nb_rows=3, nb_cols=3, primitives=MHS.primitives) 
                input(CC+'@O next? @D ')
                with open('mcmc.new.arcdsl'.format(i), 'w') as f:
                    f.write(str_from_tree(t))
            t, lp, ll = MHS.sample(1, t)
            try:
                nb_cols = 1 if i%5 else 3 
                show_task(t, nb_rows=1, nb_cols=nb_cols, primitives=MHS.primitives) 
            except:
                print('uh oh!')
                continue
        except KeyboardInterrupt:
            status('you pressed ctrl+C!')
            status('press [enter] to continue or [x] to close')
            c = input()
            if c: exit()
