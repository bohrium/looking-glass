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
from utils import CC, pre                               # ansi
from utils import secs_endured, megs_alloced            # profiling
from utils import reseed, bernoulli, geometric, uniform # math

from lg_types import tInt, tCell, tColor, tShape, tBlock, tGrid, tDir, tNoise
from lg_types import tNmbrdBlock, tBlock, tClrdCell, tPtdGrid, tGridPair       
from lg_types import tCount_, tFilter_, tArgmax_, tMap_, tRepeat_

from parse import Parser, str_from_tree, nb_nodes 
from fit_weights import WeightLearner
from resources import PrimitivesWrapper
from demo import evaluate_tree
from vis import str_from_grids, render_color
from inject import InjectivityAnalyzer

class MetropolisHastingsSampler:
    def __init__(self):
        self.primitives = PrimitivesWrapper().primitives
        self.reset_varcount()
        self.depth_bound = 20

        self.weights = WeightLearner()
        self.train_grammar()

    def train_grammar(self):
        for file_nm in glob.glob('manual*.*.arcdsl'):
            with open(file_nm) as f:
                self.weights.observe_tree(Parser(f.read()).get_tree())
        self.weights.compute_weights()

    def sample(self, nb_steps, init_t=None):
        t = self.init_propose() if init_t==None else init_t
        l = self.log_likelihood(t) 
        #for _ in tqdm.tqdm(range(nb_steps)):
        for _ in range(nb_steps):
            t, l = self.mh_step(t, l) 
        return t, l

    #=========================================================================#
    #=  0. LIKELIHOOD  =======================================================#
    #=========================================================================#
    
    def log_likelihood(
        self,
        tree,
        goal = tGridPair,  
        parent='root',
        grandp='root',
        imprimitives={},
        lastres=None,
        depth=0
    ):
        if type(tree) == str:
            ll = self.log_likelihood_of_node(
                tree,
                goal = goal,
                parent=parent,
                grandp=grandp,
                imprimitives=imprimitives,
                lastres=lastres,
                depth=depth,
            )
            return ll
        elif type(tree) == list:
            ll = self.log_likelihood_of_node(
                tree[0],
                goal = goal,
                parent=parent,
                grandp=grandp,
                imprimitives=imprimitives,
                lastres=lastres,
                depth=depth,
            )
            partial_type = self.primitives[tree[0]][1]
            for i, arg in enumerate(tree[1:]):
                arg_type, partial_type = partial_type.arg, partial_type.out
                ll += self.log_likelihood(arg,
                    goal            =   arg_type        ,
                    parent          =   (tree[0], i)    ,
                    grandp          =   parent          ,
                    imprimitives    =   imprimitives    ,
                    lastres         =   lastres         ,
                    depth           =   depth+1         ,
                )
            return ll
        elif type(tree) == dict:
            for (var_nm, var_type), body in tree.items():
                if var_nm.startswith('x'):
                    self.var_count = max(self.var_count, int(var_nm[1:])+1)
                resources = {k:v for k,v in imprimitives.items()}
                resources[var_nm] = var_type
                pre(var_type==goal.arg,
                    'expected arg of type {} but {} has type {}'.format(
                        goal.arg, var_nm, var_type 
                )) 
                ll = self.log_likelihood(body,
                    goal            =   goal.out,
                    parent          =   'root'      ,
                    grandp          =   parent      ,
                    imprimitives    =   resources   ,
                    lastres         =   var_type    ,
                    depth           =   depth+1     ,
                )
                return ll
        else:
            pre(False, 'unexpected tree type {}'.format(type(tree)))

    def reset_varcount(self):
        self.var_count = 0

    def get_fresh(self):
        self.var_count += 1
        return 'x'+str(self.var_count)

    Match = namedtuple('Match', ['token', 'name', 'hypoths']) 

    def log_likelihood_of_node(
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
            (pair<gridpair> (new_grid two two) (new_grid two two))
        ''' 
        return Parser(code).get_tree()
    
    def propose_from(self, tree, goal, parent='root', grandp='root', resources={}, lastres=None, depth=0, targ_idx=None):
        # sample resampling node uniformly
        if targ_idx is None:
            targ_idx = uniform(nb_nodes(tree)) 

        if targ_idx == 0:
            while True:
                try:
                    subtree = self.resample(goal, parent, grandp, imprimitives=resources, lastres=lastres, depth=depth)
                    break
                except InternalError:
                    print('.', end='')
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
            return (
                [head] +
                args[:arg_idx] + 
                [self.propose_from(args[arg_idx], arg_type, grandp, (head, arg_idx), resources, lastres, depth+1, targ_idx)] +
                args[arg_idx+1:] 
            )
        elif type(tree) == dict:
            for (var_nm, var_type), body in tree.items():
                resources = {k:v for k,v in resources.items()}
                resources[var_nm] = goal.arg
                return {(var_nm, var_type): self.propose_from(body, goal.out, grandp, 'root', resources, lastres, depth+1, targ_idx-1)}
        else:
            pre(False, '')

    def resample(
        self,
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
        #print(depth)
        #internal_assert(depth < self.depth_bound, 'uh oh!')

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
            codepth         =   None
        )
        logits_by_name = {
            m.token:(logits_by_name[m.token] if m.token in logits_by_name else -10.0)
            for m in matches
        }

        mm = max(
            logits_by_name[m.token]
            for m in matches
        )
        probs = np.array([
            np.exp(logits_by_name[m.token]-mm)
            if m.token in logits_by_name else 0.0
            for m in matches
        ])
        probs = probs/np.sum(probs)
        match = matches[np.random.choice(len(matches), p=probs)]

        if match.token == 'root':
            var_nm = self.get_fresh()
            resources = {k:v for k,v in imprimitives.items()}
            resources[var_nm] = goal.arg
            body = self.resample(
                goal            =   goal.out    ,
                parent          =   'root'      ,
                grandp          =   parent      ,
                imprimitives    =   resources   ,
                lastres         =   goal.arg    ,
                depth           =   depth+1     ,
            )
            return {(var_nm, goal.arg): body}
        else:
            hypotheses = [
                self.resample(
                    goal            =   h               ,
                    parent          =   (match.token, i),
                    grandp          =   parent          ,
                    imprimitives    =   imprimitives    ,
                    lastres         =   lastres         ,
                    depth           =   depth+1         ,
                )
                for i, h in enumerate(match.hypoths)
            ]
    
            return (
                ([match.name] + hypotheses[::-1])
                if hypotheses else match.name
            )
    
    #=========================================================================#
    #=  2. METROPOLIS-HASTINGS STEP  =========================================#
    #=========================================================================#
    
    def mh_step(self, old_tree, old_ll):
        self.reset_varcount()
        new_tree = self.propose_from(old_tree, tGridPair) 
        new_ll = self.log_likelihood(new_tree)
        print(CC+'ll @G {:4.2f}@D :'.format(new_ll))
        print(CC+'@P {}@D '.format(str_from_tree(new_tree)))

        mh_correction = - np.log(nb_nodes(new_tree)) + np.log(nb_nodes(old_tree)) 

        d_ll = mh_correction #+ new_ll - old_ll  
        u = uniform(1.0)
        if u < np.exp(d_ll): 
            print(CC+'@G accept! {:.2f} {:.2f} {:.2f} @D '.format(u, np.exp(d_ll), d_ll))
            return (new_tree, new_ll)
        else:
            print(CC+'@R reject! {:.2f} {:.2f} {:.2f} @D '.format(u, np.exp(d_ll), d_ll))
            return (old_tree, old_ll)


if __name__=='__main__':
    MHS = MetropolisHastingsSampler()
    t, l = MHS.sample(1)
    for _ in range(1000):
        t, l = MHS.sample(10, t)
        try:
            x, y = evaluate_tree(t, MHS.primitives)
            print(CC+str_from_grids([
                z.colors for z in [x,y]
            ], render_color))
        except:
            print('uh oh!')
            continue

