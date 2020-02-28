''' author: samtenka
    change: 2020-02-28
    create: 2020-02-16
    descrp: generate a script of type grid x grid <-- noise
    to use: type
                from generate_script import get_script 
                get_script()
'''

from collections import namedtuple

from utils import CC, pre                               # ansi
from utils import secs_endured, megs_alloced            # profiling
from utils import reseed, bernoulli, geometric, uniform # math

from lg_types import tInt, tCell, tColor, tShape, tBlock, tGrid, tDir, tNoise
from lg_types import tCount_, tFilter_, tArgmax_, tMap_, tRepeat_

from resources import PrimitivesWrapper
import tqdm


#=============================================================================#
#=====  0. PROVER  ===========================================================#
#=============================================================================#


class GrammarSampler:
    def __init__(self, verbose=False, depth_bound=15):
        self.primitives = PrimitivesWrapper().primitives
        self.weights = None
        self.verbose = verbose
        self.timeout_prob = 1e-2
        self.nb_tries = 10**2
        self.depth_bound = depth_bound

        self.var_count = 0

    def reset_varcount(self):
        self.var_count = 0

    def get_fresh(self):
        self.var_count += 1
        return 'x'+str(self.var_count)

    Match = namedtuple('Match', ['token', 'name', 'hypoths']) 

    def construct(self, goal, imprimitives={}, parent_token='root', depth_remaining=None):
        '''
            self.primitives should be a dictionary of types by name
            imprimitives    should be a dictionary of types by name
        '''
        if bernoulli(self.timeout_prob):
            pre(False, 'timeout')
        if self.verbose:
            print(CC+'current goal: @P {} @D '.format(str(goal)))
        if depth_remaining is None:
            depth_remaining = self.depth_bound
        pre(depth_remaining, 'depth reached')
    
        matches = [
            GrammarSampler.Match(nm, nm, hypoths)
            for nm, (impl, sig) in self.primitives.items()
            for conseqs, hypoths in sig.conseq_hypoth_pairs()
            if goal in conseqs
        ] 
        matches += [
            GrammarSampler.Match('resource', nm, hypoths)
            for nm, sig in imprimitives.items()
            for conseqs, hypoths in sig.conseq_hypoth_pairs()
            if goal in conseqs
        ]
        if goal.kind=='from': 
            matches.append(
                GrammarSampler.Match('root', None, [])
            )

        match = uniform(matches)
        #matches = self.weights.normalize({
        #    m: self.weights.score(
        #        m.token,
        #        parent=parent,
        #        resources=set(list(imprimitives.values()))
        #    )
        #    for m in matches
        #})

        if match.token == 'root':
            if self.verbose:
                print(CC+'introd @P {}@D '.format(goal, match))

            var_nm = self.get_fresh()
            resources = {k:v for k,v in imprimitives.items()}
            resources[var_nm] = goal.arg
            body = self.construct(goal.out, resources, 'root', depth_remaining-1)
            return (
                '(\\{}:{} -> \n{})'.format(var_nm, str(goal.arg), body)
            )
        else:
            if self.verbose:
                print(CC+'matched @P {}@D with @P {}@D '.format(goal, match.name))

            hypotheses = [
                self.construct(h, imprimitives, match.token, depth_remaining-1)
                for h in match.hypoths
            ]
    
            return (
                match.name if not hypotheses else '({} {})'.format(
                    match.name, ' '.join(h for h in hypotheses[::-1])
                )
            )

    def tenacious_construct(self, goal):
        it = range(self.nb_tries) 
        if not self.verbose: it = tqdm.tqdm(it)
        for _ in it:
            try:
               code = self.construct(goal)
               pre(code is not None, '')
               return code
            except:
                self.reset_varcount()
                continue

if __name__=='__main__':
    GS = GrammarSampler(verbose=False)
    c = GS.tenacious_construct(tInt) 
    print(c)

