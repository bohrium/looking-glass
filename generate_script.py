''' author: samtenka
    change: 2020-03-01
    create: 2020-02-16
    descrp: generate a script of type grid x grid <-- noise
    to use: type
                from generate_script import get_script 
                get_script()
'''

from collections import namedtuple
import tqdm
import numpy as np

from utils import InternalError                         # maybe
from utils import CC, pre                               # ansi
from utils import secs_endured, megs_alloced            # profiling
from utils import reseed, bernoulli, geometric, uniform # math

from lg_types import tInt, tCell, tColor, tShape, tBlock, tGrid, tDir, tNoise
from lg_types import tNmbrdBlock, tBlock, tClrdCell, tPtdGrid, tGridPair       
from lg_types import tCount_, tFilter_, tArgmax_, tMap_, tRepeat_

from parse import Parser, str_from_tree 
from fit_weights import WeightLearner
from resources import PrimitivesWrapper
from demo import evaluate_tree
from vis import str_from_grids, render_color
from inject import InjectivityAnalyzer

#=============================================================================#
#=====  0. PROVER  ===========================================================#
#=============================================================================#

class GrammarSampler:
    def __init__(self, verbose=False, depth_bound=15):
        self.primitives = PrimitivesWrapper().primitives
        self.verbose = verbose
        self.timeout_prob = 1e-2
        self.nb_tries = 10**3
        self.depth_bound = depth_bound
        self.var_count = 0

        self.weights = WeightLearner()
        self.C = InjectivityAnalyzer()

    def learn_from(self, trees):
        for t in trees:
            #print('***', end=' ')
            self.weights.observe_tree(t)
        self.weights.compute_weights()

    def reset_varcount(self):
        self.var_count = 0

    def get_fresh(self):
        self.var_count += 1
        return 'x'+str(self.var_count)

    Match = namedtuple('Match', ['token', 'name', 'hypoths']) 

    def construct(
        self,
        goal,
        parent='root',
        grandp='root',
        imprimitives={},
        lastres=None,
        depth=0,
        codepth=None,
    ):
        '''
            self.primitives should be a dictionary of types by name
            imprimitives    should be a dictionary of types by name
        '''
        if codepth==None:
            codepth=self.depth_bound

        if bernoulli(self.timeout_prob):
            pre(False, 'timeout')
        if self.verbose:
            print(CC+'current goal: @P {} @D '.format(str(goal)))
        pre(depth < self.depth_bound, 'depth reached')
    
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

        logits_by_name = self.weights.predict_logits(
            parent          =   parent                      ,
            grandp          =   grandp                      ,
            vailresources   =   set(imprimitives.values())  ,
            lastres         =   lastres                     ,
            depth           =   depth                       ,
          codepth           = codepth                       ,
        )
        ss = [logits_by_name[m.token] for m in matches if m.token in logits_by_name]
        pre(ss, 'no matches')
        #if not ss:
        #    print(CC+'\n@R {} @G {} @D '.format(len(logits_by_name), sorted(list(logits_by_name.keys())[:5])))
        #    print(CC+'@R {} @P {} @D '.format(len(matches), sorted(list(m.token for m in matches)[:5])))
        #    input()
        mm = max(
            logits_by_name[m.token]
            for m in matches
            if m.token in logits_by_name
        )
        probs = np.array([
            np.exp(logits_by_name[m.token]-mm)
            if m.token in logits_by_name else 0.0
            for m in matches
        ])
        probs = probs/np.sum(probs)
        match = matches[np.random.choice(len(matches), p=probs)]

        if match.token == 'root':
            if self.verbose:
                print(CC+'introd @P {}@D '.format(goal, match))

            var_nm = self.get_fresh()
            resources = {k:v for k,v in imprimitives.items()}
            resources[var_nm] = goal.arg
            body = self.construct(
                goal            =   goal.out    ,
                parent          =   'root'      ,
                grandp          =   parent      ,
                imprimitives    =   resources   ,
                lastres         =   goal.arg    ,
                depth           =   depth+1     ,
                codepth         = max(0, codepth-1),
            )
            return (
                '\\{}:{} -> \n{}'.format(var_nm, str(goal.arg), body)
            )
        else:
            if self.verbose:
                print(CC+'matched @P {}@D with @P {}@D '.format(goal, match.name))

            get_cd = (
                lambda i: 
                min(codepth-1, geometric(2.0/(
                    (self.weights.w_codepth_parent[self.weights.parents.idx((match.token, i))]
                    if (match.token, i) in self.weights.parents.as_dict() else 0.5/codepth) + 
                    (self.weights.w_codepth_grandp[self.weights.parents.idx(parent)]
                    if parent in self.weights.parents.as_dict() else 0.5/codepth) 
                )))
            )
            hypotheses = [
                self.construct(
                    goal            =   h               ,
                    parent          =   (match.token, i),
                    grandp          =   parent          ,
                    imprimitives    =   imprimitives    ,
                    lastres         =   lastres         ,
                    depth           =   depth+1         ,
                  codepth           = (
                        #max(0,min(codepth-1,geometric(2*codepth)))
                        get_cd(i)
                        )
                )
                for i, h in enumerate(match.hypoths)
            ]
    
            return (
                match.name if not hypotheses else '({} {})'.format(
                    match.name, ' '.join(h for h in hypotheses[::-1])
                )
            )

    def tenacious_construct(self, goal, inner_execs=3):
        it = range(self.nb_tries) 
        if not self.verbose: it = tqdm.tqdm(it)
        for _ in it:
            try:
                code = self.construct(goal)
                P = Parser(code)
                t = P.get_tree()
                if not self.C.is_interesting(t):
                    print(CC+'@R uninteresting! @D ')
                    assert False
                msgs = set([])
                for _ in range(inner_execs):
                    try:
                        x,y = evaluate_tree(t, self.primitives)
                        #x_,y_ = evaluate_tree(t, self.primitives)
                        if set(e for r in y.colors for e in r) in [{'K'}, set([])]:
                            print(CC+'@R all blank! @D ')
                            assert False
                        #if y and y_: 
                        #    print(CC+'@R constant! @D ')
                        #    assert False
                        #if x==y and x_==y_: 
                        #    print(CC+'@R identity! @D ')
                        #    assert False
                        break
                    except InternalError as e:
                        if e.msg not in msgs:
                            print(CC+'\n@R : @B {} @D '.format(e.msg), end='')
                            #if e.msg.startswith('req'):
                            #    print(str_from_tree(t))
                            #    input()
                            msgs.add(e.msg)
                        continue
                else:
                    print(CC+'@R unrunnable! @D ')
                    assert False

                if code is not None:
                    return code
            except AssertionError:
                pass
            self.reset_varcount()

if __name__=='__main__':
    print(CC+'@P learning from examples...@D ')
    GS = GrammarSampler(verbose=False)

    CODE_FILE_NMS = [
        'manual.003.arcdsl',
        'manual.006.arcdsl',
        'manual.007.arcdsl',
        'manual.008.arcdsl',
        'manual.016.arcdsl',
        'manual.022.arcdsl',
        'manual.023.arcdsl',
        'manual.032.arcdsl',
        'manual.034.arcdsl',
        'manual.037.arcdsl',
    ]
    trees = []
    for file_nm in CODE_FILE_NMS:
        with open(file_nm) as f:
            trees.append(Parser(f.read()).get_tree())
    GS.learn_from(trees)

    for I in range(10):
        print(CC+'@P sampling new program...@D ')
        for _ in range(1):
            try:
                code = GS.tenacious_construct(tGridPair) 
                P = Parser(code)
                t = P.get_tree()
                break
            except TypeError:
                continue
        print(CC+'@O ')
        print(CC+'found... \n@P {} @D '.format(str_from_tree(t)))
        GS.weights.observe_tree(t) # for printing
        print(CC+'@D ')
        with open('boo{:02d}.arcdsl'.format(I), 'w') as f:
            f.write(str_from_tree(t))

        print(CC+'@P executing program {}...@D '.format(I))
        input()

        for _ in range(2):
            xys = [] 
            for _ in range(3):
                for _ in range(10):
                    try:
                        xys += list(evaluate_tree(t, GS.primitives))
                        break
                    except InternalError:
                        continue
            print(CC+str_from_grids([
                z.colors for z in xys
            ], render_color))

        print(CC+'@P done executing program {}...@D '.format(I))
        input()
