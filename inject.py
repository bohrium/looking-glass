''' author: samtenka
    change: 2020-03-01
    create: 2019-02-29
    descrp: 
    to use: Test whether a tree is injective and non-trivial by: 

                from inject import InjectivityAnalyzer
                C = InjectivityAnalyzer()
                if C.is_interesting(tree): ...

'''

import sys
import types

from utils import InternalError # maybe
from utils import CC, pre       # ansi

from depend import DepAnalyzer, DepType
from lg_types import TS
from lg_types import tInt, tColor, tPtdGrid, tGridPair
from parse import Parser, str_from_tree
from resources import PrimitivesWrapper

def coarsen_type(lg_type):
    if lg_type.kind == 'base':
        if lg_type in TS.product_decompositions:  
            fst, snd = TS.product_decompositions[lg_type]
            fst = coarsen_type(fst) 
            snd = coarsen_type(snd) 
            return fst.pair(snd)
        else:
            return DepType('base') 
    elif lg_type.kind == 'from':
        out = coarsen_type(lg_type.out)
        arg = coarsen_type(lg_type.arg)
        return out.frm(arg)
    elif lg_type.kind == 'mset':
        return coarsen_type(lg_type.child) 
    else:
        pre(False, 'unknown type kind!')

def coarsen_tree(tree):
    if type(tree) == str:
        return tree 
    elif type(tree) == list:
        return [coarsen_tree(elt) for elt in tree]
    elif type(tree) == dict:
        return {
            (nm, coarsen_type(lg_type)): coarsen_tree(body)
            for (nm, lg_type), body in tree.items()
        } 
    else:
        pre(False, 'unknown tree type!')

class InjectivityAnalyzer:
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.reset_noise_count()

    def reset_noise_count(self):
        self.nb_noise_vars = 0 

    def prepare_analyzer(self):
        sensitives = {'noise{}'.format(i) for i in range(self.nb_noise_vars)}
        tBase = DepType('base')
        #sigs_by_nm = {
        #    'pair': tBase.pair(tBase).frm(tBase).frm(tBase),
        #    'mix': tBase.frm(tBase).frm(tBase),
        #}
        P = PrimitivesWrapper()
        sigs_by_nm = {
            nm:coarsen_type(t) for nm,(impl, t) in P.primitives.items()
        } 
        self.DA = DepAnalyzer(sensitives, sigs_by_nm, verbose=self.verbose)

    def label_tree(self, tree):
        ''' tree may be lg tree or dep tree '''
        if type(tree) == str:
            if tree=='noise':
                tree = 'noise{}'.format(self.nb_noise_vars) 
                self.nb_noise_vars += 1
            tree = tree[:] 
            return tree 
        elif type(tree) == list:
            return [self.label_tree(elt) for elt in tree]
        elif type(tree) == dict:
            return {
                (nm, t): self.label_tree(body)
                for (nm, t), body in tree.items()
            } 
        else:
            pre(False, 'unknown tree type!')

    def dependencies_of_pair(self, lg_tree): 
        # order matters in these 3 lines (stateful due to self.nb_noise_vars):
        self.reset_noise_count()
        labeled_tree = self.label_tree(lg_tree) 
        coarsened_tree = coarsen_tree(labeled_tree) 
        if self.verbose:
            print(CC+'ct: @P {} @D '.format(str_from_tree(coarsened_tree)))
        self.prepare_analyzer()
   
        dependency_value = self.DA.abstract_eval(coarsened_tree)
        x_dependencies = dependency_value.fst.squash() 
        y_dependencies = dependency_value.snd.squash() 
        return labeled_tree, x_dependencies, y_dependencies

    def is_interesting(self, lg_tree):
        ''' For a tree representing a function to a pair type X times Y,  
            test whether function's support, as a relation between X and Y, 
            has an injective transpose.  We assume that all noise that has a
            syntactic chain to X is fully recoverable from X's value. 
        '''
        _, x_deps, y_deps = self.dependencies_of_pair(lg_tree)
        print(CC+ 'dependencies @G {} @D and @G {} @D '.format(x_deps, y_deps))
        is_injective = (y_deps.difference(x_deps) == set([]))
        is_nonconstant = (y_deps != set([])) 
        is_nonidentity = (x_deps.difference(y_deps) != set([]))
        return (
            is_injective
            #and is_nonconstant
            #and is_nonidentity 
        )

if __name__=='__main__':
    code = '((\\n:noise -> (pair (mix n noise) (mix n noise))) noise)'
    tree = Parser(code).get_tree()
    C = InjectivityAnalyzer()
    labeled_tree, x_deps, y_deps = C.dependencies_of_pair(tree)
    print(CC+'analyzing @P {} @D '.format(labeled_tree))
    print(CC+'x depends on: @O {} @D '.format(x_deps))
    print(CC+'y depends on: @O {} @D '.format(y_deps))
    pre(x_deps=={'noise0', 'noise2'}, 'failed test')
    pre(y_deps=={'noise1', 'noise2'}, 'failed test')
