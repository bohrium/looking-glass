''' author: samtenka
    change: 2020-03-01
    create: 2019-02-29
    descrp: 
    to use: 
'''

import sys
import types

from utils import InternalError # maybe
from utils import CC, pre       # ansi

from depend import DepAnalyzer, DepType
from lg_types import TS
from lg_types import tInt, tColor
from parse import Parser

def coarsen_type(lg_type):
    if lg_type.kind == 'base':
        if lg_type.name in TS.product_decompositions:  
            fst, snd = TS.product_decompositions[lg_type.name]
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

class Coarsener:
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.reset_noise_count()

    def reset_noise_count(self):
        self.nb_noise_vars = 0 

    def prepare_analyzer(self):
        sensitives = {'noise{}'.format(i) for i in range(self.nb_noise_vars)}
        tBase = DepType('base')
        sigs_by_nm = {
            'pair': tBase.pair(tBase).frm(tBase).frm(tBase),
            'mix': tBase.frm(tBase).frm(tBase),
        }
        self.DA = DepAnalyzer(sensitives, sigs_by_nm, verbose=self.verbose)

    def label_tree(self, tree):
        ''' tree may be lg tree or dep tree '''
        if type(tree) == str:
            if tree=='noise':
                tree = 'noise{}'.format(self.nb_noise_vars) 
                self.nb_noise_vars += 1
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

    def obstructions_to_injectivity(self, lg_tree, verbose=False): 
        # order matters in these 3 lines (stateful due to self.nb_noise_vars):
        self.reset_noise_count()
        labeled_tree = self.label_tree(lg_tree) 
        coarsened_tree = coarsen_tree(labeled_tree) 
        self.prepare_analyzer()
   
        dependency_value = self.DA.abstract_eval(coarsened_tree)
        x_dependencies = dependency_value.fst.squash() 
        y_dependencies = dependency_value.snd.squash() 
        return labeled_tree, y_dependencies.difference(x_dependencies)

if __name__=='__main__':
    code = '((\\n:noise -> (pair (mix n noise) (mix n noise))) noise)'
    tree = Parser(code).get_tree()
    C = Coarsener()
    labeled_tree, obstructions = C.obstructions_to_injectivity(tree)
    print(CC+'analyzing @P {} @D '.format(labeled_tree))
    print(CC+'bad dependencies: @O {} @D '.format(obstructions))
