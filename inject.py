''' author: samtenka
    change: 2020-02-29
    create: 2019-02-29
    descrp: 
    to use: 
'''

import sys
import types

from utils import InternalError # maybe
from utils import CC, pre       # ansi

from depend import DepAnalyzer

def injective(tree)

sensitives = {'NOISE', 'DIN', 'SOUND'}
sigs_by_nm = {
    'rac3': tBase.frm(tBase).frm(tBase).frm(tBase),
    'rac2': tBase.frm(tBase).frm(tBase),
    'rac1': tBase.frm(tBase),
    'cow': tBase,
    'mix': tBase.frm(tPair),
    'mymix': tPair.frm(tPair).frm(tPair),
    'mymap': tPair.frm(tBase.frm(tBase)).frm(tPair),
}

DA = DepAnalyzer(sensitives, sigs_by_nm)

dependency_value = DA.abstract_eval(tree)
is_dependent = dependency_value.query('my_noise_6') 

