''' author: samtenka
    change: 2020-03-01
    create: 2020-02-11
    descrp: a type ontology for the looking glass dsl
    to use: To use named types and type constructors, import:

                from lg_types import (
                    tNoise, tInt, tColor, tShape, tGrid, tCell, tDir, tBlock,
                    tClrdCell, tNmbrdColor, tNmbrdBlock, tPtdGrid, tGridPair 
                )
                from lg_types import (
                    TPred_, tRepeat_, tCount_, tLen_, tFilter_, tArgmax_,
                    tMap_, TBinop_
                )

            For meta-data on types, query TS (an instance of LGTypeSystem):

                from lg_types import TS 
                TS.product_decomposition[tPtdGrid]  #   == (tGrid, tCell)
'''

from utils import pre, CC   # ANSI

#=============================================================================#
#=====  0. IMPLEMENTATION OF RECORD STRUCTURE FOR LG TYPES  ==================#
#=============================================================================#

class LGType:

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #~~~~~~~~~  0.0 Constructors  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    def __init__(self, kind, **kwargs):
        pre(kind in ['base', 'mset', 'from'],
            'unknown kind `{}`'.format(kind)
        ) 
        self.kind = kind
        if self.kind=='base':
            self.name = kwargs['name'] 
        elif self.kind=='from':
            self.out = kwargs['out'] 
            self.arg  = kwargs['arg'] 
        elif self.kind == 'mset':
            self.child = kwargs['child'] 

    def frm(self, rhs):
        return LGType('from', out=self, arg=rhs)

    def s(self):
        return LGType('mset', child=self)

    def __eq__(self, rhs):
        return repr(self)==repr(rhs)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #~~~~~~~~~  0.1 Display  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    def __repr__(self):
        if self.kind=='base':
            return self.name
        elif self.kind=='from':
            return '{} <- {}'.format(
                repr(self.out),
                ('({})' if self.arg.kind=='from' else '{}').format(
                    repr(self.arg)
                )
            )
        elif self.kind=='mset':
            return '{{{}}}'.format(repr(self.child))

    def __str__(self):   
        return repr(self)
        #if self.kind=='base':
        #    return self.name
        #elif self.kind=='from':
        #    return '{}_by_{}'.format(str(self.out), str(self.arg))
        #elif self.kind=='mset':
        #    return '{}s'.format(str(self.child))

    def __hash__(self):
        return hash(repr(self))

    def conseq_hypoth_pairs(self):
        '''
        '''
        if self.kind=='base': return [([self], [])]
        if self.kind=='from':
            return (
                #[([self], [])] + 
                [
                    (conseqs, hypoths + [self.arg])
                    for conseqs, hypoths in self.out.conseq_hypoth_pairs()
                ]
            )
        if self.kind=='mset': return [([self], [])]

#=============================================================================#
#=====  1. CLASS FOR TYPE SYSTEM  ============================================#
#=============================================================================#

class LGTypeSystem:
    def __init__(self):
        self.base_types_by_nm = {}
        self.product_decompositions = {}

    def add_base(self, name): 
        new_type = LGType('base', name=name)
        self.base_types_by_nm[name] = new_type
        return new_type

    def add_product(self, name, fst, snd): 
        new_type = self.add_base(name)
        self.product_decompositions[new_type] = (fst, snd) 
        return new_type

#=============================================================================#
#=====  2. ACTUAL LG TYPES  ==================================================#
#=============================================================================#

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~  2.0 Atomic Types  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

TS = LGTypeSystem() 

tNoise = TS.add_base('noise')
tInt   = TS.add_base('int')
tColor = TS.add_base('color')
tShape = TS.add_base('shape')
tGrid  = TS.add_base('grid')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~  2.1 Named Product Types  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

tCell           = TS.add_product('cell'         , tInt  , tInt  )
tDir            = TS.add_product('dir'          , tInt  , tInt  )
tBlock          = TS.add_product('block'        , tShape, tColor)
tClrdCell       = TS.add_product('clrdcell'     , tCell , tColor)
tNmbrdColor     = TS.add_product('nmbrdcolor'   , tInt  , tColor)
tNmbrdBlock     = TS.add_product('nmbrdblock'   , tBlock, tInt  )
tPtdGrid        = TS.add_product('ptdgrid'      , tGrid , tCell )
tGridPair       = TS.add_product('gridpair'     , tGrid , tGrid )
tNmbrdGrid      = TS.add_product('nmbrdgrid'    , tGrid , tInt  )

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~  2.2 Templates for Fake Polymorphism  ~~~~~~~~~~~~~~~~~~~~~~~~~#

tPred_   = lambda t:   tInt.frm(t)
tRepeat_ = lambda t:   t.s().frm(tInt)
tCount_  = lambda t:   tInt.frm(t.s())
tLen_    = lambda t:   tInt.frm(t.s())
tFilter_ = lambda t:   t.s().frm(tInt.frm(t)).frm(t.s())
tArgmax_ = lambda t:   t.frm(tInt.frm(t)).frm(t.s())
tMap_    = lambda t,u: t.s().frm(t.frm(u)).frm(u.s())
tBinop_  = lambda t:   t.frm(t).frm(t)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~  2.3 Examples of Display  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

if __name__ == '__main__':
    types_by_nm = {
    	'shapes'            :	tShape.s()              ,
    	'decompose shape'   :	tCell.s().frm(tShape)   , # TODO: implement in resources
    	'count cells'       :	tCount_(tCell)          ,
    	'argmax shapes'     :	tArgmax_(tShape)        ,
    	'filter blocks'     :	tFilter_(tBlock)        ,
    	'map'               :	tMap_(tColor,tGrid)     ,
    	'classify color'    :	tPred_(tColor)          ,
    	'binop on grids'    :	tBinop_(tGrid)         	,
    }
    for nm, lg_type in types_by_nm.items():
        print(CC+'{} @O {} @D '.format(nm.ljust(20), repr(lg_type)))

