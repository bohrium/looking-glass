''' author: samtenka
    change: 2020-02-25
    create: 2020-02-11
    descrp: a type ontology for the looking glass dsl
    to use: import
                from lg_types import tInt, tCell, tColor, tBlock, tGrid 
                from lg_types import tCount_, tFilter_, tArgmax_, tMap_
'''

from utils import pre

#=============================================================================#
#=====  0. IMPLEMENTATION OF RECORD STRUCTURE FOR LG TYPES  ==================#
#=============================================================================#

class LGType:

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #~~~~~~~~~  0.0 Constructors  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    def __init__(self, kind, **kwargs):
        pre(kind in ['base', 'mset', 'from', 'prod'],
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
        #elif self.kind == 'prod':
        #    self.fst = kwargs['fst'] 
        #    self.snd = kwargs['snd'] 

    def frm(self, rhs):
        return LGType('from', out=self, arg=rhs)

    def s(self):
        return LGType('mset', child=self)

    #def times(self, rhs):
    #    return LGType('prod', fst=self, snd=rhs)

    def __eq__(self, rhs):
        return str(self)==str(rhs)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #~~~~~~~~~  0.1 Display  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    def __str__(self):
        if self.kind=='base':
            return self.name
        elif self.kind=='from':
            return '{} <- {}'.format(
                ('({})' if self.out.kind in ('prod') else '{}').format(
                    str(self.out)
                ),
                ('({})' if self.arg.kind in ('from', 'prod') else '{}').format(
                    str(self.arg)
                )
            )
        elif self.kind=='mset':
            return '{{{}}}'.format(str(self.child))
        elif self.kind=='prod':
            return '{} x {}'.format(str(self.fst), str(self.snd))

    def __hash__(self):
        return hash(str(self))

    def conseq_hypoth_pairs(self):
        '''
        TODO: add constructor names (fst, snd) for product?
        '''
        if self.kind=='base': return [([self], [])]
        if self.kind=='from':
            return [([self], [])] + [
                (conseqs, hypoths + [self.arg])
                for conseqs, hypoths in self.out.conseq_hypoth_pairs()
            ]
        if self.kind=='mset': return [([self], [])]
        #if self.kind=='prod':
        #    return [(self.fst.conseqs()+self.snd.conseqs(), [])]

#=============================================================================#
#=====  1. ACTUAL LG TYPES  ==================================================#
#=============================================================================#

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~  1.0 Base Types  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

tNoise          = LGType('base', name='noise')
tInt            = LGType('base', name='int')
tColor          = LGType('base', name='color')
tShape          = LGType('base', name='shape')
tGrid           = LGType('base', name='grid')

tCell           = LGType('base', name='cell')  # (int, int)
tDir            = LGType('base', name='dir')   # (int, int) 
tNmbrdBlock     = LGType('base', name='nmbrdblock')
tBlock          = LGType('base', name='block') # (shape, color)
tClrdCell       = LGType('base', name='clrdcell')
tPtdGrid        = LGType('base', name='ptdgrid')
tGridPair       = LGType('base', name='gridpair')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~  1.2 Templates for Fake Polymorphism  ~~~~~~~~~~~~~~~~~~~~~~~~~#

tPred_   = lambda t:   tInt.frm(t)
tRepeat_ = lambda t:   t.s().frm(tInt)
tCount_  = lambda t:   tInt.frm(t.s())
tLen_    = lambda t:   tInt.frm(t.s())
tFilter_ = lambda t:   t.s().frm(tInt.frm(t)).frm(t.s())
tArgmax_ = lambda t:   t.frm(tInt.frm(t)).frm(t.s())
tMap_    = lambda t,u: t.s().frm(t.frm(u)).frm(u.s())
tBinop_  = lambda t:   t.frm(t).frm(t)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~  1.3 Examples of Display  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

if __name__ == '__main__':
    #print('colored block  \t', tBlock.times(tColor))
    print('blocks         \t', tBlock.s())
    print('decompose block\t', tCell.s().frm(tBlock))
    print('count cells    \t', tCount_(tCell))
    print('argmax blocks  \t', tArgmax_(tBlock))
    print('filter blocks  \t', tFilter_(tBlock))
    print('map            \t', tMap_(tColor,tBlock))
    print('classify color \t', tPred_(tColor))
    print('binop on grids \t', tBinop_(tGrid))
