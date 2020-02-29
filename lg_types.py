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
        if self.kind=='base':
            return self.name
        elif self.kind=='from':
            return '{}_by_{}'.format(str(self.out), str(self.arg))
        elif self.kind=='mset':
            return '{}s'.format(str(self.child))

    def __hash__(self):
        return hash(repr(self))

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
tNmbrdColor     = LGType('base', name='nmbrdcolor')
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
    print('blocks         \t', repr(tBlock.s()))
    print('decompose block\t', repr(tCell.s().frm(tBlock)))
    print('count cells    \t', repr(tCount_(tCell)))
    print('argmax blocks  \t', repr(tArgmax_(tBlock)))
    print('filter blocks  \t', repr(tFilter_(tBlock)))
    print('map            \t', repr(tMap_(tColor,tBlock)))
    print('classify color \t', repr(tPred_(tColor)))
    print('binop on grids \t', repr(tBinop_(tGrid)))
