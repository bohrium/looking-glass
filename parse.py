''' author: samtenka
    change: 2020-02-26
    create: 2019-02-26
    descrp: translate textual code to python function or program tree 
    to use: get tree of some textual code as follows:
                from parse import Parser
                tree = Parser(code).get_tree()
'''

import numpy as np

from utils import ARC_path, InternalError
from utils import CC, pre                               # ansi
from utils import secs_endured, megs_alloced            # profiling
from utils import reseed, bernoulli, geometric, uniform # math

from shape import ShapeGen 
from block import GENERIC_COLORS, Block, block_equals
from grid import Grid

from vis import str_from_grids, render_color

from lg_types import tInt, tCell, tColor, tBlock, tGrid 

class Parser:
    def __init__(self, string):
        self.string = string
        self.i=0

    def get_tree(self):
        self.skip_space()
        tree = self.get_term()
        pre(self.at_end(), 'unable to parse whole string'+'#{}#'.format(self.string[self.i:]))
        return tree

    def at_end(self):
        return self.i==len(self.string)
    def peek(self):
        return self.string[self.i] if self.i!=len(self.string) else '\0'
    def match(self, s):
        assert self.string[self.i:self.i+len(s)]==s
        self.i+=len(s)
    def march(self):
        self.i+=1
    def skip_space(self):
        while not self.at_end() and (self.peek() in ' \n'):
            self.march()

    def get_identifier(self): 
        old_i = self.i
        while self.peek() in 'abcdefghijklmnopqrstuvwxyz_': self.march()
        return self.string[old_i:self.i]

    def get_term(self): 
        if self.peek()=='(':
            self.match('(')
            self.skip_space()
            tree = [self.get_term()]
            while self.peek()!=')': 
                tree.append(self.get_term())
            self.match(')')
        elif self.peek()=='\\':
            self.match('\\')
            var_nm = self.get_identifier()
            self.skip_space()
            self.match(':')
            self.skip_space()
            type_nm = self.get_identifier()
            self.skip_space()
            self.match('->')
            self.skip_space()
            body = self.get_term() 
            tree = {(var_nm, type_nm):body} 
        elif self.peek() in 'abcdefghijklmnopqrstuvwxyz':
            tree = self.get_identifier()
        else:
            pre(False, 'unknown character #{}#'.format(self.peek()))

        self.skip_space()
        return tree

