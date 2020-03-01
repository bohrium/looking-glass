''' author: samtenka
    change: 2020-03-01
    create: 2019-02-26
    descrp: Translate textual code to python function or program tree. 
    to use: Obtain the tree of some text `code` as follows:

                from parse import Parser
                tree = Parser(code).get_tree()

            We use a light-weight representation of trees (in terms of Python's
            built-in types instead of a new class): strings name atoms; lists
            represent potentially-multiple application; and single-item
            dictionaries represent lambdas, where the key is a tuple giving the
            name and type of the introduced variable and value is the lambda
            body.  For example, the following LISP style text:

                (map_over my_elts (\elt:int -> (plus elt five)))

            would be parsed into the following tree (a Python object):

                ['map_over',
                 'my_elts',
                  {('elt', tInt): ['plus', 'elt', 'five']}
                ]
'''

import numpy as np

from utils import CC, pre   # ansi

from lg_types import TS

class Parser:
    '''
    '''

    ALPHA = 'abcdefghijklmnopqrstuvwxyz'
    ALPHA_NUM = 'abcdefghijklmnopqrstuvwxyz_<>0123456789'

    def __init__(self, string):
        self.string = string
        self.i=0

    def get_tree(self):
        self.skip_space()
        tree = self.get_term()
        pre(self.at_end(),
            'parsing failure near: ...@G {}@R {}...'.format(
                self.string[:self.i][-50:],
                self.string[self.i:][:50]
            )
        )
        return tree

    def at_end(self):
        return self.i==len(self.string)
    def peek(self):
        return self.string[self.i] if self.i!=len(self.string) else '\0'
    def march(self):
        self.i+=1
    def skip_space(self):
        while not self.at_end() and (self.peek() in ' \n'):
            self.march()
    def match(self, s):
        ''' matches then skips space'''
        prefix = self.string[self.i:][:len(s)]
        pre(prefix==s, 'expected `{}` but saw `{}`'.format(s, prefix))
        self.i+=len(s)
        self.skip_space()

    def get_identifier(self): 
        '''
        '''
        old_i = self.i
        while self.peek() in Parser.ALPHA_NUM: self.march()
        nm = self.string[old_i:self.i]
        self.skip_space()
        return nm 

    def get_type(self): 
        t = self.get_hypothesis_free_type()
        while self.peek() == '<':
            self.match('<-')
            hypo = self.get_hypothesis_free_type() 
            t = t.frm(hypo)
        return t 

    def get_hypothesis_free_type(self): 
        '''
        '''
        if self.peek()=='{':
            self.match('{')
            t = self.get_type().s()
            self.match('}')
        elif self.peek()=='(':
            self.match('(')
            t = self.get_type()
            self.match(')')
        elif self.peek() in Parser.ALPHA:
            nm = self.get_identifier()
            t = TS.base_types_by_nm[nm]
        else:
            pre(False, 'unknown symbol when parsing type!')
        return t

    def get_term(self): 
        '''
        '''
        if self.peek()=='(':
            self.match('(')
            tree = [self.get_term()]
            while self.peek()!=')': 
                tree.append(self.get_term())
            self.match(')')
        elif self.peek()=='\\':
            self.match('\\')
            var_nm = self.get_identifier()
            self.match(':')
            t = self.get_type()
            self.match('->')
            body = self.get_term() 
            tree = {(var_nm, t):body} 
        elif self.peek() in Parser.ALPHA:
            tree = self.get_identifier()
        else:
            pre(False, 'unknown character `{}`'.format(self.peek()))
        return tree

if __name__=='__main__':
    code = '(map_over my_elts (\\elt:int -> (plus elt five)))'
    tree = Parser(code).get_tree()
    print(tree)
