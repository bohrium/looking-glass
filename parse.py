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

def get_depth(tree):
    if type(tree) == str:
        return 0
    elif type(tree) == list:
        return 1+max(get_depth(elt) for elt in tree)
    elif type(tree) == dict: 
        for (nm, t), body in tree.items():
            return 0+max(get_depth(body) for elt in tree)

def str_from_tree_flat(tree):
    if type(tree) == str:
        return tree
    elif type(tree) == list:
        return '({})'.format(' '.join(map(str_from_tree_flat, tree)))
    elif type(tree) == dict: 
        for (nm, t), body in tree.items():
            return '\\{}:{} -> {}'.format(nm, t, str_from_tree_flat(body)) 

def str_from_tree(tree, depth=0, delim='   '):
    tab = delim*depth

    as_flat = '{}{}'.format(tab, str_from_tree_flat(tree))
    if len(as_flat) < 80:
        return as_flat

    if type(tree) == str:
        return '{}{}'.format(tab, tree)
    elif type(tree) == list:
        if type(tree[0])==str:
            for prefix, (length, dd) in {'split':(3,0), 'repeat':(4,1), 'fold':(4,1)}.items(): 
                if not(len(tree)==length and tree[0].startswith(prefix)): continue
                lines = list(str_from_tree(elt, depth+dd) for elt in tree)
                rtrn = '{}({}'.format(tab, ' '.join(piece.strip() for piece in lines[:-1]))
                rtrn += (' {}\n{})'.format(
                    lines[-1].split('\n')[0].strip(),
                    '\n'.join(lines[-1].split('\n')[1:])
                )) if '\n' in lines[-1] else '{}\n'.format(lines[-1].strip()) 
                return rtrn
        lines = list(str_from_tree(elt, depth+1) for elt in tree)
        rtrn = '{}({}'.format(tab, lines[0].lstrip())
        rtrn += ('\n'+'\n'.join(lines[1:])) if lines[1:] else ''
        rtrn += ')'
        return rtrn
    elif type(tree) == dict: 
        for (nm, t), body in tree.items():
            rtrn = '{}\\{}:{} -> \n'.format(tab, nm, t) 
            rtrn += str_from_tree(body, depth+0)
            return rtrn

class Parser:
    '''
    '''

    ALPHA = 'abcdefghijklmnopqrstuvwxyz'
    ALPHA = ALPHA + ALPHA.upper()
    ALPHA_NUM = ALPHA + '_0123456789'
    BRACKETS = '<>-{}'

    def __init__(self, string):
        self.string = string
        self.i=0

    def parse_assert(self, cond, message):
        pre(cond,
            'parse fail: @O {} @D near: ...@G {}@R {}...'.format(
                message,
                self.string[:self.i][-50:],
                self.string[self.i:][:50]
            )
        )


    def get_tree(self):
        self.skip_space()
        tree = self.get_term()
        self.parse_assert(self.at_end(), 'program has extra characters')
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
        self.parse_assert(prefix==s,
            'expected `{}` but saw `{}`'.format(s, prefix)
        )
        self.i+=len(s)
        self.skip_space()

    def get_identifier(self, parameterized=False): 
        '''
        '''
        chars = Parser.ALPHA_NUM + (Parser.BRACKETS if parameterized else '') 

        old_i = self.i
        while self.peek() in chars: self.march()
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
            self.parse_assert(False,
                'type contains foreign symbol `{}`'.format(
                    self.peek()
                )
            )
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
            tree = self.get_identifier(parameterized=True)
        else:
            self.parse_assert(False,
                'term contains foreign symbol `{}`'.format(
                    self.peek()
                )
            )
        return tree

if __name__=='__main__':
    code = '(map_over<> my_elts (\\elt:int -> (plus elt five)))'
    tree = Parser(code).get_tree()
    print(CC + 'from code @P {} @D we find tree \n@O {} @D '.format(
        code, str_from_tree(tree)
    ))

    code = '(map_over my_counters (\\f:int<-{int} -> \\xs:{int} -> (f (wrap (f xs)))))'
    tree = Parser(code).get_tree()
    print(CC + 'from code @P {} @D we find tree \n@O {} @D '.format(
        code, str_from_tree(tree)
    ))


