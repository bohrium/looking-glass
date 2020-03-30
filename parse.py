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

def nb_nodes(tree):
    if type(tree) == str:
        return 1
    elif type(tree) == list:
        return sum(nb_nodes(elt) for elt in tree)
    elif type(tree) == dict: 
        for (nm, t), body in tree.items():
            return 1+nb_nodes(body)

def get_height(tree):
    if type(tree) == str:
        return 0
    elif type(tree) == list:
        print(tree)
        return 1+max(get_height(elt) for elt in tree[1:])
    elif type(tree) == dict: 
        for (nm, t), body in tree.items():
            return 1+get_height(body)

def str_from_tree_flat(tree):
    if type(tree) == str:
        return tree
    elif type(tree) == list:
        return '({})'.format(' '.join(map(str_from_tree_flat, tree)))
    elif type(tree) == dict: 
        for (nm, t), body in tree.items():
            return '\\{}:{} -> {}'.format(nm, t, str_from_tree_flat(body)) 

def str_from_tree(tree, parent='', depth=0):
    if type(tree) == str:
        return tree
    elif type(tree) == list:
        new_depth = depth + (0 if parent[:4] in tree[0][:4] else 1)

        as_flats = [str_from_tree_flat(k) for k in tree]
        best_indices = sorted((len(f)+1, i) for i,f in enumerate(as_flats)) 
        indices_to_expand = []
        len_sum = 0
        for l,i in best_indices:
            len_sum += l
            if not (len_sum + 2*new_depth < 80): break
            indices_to_expand.append(i)

        return (
            '(\n' + '  '*new_depth +
            ' '.join(
                as_flats[i] if i in indices_to_expand else
                str_from_tree(k, parent=tree[0], depth=new_depth)
                for i, k in enumerate(tree)
            ) +
            '\n' + '  '*depth + ')'
        )
    elif type(tree) == dict:
        for (nm, tp), body in tree.items():
            return (
                '\\{}:{} -> '.format(nm, tp) +
                str_from_tree(body, parent=parent, depth=depth).strip()
            )


class Parser:
    '''
    '''

    ALPHA = 'abcdefghijklmnopqrstuvwxyz'
    ALPHA = ALPHA + ALPHA.upper()
    ALPHA_NUM = ALPHA + '_0123456789'
    BRACKETS = '<>-{}'

    def __init__(self, string):
        self.string = '\n'.join(
            ln for ln in string.split('\n') if not ln.startswith('//')
        )
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


