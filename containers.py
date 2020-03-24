''' author: samtenka
    change: 2020-03-23
    create: 2019-03-23
    descrp: 
    to use: 

'''

from utils import uniform # math

class ListByKey: 
    '''
        Maintain a key->[val] map with easy adding and sampling.
    '''

    def __init__(self):
        self.data = {}

    def add(self, key, val):
        if key not in self.data:
            self.data[key] = []
        self.data[key].append(val)

    def keys(self):
        return self.data.keys()

    def sample(self, key):
        return uniform(self.data[key])

    def len_at(self, key):
        return len(self.data[key])

class Index:
    ''' Maintain a bijection from given hashable objects to contiguous Ints '''

    def __init__(self, items=set()):
        self.indices_by_elt = { v:i for i,v in enumerate(sorted(items)) }

    def __str__(self):
        return str(self.indices_by_elt)

    def __len__(self):
        return len(self.indices_by_elt)

    def __contains__(self, elt):
        return elt in self.indices_by_elt

    def add(self, elt):
        if elt in self.indices_by_elt: return
        self.indices_by_elt[elt] = len(self.indices_by_elt) 

    def idx(self, elt):
        return self.indices_by_elt[elt] if elt in self.indices_by_elt else None


