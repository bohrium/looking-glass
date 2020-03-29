''' author: samtenka
    change: 2020-03-17
    create: 2019-02-23
    descrp: visualize grids
    to use: 
'''

import numpy as np

from utils import CC, pre                       # ansi
from utils import secs_endured, megs_alloced    # profiling
from utils import reseed, bernoulli, geometric  # math

from small_trainset import get_grids, get_hardness

colors = 'KBRGYAPOCN'
render_color = (
    lambda c:
        '@{} \u2588\u2588@D '.format(c)
        if c!='K' else '@D  \u00b7@D '
)
render_number = (lambda n: render_color(colors[n]))

def str_from_grids(grids, render=render_number): 
    ''' Return a colorizable string given a list of (potentially non-uniformly
        shaped) grids of numbers or of colors.
    '''
    if not grids:
        return ''
    heights, widths = ([(g.shape[i] if i<len(g.shape) else 0) for g in grids] for i in range(2))

    lines = ['' for h in range(2+max(heights))]
    for g, H, W in zip(grids, heights, widths):
        lines[0]   += ' {} '.format('_'*2*W)                    # top
        for r in range(H):
            lines[1+r] += '|' + ''.join(                        # content
                render(g[r,c])
                for c in range(W)
            ) + '|'
        lines[1+H] += '`{}`'.format('`'*2*W)                    # bottom
        for h in range(1+H+1, len(lines)):                          
            lines[h] += ' {} '.format(' '*2*W)                  # under padding
    return '\n'.join(lines)

if __name__=='__main__':
    HARDNESS = 1
    print(CC+'hi!  here are some @R actual ARC @D tasks!')
    print(CC+'only showing hardness @P {} @D tasks...'.format(HARDNESS))
    for i in range(100):
        if get_hardness(i)!=HARDNESS: continue
        print(CC + 'task @O {}@D '.format(i))

        total_width = 0
        grids = [] 
        for x, y in get_grids(i, 'train')+get_grids(i, 'test'):
            grids += [x, y, np.zeros((0,0))]
            total_width += 2*x.shape[1]+2 + 2*y.shape[1]+2
            if total_width>=60: 
                print(CC + str_from_grids(grids))
                total_width=0
                grids=[]
        if grids:
            print(CC + str_from_grids(grids))

        input('next?')
