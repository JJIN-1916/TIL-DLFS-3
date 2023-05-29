if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


import math

import numpy as np
import dezero.functions as F
from dezero import Variable
from dezero.utils import plot_dot_graph 


x = Variable(np.array(1.0))
y = F.tanh(x)
x.name = 'x'
y.name = 'y'
y.backward(create_graph=True)

# iters = 0 # 1차 미분
# iters = 1 # 2차 미분
iters = 6

for i in range(iters):
    gx = x.grad
    x.cleargrad()
    gx.backward(create_graph=True)


# 계산 그래프 그리기 
gx = x.grad
gx.name = 'gx' + str(iters+1)
plot_dot_graph(gx, verbose=False, to_file=f'./dot/tanh_{iters+1}.png')