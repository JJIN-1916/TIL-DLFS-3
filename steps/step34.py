if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


import math

import numpy as np
import matplotlib.pyplot as plt
import dezero.functions as F
from dezero import Variable


x = Variable(np.linspace(-7, 7, 200))
y = F.sin(x)
y.backward(create_graph=True)

logs = [y.data]

for i in range(3):
    print('-'*100)
    print(i, x.grad.data)
    print('-'*100)
    logs.append(x.grad.data)
    gx = x.grad
    print('gx', gx)
    x.cleargrad()
    print('x.grad1',x.grad)
    gx.backward(create_graph=True)
    print('x.grad2',x.grad)


# 그래프 그리기 
labels = ["y=sin(x)", "y'", "y''", "y'''"]
for i, v in enumerate(logs):
    plt.plot(x.data, logs[i], label=labels[i])
plt.legend(loc='lower right')
plt.savefig('./steps/step34_plot.png')
plt.show()