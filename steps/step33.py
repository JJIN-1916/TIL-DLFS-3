if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


import math

import numpy as np
from dezero import Variable
from dezero import Function
from dezero.utils import plot_dot_graph

class Sin(Function):
    def forward(self, x):
        y = np.sin(x)
        return y
    
    def backward(self, gy):
        x = self.inputs[0].data
        gx = gy * np.cos(x)
        return gx
    

def sin(x):
    return Sin()(x)


def my_sin(x, threshold=0.0001):
    y = 0
    for i in range(100000):
        c = (-1) ** i / math.factorial(2 * i + 1)
        t = c * x ** (2 * i + 1)
        y = y + t
        if abs(t.data) < threshold:
            break
    return y


def rosenbrock(x0, x1):
    y = 100 * (x1 - x0 ** 2) ** 2 + (1 - x0) ** 2
    return y


def f(x):
    y = x ** 4 - 2 * x ** 2
    return y


# x = Variable(np.array(2.0))
# y = f(x)
# print('y', y)
# y.backward(create_graph=True)
# print('x.grad', x.grad)

# gx = x.grad
# x.cleargrad() # 미분값 재설정
# gx.backward()
# print('x.grad', x.grad)

x = Variable(np.array(2.0))
iters = 10

for i in range(iters):
    print(i, x)
    y = f(x)
    x.cleargrad()
    y.backward(create_graph=True)

    gx = x.grad
    x.cleargrad()
    gx.backward()
    gx2 = x.grad

    x.data -= gx.data / gx2.data