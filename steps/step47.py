if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


import math

import numpy as np
from dezero import Variable, as_variable
from dezero import optimizers
import dezero.functions as F
from dezero.models import MLP

Variable.__getitem__ = F.get_item # Variable의 메서드로 설정

# # 1
# x = Variable(np.array([[1,2,3], [4,5,6]]))
# y = F.get_item(x, 1)
# print(y)

# y.backward()
# print(x.grad)

# # 2
# x = Variable(np.array([[1,2,3], [4,5,6]]))
# indices = np.array([0, 0, 1])
# y = F.get_item(x, indices)
# print(y)

# # 3
# y = x[1]
# print(y)

# y = x[:, 2]
# print(y)

# # 4
# def softmax1d(x):
#     x = as_variable(x)
#     y = F.exp(x)
#     sum_y = F.sum(y)
#     return y / sum_y

# model = MLP((10, 3))

# x = np.array([[0.2, -0.4]])
# y = model(x)
# p = softmax1d(y)

# print(y)
# print(p)

# 5
model = MLP((10, 3))
x = np.array([[0.2, -0.4], [0.3, 0.5], [1.3, -3.2], [2.1, 0.3]])
t = np.array([2, 0, 1, 0])
y = model(x)
loss = F.softmax_cross_entropy_simple(y, t)
print(loss)
