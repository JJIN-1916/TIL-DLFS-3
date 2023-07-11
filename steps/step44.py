if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


import math

import numpy as np
from dezero.core import Variable, Parameter
from dezero.layers import Layer
import dezero.functions as F
import dezero.layers as L 

# 1
# x = Variable(np.array(1.0))
# p = Parameter(np.array(2.0))
# y = x * p

# print(isinstance(p, Parameter))
# print(isinstance(x, Parameter))
# print(isinstance(y, Parameter))

# 2
# layer = Layer()
# layer.p1 = Parameter(np.array(1.0))
# layer.p2 = Parameter(np.array(2.0))
# layer.p3 = Variable(np.array(3.0))
# layer.p4 = 'test'

# print(layer._params)
# print('-----------------')

# for name in layer._params:
#     print(name, layer.__dict__[name])

# 3
# class Linear(Layer):
#     def __init__(self, in_size, out_size, nobias=False, dtype=np.float32):
#         super().__init__()

#         I, O = in_size, out_size
#         W_data = np.random.randn(I, O).astype(dtype) * np.sqrt(1 / I)
#         self.W = Parameter(W_data, name='W')
#         if nobias:
#             self.b = None
#         else:
#             self.b = Parameter(np.zeros(0, dtype=dtype), name='b')

#     def forward(self, x):
#         y = F.linear_simple(x, self.W, self.b)
#         return y

# 데이터셋
np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1) # 데이터 생성에 sin 함수 이용

## 추가
l1 = L.Linear(10) # 출력 크기 지정
l2 = L.Linear(1)

# 신경망 추론
def predict(x):
    # y = F.linear_simple(x, W1, b1)
    y = l1(x)
    y = F.sigmoid_simple(y)
    # y = F.linear_simple(y, W2, b2)
    y = l2(y)
    return y

lr = 0.2
iters = 10000

# 신경망 학습
for i in range(iters):
    y_pred = predict(x)
    loss = F.mean_squared_error(y, y_pred)

    # W1.cleargrad()
    # b1.cleargrad()
    l1.cleargrads()
    # W2.cleargrad()
    # b2.cleargrad()
    l2.cleargrads()
    loss.backward()

    # W1.data -= lr * W1.grad.data
    # b1.data -= lr * b1.grad.data
    # W2.data -= lr * W2.grad.data
    # b2.data -= lr * b2.grad.data
    for l in [l1, l2]:
        for p in l.params():
            p.data -= lr * p.grad.data
    if i % 1000 == 0: # 1000회마다 출력
        print(loss)

# print(W1, b1, W2, b2)