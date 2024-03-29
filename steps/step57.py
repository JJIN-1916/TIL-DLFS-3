if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


import numpy as np
import dezero 
from dezero import Variable
import dezero.functions as F


# 1 
# x1 = np.random.rand(1, 3, 7, 7) # 배치 크기 = 1
# col1 = F.im2col(x1, kernel_size=5, stride=1, pad=0, to_matrix=True)
# print(col1.shape)

# x2 = np.random.rand(10, 3, 7, 7) # 배치 크기 = 10
# kernel_size = (5, 5)
# stride = (1, 1)
# pad = (0, 0)
# col2 = F.im2col(x2, kernel_size, stride, pad, to_matrix=True)
# print(col2.shape)

# 2
# from dezero.utils import pair

# print(pair(1))
# print(pair((1, 2)))

# 3 
from dezero.functions_conv import conv2d_simple

N, C, H, W = 1, 5, 15, 15
OC, (KH, KW) = 8, (3, 3)

x = Variable(np.random.randn(N, C, H, W))
W = np.random.randn(OC, C, KH, KW)
y = conv2d_simple(x, W, b=None, stride=1, pad=1)
y.backward()

print(y.shape)
print(x.grad.shape)