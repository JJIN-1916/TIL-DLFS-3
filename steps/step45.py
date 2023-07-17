if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


import math

import numpy as np
from dezero import Variable, Model
import dezero.functions as F
import dezero.layers as L 
import dezero

# 1
# model = Layer()
# model.l1 = L.Linear(5)
# model.l2 = L.Linear(3)

# # 추론을 수행하는 함수
# def predict(model, x):
#     y = model.l1(x)
#     y = F.sigmoid_simple(y)
#     y = model.l2(y)
#     return y

# # 모든 매개변수에 접근
# for p in model.params():
#     print(p)

# # 모든 매개변수의 기울기를 재설정
# model.cleargrads()

# 2
# class TwoLayerNet(Layer):
#     def __init__(self, hidden_size, out_size):
#         super().__init__()
#         self.l1 = L.Linear(hidden_size)
#         self.l2 = L.Linear(out_size)

#     def forward(self, x):
#         y = F.sigmoid_simple(self.l1(x))
#         y = self.l2(y)
#         return y

# 3
# class TwoLayerNet(Model):
#     def __init__(self, hidden_size, out_size):
#         super().__init__()
#         self.l1 = L.Linear(hidden_size)
#         self.l2 = L.Linear(out_size)

#     def forward(self, x):
#         y = F.sigmoid_simple(self.l1(x))
#         y = self.l2(y)
#         return y


# x = Variable(np.random.randn(5, 10), name='x')
# model = TwoLayerNet(100, 10)
# model.plot(x)


# 데이터셋
np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1) # 데이터 생성에 sin 함수 이용

# 하이퍼파라미터 설정
lr = 0.2
max_iter = 10000
hidden_size = 10

# 모델 정의
class TwoLayerNet(Model):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.l1 = L.Linear(hidden_size)
        self.l2 = L.Linear(out_size)

    def forward(self, x):
        y = F.sigmoid_simple(self.l1(x))
        y = self.l2(y)
        return y

model = TwoLayerNet(hidden_size, 1)

# 학습 시작
for i in range(max_iter):
    y_pred = model(x)
    loss = F.mean_squared_error(y, y_pred)

    model.cleargrads()
    loss.backward()

    for p in model.params():
        p.data -= lr * p.grad.data
    if i % 1000 == 0:
        print(loss)

