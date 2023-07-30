if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


import time
import numpy as np
import dezero 
import dezero.functions as F
from dezero import optimizers
from dezero import DataLoader
from dezero.models import MLP

# 1
x = np.array([1,2,3])
np.save('test.npy', x)

x = np.load('test.npy')
print(x)

# 2
# x1 = np.array([1,2,3])
# x2 = np.array([4,5,6])

# np.savez('./steps/step53_test.npz', x1=x1, x2=x2)

# arrays = np.load('./steps/step53_test.npz')
# x1 = arrays['x1']
# x2 = arrays['x2']
# print(x1)
# print(x2)

# 3
# x1 = np.array([1,2,3])
# x2 = np.array([4,5,6])
# data = {'x1':x1, 'x2':x2}

# np.savez('./steps/step53_test.npz', **data)

# arrays = np.load('./steps/step53_test.npz')
# x1 = arrays['x1']
# x2 = arrays['x2']
# print(x1)
# print(x2)

# 4 
import os
import dezero
# import dezero.functions as F
# from dezero import optimizers
# from dezero import DataLoader
# from dezero.models import MLP

from dezero.datasets import MNIST

max_epoch = 5
batch_size = 100

train_set = MNIST(train=True)
train_loader = DataLoader(train_set, batch_size)
model = MLP((1000, 10))
optimizer = optimizers.SGD().setup(model)

# 매개변수 읽기
if os.path.exists('./steps/step53_my_mlp.npz'):
    model.load_weights('./steps/step53_my_mlp.npz')

for epoch in range(max_epoch):
    sum_loss = 0

    for x, t in train_loader:
        y = model(x)
        loss = F.softmax_cross_entropy(y, t)
        model.cleargrads()
        loss.backward()
        optimizer.update()
        sum_loss += float(loss.data) * len(t)

    print('epoch: {}, train loss: {:.4f}'.format(
        epoch+1, sum_loss/len(train_set)))
    
# 매개변수 저장하기
model.save_weights('./steps/step53_my_mlp.npz')
    