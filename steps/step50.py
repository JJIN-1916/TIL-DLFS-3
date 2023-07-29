if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


import math

import numpy as np
import matplotlib.pyplot as plt
from dezero import Variable
from dezero import optimizers
import dezero.functions as F
from dezero.models import MLP
from dezero.datasets import get_spiral

# Variable.__getitem__ = F.get_item # Variable의 메서드로 설정

# # 1
# t = [1, 2, 3]
# x = iter(t)
# print(next(x))
# print(next(x))
# print(next(x))
# # print(next(x))

# 2
# class MyIterator:
#     def __init__(self, max_cut):
#         self.max_cut = max_cut
#         self.cnt = 0

#     def __iter__(self):
#         return self
    
#     def __next__(self):
#         if self.cnt == self.max_cut:
#             raise StopIteration
        
#         self.cnt += 1
#         return self.cnt

# obj = MyIterator(5)
# for x in obj:
#     print(x)

# 3 
# from dezero.datasets import Spiral
# from dezero import DataLoader

# batch_size = 10
# max_epoch = 1

# train_set = Spiral(train=True)
# test_set = Spiral(train=False)
# train_loader = DataLoader(train_set, batch_size)
# test_loader = DataLoader(test_set, batch_size, shuffle=False)

# for epoch in range(max_epoch):
#     for x, t in train_loader:
#         print(x.shape, t.shape)
#         break

#     for x, t in test_loader:
#         print(x.shape, t.shape)
#         break

# # 4
# import numpy as np
# import dezero.functions as F

# y = np.array([[0.2, 0.8, 0], [0.1, 0.9, 0], [0.8, 0.1, 0.1]])
# t = np.array([1, 2, 0])
# acc = F.accuracy(y, t)
# print(acc)

import dezero
from dezero import DataLoader


max_epoch = 300
batch_size = 30
hidden_size = 10
lr = 1.0

train_set = dezero.datasets.Spiral(train=True)
test_set = dezero.datasets.Spiral(train=False)
train_loader = DataLoader(train_set, batch_size)
test_loader = DataLoader(test_set, batch_size)

model = MLP((hidden_size, 3))
optimizer = optimizers.SGD(lr).setup(model)

epoch_ls = []
train_loss_ls = []
test_loss_ls = []
train_acc_ls = []
test_acc_ls = []

for epoch in range(max_epoch):
    epoch_ls.append(epoch)
    sum_loss, sum_acc = 0, 0

    for x, t in train_loader:
        y = model(x)
        loss = F.softmax_cross_entropy(y, t)
        acc = F.accuracy(y, t)
        model.cleargrads()
        loss.backward()
        optimizer.update()
        sum_loss += float(loss.data) * len(t)
        sum_acc += float(acc.data) * len(t)

    train_loss = sum_loss/len(train_set)
    train_loss_ls.append(train_loss)
    train_acc = sum_acc/len(train_set)
    train_acc_ls.append(train_acc)
    print('epoch: {}'.format(epoch+1))
    print('train loss: {:.4f}, accuracy: {:.4f}'.format(
        train_loss, train_acc))

    sum_loss, sum_acc = 0, 0
    with dezero.no_grad():
        for x, t in test_loader:
            y = model(x)
            loss = F.softmax_cross_entropy(y, t)
            acc = F.accuracy(y, t)
            sum_loss += float(loss.data) * len(t)
            sum_acc += float(acc.data) * len(t)
        
    test_loss = sum_loss/len(test_set)
    test_loss_ls.append(test_loss)
    test_acc = sum_acc/len(test_set)
    test_acc_ls.append(test_acc)
    print('test loss: {:.4f}, accuracy: {:.4f}'.format(
        test_loss, test_acc
    ))
fig, ax = plt.subplots(1, 2, figsize=(8, 4))

ax[0].plot(epoch_ls, train_loss_ls)
ax[0].plot(epoch_ls, test_loss_ls)
ax[0].set_xlabel('epoch')
ax[0].set_ylabel('loss')

ax[1].plot(epoch_ls, train_acc_ls)
ax[1].plot(epoch_ls, test_acc_ls)
ax[1].set_xlabel('epoch')
ax[1].set_ylabel('acc')

plt.tight_layout()
plt.savefig('./steps/step50_plot.png')
plt.show()
