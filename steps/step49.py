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

# 1
import dezero
# train_set = dezero.datasets.Spiral(train=True)
# print(train_set[0])
# print(len(train_set))

train_set = dezero.datasets.Spiral(train=True)

# 전반 코드
max_epoch = 300
batch_size = 30
hidden_size = 10
lr = 1.0

model = MLP((hidden_size, 3))
optimizer = optimizers.SGD(lr).setup(model)

# 후반 코드
data_size = len(train_set)
max_iter = math.ceil(data_size/batch_size) # 소수점 반올림

for epoch in range(max_epoch):
    # 데이터셋 인덱스 뒤섞기
    index = np.random.permutation(data_size)
    sum_loss = 0

    for i in range(max_iter):
        # 미니배치 생성
        batch_index = index[i * batch_size : (i + 1) * batch_size]
        batch = [train_set[i] for i in batch_index]
        batch_x = np.array([example[0] for example in batch])
        batch_t = np.array([example[1] for example in batch])

        # 기울기 산출 / 매개변수 갱신
        y = model(batch_x)
        loss = F.softmax_cross_entropy(y, batch_t)
        model.cleargrads()
        loss.backward()
        optimizer.update()
        sum_loss += float(loss.data) * len(batch_t)

    # 에포크마다 학습 경과 출력
    avg_loss = sum_loss / data_size
    print('epoch %d, loss %.2f' % (epoch + 1, avg_loss))

    plt.plot(epoch+1, avg_loss, marker='.', color='blue')

# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.savefig('./steps/step48_loss_plot.png')
# plt.figure().clear()
# # plt.show()

# markers = ['o', '^', 'x']
# colors = ['peachpuff', 'yellowgreen', 'lightsteelblue']

# for x_0 in np.linspace(-1, 1, 100):
#     for x_1 in np.linspace(-1, 1, 100):
#         test_y = model(Variable(np.array([x_0, x_1])))
#         test_p = F.sigmoid_simple(test_y)
#         test_y2 = np.array(test_p.data).argmax()
#         plt.plot(x_0, x_1, color=colors[test_y2], marker='o')

# colors = ['orange', 'green', 'blue']
# for x_i, t_i in zip(x, t):
#     plt.plot(x_i[0], x_i[1], marker=markers[t_i], color=colors[t_i])


# plt.savefig('./steps/step48_results.png')
# # plt.figure().clear()
# plt.show()
