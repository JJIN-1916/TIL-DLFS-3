if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


import numpy as np
import dezero 
from dezero import Variable
import dezero.functions as F
import dezero.layers as L

# 1
# rnn = L.RNN(10) # 은닉층의 크기만 지정
# x = np.random.rand(1, 1)
# h = rnn(x)
# print(h.shape)

# 2
from dezero import Model

class SimpleRNN(Model):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.rnn = L.RNN(hidden_size)
        self.fc = L.Linear(out_size)

    def reset_state(self):
        self.rnn.reset_state()

    def forward(self, x):
        h = self.rnn(x)
        y = self.fc(h)
        return y

# seq_data = [np.random.randn(1, 1) for _ in range(1000)] # 더미 시계열 데이터
# xs = seq_data[0:-1]
# ts = seq_data[1:] # 정답 데이터 : xs 보다 한단계 앞선 데이터

# model = SimpleRNN(10, 1)

# loss, cnt = 0, 0
# for x, t in zip(xs, ts):
#     y = model(x)
#     loss += F.mean_squared_error(y, t)
#     cnt += 1
#     if cnt == 2:
#         model.cleargrads()
#         loss.backward()
#         break

# 3
# import matplotlib.pyplot as plt

# train_set = dezero.datasets.SinCurve(train=True)
# print(len(train_set))
# print(train_set[0])
# print(train_set[1])
# print(train_set[2])

# # 그래프 그리기
# xs = [example[0] for example in train_set]
# ts = [example[1] for example in train_set]
# plt.plot(np.arange(len(xs)), xs, label='xs')
# plt.plot(np.arange(len(ts)), xs, label='ts')
# plt.savefig('./steps/step59_trainset_plot.png')
# plt.show()

# 4
import matplotlib.pyplot as plt

# 하이퍼파라미터 설정
max_epoch = 100
hidden_size = 100
bptt_length = 30 # BPTT 길이
train_set = dezero.datasets.SinCurve(train=True)
seqlen = len(train_set)

model = SimpleRNN(hidden_size, 1)
optimizer = dezero.optimizers.Adam().setup(model)

# 학습 시작
for epoch in range(max_epoch):
    model.reset_state()
    loss, count = 0, 0

    for x, t in train_set:
        x = x.reshape(1, 1) # 형상을 (1, 1)로 변환
        y = model(x)
        loss += F.mean_squared_error(y, t)
        count += 1

        # Truncated BPTT 의 타이밍 조정
        if count % bptt_length == 0 or count == seqlen:
            model.cleargrads()
            loss.backward()
            loss.unchain_backward() # 연결 끊기
            optimizer.update()

    avg_loss = float(loss.data) / count
    print('| epoch %d | loss %f' % (epoch+1, avg_loss))

xs = np.cos(np.linspace(0, 4 * np.pi, 1000))
model.reset_state() # 모델 재설정
pred_list = []

with dezero.no_grad():
    for x in xs:
        x = np.array(x).reshape(1, 1)
        y = model(x)
        pred_list.append(float(y.data))

plt.plot(np.arange(len(xs)), xs, label='y=cos(x)')
plt.plot(np.arange(len(xs)), pred_list, label='predict')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.savefig('./steps/step59_pred_plot.png')
plt.show()