if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


import numpy as np
import dezero 
import dezero.functions as F
from dezero import test_mode


# 1 (학습 시 dropout)
# dropout_ratio = 0.6
# x = np.ones(10)
# mask = np.random.rand(10) > dropout_ratio 
# # 0.0~1.0 사이의 값을 임의로 10개 생성, dropout_ratio 와 비교하여 큰 원소는 True, 작은 원소는 False
# # 이렇게 생성된 mask는 False의 비율이 평균적으로 60% 
# y = x * mask
# # False에 대응하는 원소 x의 원소를 0으로 설정한다.(즉, 삭제)
# print(y)

# 2 (테스트 시 dropout)
# scale = 1 - dropout_ratio
# y = x * scale
# print(y)

# 3 (역 드롭아웃)
# # 학습 시
# scale = 1 - dropout_ratio
# mask = np.random.rand(10) > dropout_ratio 
# y = x * mask / scale
# print(y)

# # 테스트 시 
# y = x
# print(y)

# 4 
x = np.ones(5)
print(x)

# train
y = F.dropout(x)
print(y)

# test
with test_mode():
    y = F.dropout(x)
    print(y)