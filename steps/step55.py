if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


import numpy as np
import dezero 
import dezero.functions as F
from dezero import test_mode


# 1 (합성곱, 패딩, 스트라이드를 활용한 출력 크기 계산 방법)
def get_conv_outsize(input_size, kernel_size, stride, pad):
    return(input_size + pad * 2 - kernel_size) // stride + 1

H, W = 4, 4 # input size
KH, KW = 3, 3 # kernel size
SH, SW = 1, 1 # stride(세로, 가로)
PH, PW = 1, 1 # padding (세로, 가로)

OH = get_conv_outsize(H, KH, SH, PH)
OW = get_conv_outsize(W, KW, SW, PW)
print(OH, OW)