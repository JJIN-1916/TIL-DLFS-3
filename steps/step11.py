import unittest

import numpy as np


class Variable:
    def __init__(self, data):
        if data is not None:
             if not isinstance(data, np.ndarray):
                  raise TypeError('{}은(는) 지원하지 않습니다.'.format(type(data)))

        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
         self.creator = func

    def backward(self):
      if self.grad is None:
           self.grad = np.ones_like(self.data)

      # 반복문을 사용한 구현
      funcs = [self.creator]
      while funcs:
           f = funcs.pop() # 함수를 가져온다
           x, y = f.input, f.output # 함수의 입력과 출력을 가져온다.
           x.grad = f.backward(y.grad) # backward 메서드를 호출한다.

           if x.creator is not None:
                funcs.append(x.creator) # 하나 앞의 함수를 리스트에 추가한다.


class Function:
    def __call__(self, inputs):
        # x = input.data # Variable 이라는 '상자'에서 실제 데이터를 꺼낸 다음
        xs = [x.data for x in inputs]
        # y = self.forward(x) # forward 메서드에서 구체적인 계산을 함
        ys = self.forward(xs)
        # output = Variable(as_array(y)) # 계산 결과를 Variable에 넣고
        outputs = [Variable(as_array(y)) for y in ys]
        # output.set_creator(self) # 자신이 창조자라고 원산지 표시를 함
        for output in outputs:
             output.set_creator(self)
        
        self.input = input 
        self.output = output 
        return output
    
    def forward(self, x):
          raise NotImplementedError()
    
    def backward(self, gy):
          raise NotImplementedError()
    

class Add(Function):
     def forward(self, xs):
          x0, x1 = xs
          y = x0 + x1
          return (y,)


class Square(Function):
      def forward(self, x):
            return x  ** 2
      
      def backward(self, gy):
            x = self.input.data
            gx = 2 * x * gy
            return gx


class Exp(Function):
      def forward(self, x):
            return np.exp(x)
      
      def backward(self, gy):
            x = self.input.data
            gx = np.exp(x) * gy
            return gx


def as_array(x):
     if np.isscalar(x):
          return np.array(x)
     return x


def square(x):
     return Square()(x)


def exp(x):
     return Exp()(x)


def numerical_diff(f, x, eps=1e-4):
      x0 = Variable(x.data - eps)
      x1 = Variable(x.data + eps)
      y0 = f(x0)
      y1 = f(x1)
      return (y1.data - y0.data) / (2 * eps)


class SquareTest(unittest.TestCase):
     def test_forward(self):
          x = Variable(np.array(2.0))
          y = square(x)
          expected = np.array(4.0)
          self.assertEqual(y.data, expected)

     def test_backward(self):
          x = Variable(np.array(3.0))
          y = square(x)
          y.backward()
          expected = np.array(6.0)
          self.assertEqual(x.grad, expected)

     def test_gradient_check(self):
          x = Variable(np.random.rand(1))
          y = square(x)
          y.backward()
          num_grad = numerical_diff(square, x)
          flg = np.allclose(x.grad, num_grad) # 가까운지 판정!!
          self.assertTrue(flg)


xs = [Variable(np.array(2)), Variable(np.array(3))] # 리스트로 준비
f = Add()
ys = f(xs)
print('ys:', ys.data)

# error 발생 
# y = ys[0] 
# print(y.data)

# unittest.main()