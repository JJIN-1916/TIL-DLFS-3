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
          # 수정 전
          #  x, y = f.input, f.output # 함수의 입력과 출력을 가져온다.
          #  x.grad = f.backward(y.grad) # backward 메서드를 호출한다.
          # 수정 후
           gys = [output.grad for output in f.outputs] # outputs에 담겨있는 미분값들을 리스트에 담는다.
           gxs = f.backward(*gys) # 함수 f의 역전파를 호출한다.
           if not isinstance(gxs, tuple): # gxs가 튜플이 아니라면 튜플로 변환한다.
                gxs = (gxs,) 
           for x, gx in zip(f.inputs, gxs): # 역전파로 전파되는 미분값을 Variable 인스턴스 변수 grad에 저장한다.
                x.grad = gx

                if x.creator is not None:
                     funcs.append(x.creator) # 하나 앞의 함수를 리스트에 추가한다.


class Function:
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(*xs) # 별표를 붙여 언팩
        if not isinstance(ys, tuple): # 튜플이 아닌 경우 추가 지원
          ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]
        for output in outputs:
             output.set_creator(self)
        
        self.inputs = inputs 
        self.outputs = outputs 
        return outputs if len(outputs) > 1 else outputs[0]
     
    def forward(self, x):
          raise NotImplementedError()
    
    def backward(self, gy):
          raise NotImplementedError()
    

class Add(Function):
     def forward(self, x0, x1):
          y = x0 + x1
          return y
     
     def backward(self, gy):
          return gy, gy


class Square(Function):
      def forward(self, x):
            return x  ** 2
      
      def backward(self, gy):
          # 수정 전
          #   x = self.input.data
          # 수정 후
            x = self.inputs[0].data
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


def add(x0, x1):
     return Add()(x0, x1)


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


x = Variable(np.array(2.0)) 
y = Variable(np.array(3.0))

z = add(square(x), square(y))
z.backward()
print(z.data)
print(x.grad)
print(y.grad)

# unittest.main()