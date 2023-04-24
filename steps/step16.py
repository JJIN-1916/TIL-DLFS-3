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
        self.generation = 0 # 세대 수를 기록하는 변수

    def set_creator(self, func):
         self.creator = func
         self.generation = func.generation + 1 # 세대를 기록한다(부모 세대 + 1)

    def backward(self):
      if self.grad is None:
           self.grad = np.ones_like(self.data)
      
      funcs = []
      seen_set = set()

      def add_func(f):
           if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)
      
      add_func(self.creator)
      
      while funcs:
           f = funcs.pop() # 함수를 가져온다
           gys = [output.grad for output in f.outputs] # outputs에 담겨있는 미분값들을 리스트에 담는다.
           gxs = f.backward(*gys) # 함수 f의 역전파를 호출한다.
           if not isinstance(gxs, tuple): # gxs가 튜플이 아니라면 튜플로 변환한다.
                gxs = (gxs,) 
           for x, gx in zip(f.inputs, gxs): # 역전파로 전파되는 미분값을 Variable 인스턴스 변수 grad에 저장한다.
                if x.grad is None:
                      x.grad = gx # 미분값을 처음 설정하는 경우 그대로 대입
                else:
                      x.grad = x.grad + gx # 그 다음번부터는 전달된 미분값을 더해주도록 수정

                if x.creator is not None:
                     add_func(x.creator)
               
    def cleargrad(self):
         self.grad = None


class Function:
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(*xs) # 별표를 붙여 언팩
        if not isinstance(ys, tuple): # 튜플이 아닌 경우 추가 지원
          ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        self.generation = max([x.generation for x in inputs])
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
a = square(x)
y = add(square(a), square(a))
y.backward()

print(y.data)
print(x.grad)