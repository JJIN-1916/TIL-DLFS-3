import numpy as np


class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
         self.creator = func

    def backward(self):
      # 재귀를 사용한 구현
      #    f = self.creator
      #    if f is not None:
      #       x = f.input
      #       x.grad = f.backward(self.grad)
      #       x.backward()
      # 반복문을 사용한 구현
      funcs = [self.creator]
      while funcs:
           f = funcs.pop() # 함수를 가져온다
           x, y = f.input, f.output # 함수의 입력과 출력을 가져온다.
           x.grad = f.backward(y.grad) # backward 메서드를 호출한다.

           if x.creator is not None:
                funcs.append(x.creator) # 하나 앞의 함수를 리스트에 추가한다.


class Function:
    def __call__(self, input):
        x = input.data # 데이터를 꺼낸다.
        # y = x ** 2 # 실제 계산
        y = self.forward(x)
        output = Variable(y) # Variable 형태로 되돌린다.
        output.set_creator(self) # 출력 변수에 창조자를 설정한다.
        self.input = input # 입력 변수를 기억(보관)한다.
        self.output = output # 출력도 저장한다.
        return output
    
    def forward(self, x):
          raise NotImplementedError()
    
    def backward(self, gy):
          raise NotImplementedError()
    

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
      

def numerical_diff(f, x, eps=1e-4):
      x0 = Variable(x.data - eps)
      x1 = Variable(x.data + eps)
      y0 = f(x0)
      y1 = f(x1)
      return (y1.data - y0.data) / (2 * eps)


A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)

# 역전파
y.grad = np.array(1.0)
y.backward()
print(x.grad)