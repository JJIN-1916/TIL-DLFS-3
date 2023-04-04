import numpy as np


class Variable:
	def __init__(self, data):
		self.data = data


class Function:
    def __call__(self, input):
        x = input.data # 데이터를 꺼낸다.
        # y = x ** 2 # 실제 계산
        y = self.forward(x)
        output = Variable(y) # Variable 형태로 되돌린다.
        return output
    
    def forward(self, x):
          raise NotImplementedError()
    

class Square(Function):
      def forward(self, x):
            return x  ** 2


x = Variable(np.array(10))
# f = Function()
f = Square()
y = f(x)

print(type(y))
print(y.data)