## Deep-Learning-from-Scratch-3

<details>
<summary>step01</summary>

---
## 1.1 변수란 
- 상자와 데이터는 별개
- 상자에 데이터가 들어감(할당)
- 상자 속을 보면 데이터를 알 수 있음(참조)

## 1.2 Variable 클래스 구현
- Variable 클래스 선언 
- 클래스로 인스턴스를 만듬. 인스턴스는 데이터를 담은 상자가 됨

## 1.3 [보충] 넘파이의 다차원 배열
- 0차원 배열(0차원 텐서) : 스칼라(scalar)
- 1차원 배열(1차원 텐서) : 벡터(vector) -> 축이 1개
- 2차원 배열(2차원 텐서)  : 행렬(matrix) -> 축이 2개
- 다차원 배열 : 텐서(tensor)
- 3차원 벡터와 3차원 배열은 다른 의
---

</details>

<details>
<summary>step02</summary>

---
## 2.1 힘수란 
- 어떤 변수로부터 다른 변수로의 대응 관계를 정한 것 (x -> *f* -> y)

## 2.2 Function 클래스 구현
- Variable 인스턴스를 다룰 수 있는 함수를 Function 클래스로 구현  
    - Function 클래스는 Variable 인스턴스를 입력받아 Variable 인스턴스를 출력함
    - Variable 인스턴스의 실제 데이터는 인스턴스 변수인 data에 있음

## 2.3 Function 클래스의 이용 
- DeZero 함수의 충족 사항   
    - Function 클래스는 기반 클래스로서, 모든 함수에 공통되는 기능을 구현한다.
    - 구체적인 함수는 Function 클래스를 상속한 클래스에서 구현한다.
- Note : Function 클래스의 forward 메서드는 예외를 발생시킨다. 이렇게 해두면 Function 클래스의 forward 메서드를 직접 호출한 사람에게 '이 메서드는  상속하여 구현해야 한다.' 는 사실을 알려줄 수 있다
---

</details>

<details>
<summary>step03</summary>

---
## 3.1 Exp 함수 구현 
- $y=e^x$ 구현. $e$는 자연로그의 밑, 2.718..., 오일러의 수, 네이피어 상수

## 3.2 함수 연결
- Function 클래스의 __call__ 메서드의 입,출력이 모두 Variable 인스턴스이기 때문에 함수 연결이 가능함
- $y=(e^{x^2})^2$ 계산도 가능
- 어려 함수로 구성된 함수를 **합성 합수** 라고 함
- 일련의 계산으로 계산 그래프로 그림 이유는? 
    - 계산 그래프를 이용하면 각 변수에 대한 미분을 효율적으로 계산할 수 있기 떄문임
    - 변수별 미분을 계산하는 알고리즘이 바로 **역전파**
---

</details>

<details>
<summary>step04</summary>

---
## 4.1 미분이란 
- 미분은 변화율 
    - 물체의 시간에 따른 위치 변화율(위치의 미분)은 속도
    - 시간에 대한 속도 변화율(속도의 미분)은 가속도 
- 정의는 '극한으로 가는 짧은 시간(순간)'에서의 변화량 
$$f'(x)=\displaystyle\lim_{h{\rightarrow}0}{\frac{f(x+h)-f(x)}{h}}$$
- $y=f(x)$가 어떤 구간에서 미분이 가능하다면 $f'(x)$ 도 함수이며, $f(x)$의 도함수라고 함

## 4.2 수치 미분 구현
- 컴퓨터는 극한을 취급할 수 없으니 극한과 비슷한 값으로 대체 
- $h=0.0001(=1e-4)$ 와 같은 매우 작은 값을 이용하여 함수의 변화량을 구하는 방법을 **수치미분**(numerical differentiation)이라고 함
- 수치 미분은 작은 값을 사용하여 '진정한 미분'을 근사함, 따라서 어쩔 수 없이 오차가 포함됨
- 이 오차를 줄이는 방법으로 '중앙차분', $f(x)$와 $f(x+h)$의 차이를 구하는 대신 $f(x-h)$와 $f(x+h)$의 차이를 구함

## 4.3 합성 함수의 미분
- 합성함수 $y=(e^{x^2})^2$ 의 미분
- 미분한 값이 3.297... 이라면, x를 0.5에서 작은 값만큼 변화시키면 y는 작은 값의 3.297...배만큼 변한다는 의미
- 복잡한 함수라도 원하는 함수를 선언한 후 미분 가능, 그러나 수치 미분의 문제가 있음

## 4.4 수치 미분의 문제점
- 수치 미분의 결과에는 오차가 포함되어있음 대부분의 경우 매우 작지만 어떤 계산이냐에 따라 커질 수도 있음
    - 오차가 포함되기 쉬운 이유는 주로 '자릿수 누락'
    - 차이를 구하는 계산은 주로 크기가 비슷한 값들을 다루므로 자릿수 누락이 생겨 유효 자릿수가 줄어들 수 있음
- 더 심각한 문제는 계산량이 많다는 점, 변수가 여러 개인 계산을 미분할 경우 변수 각각을 미분해야하기 떄문
- 신경망에서는 매개변수를 수백만 개 이상 사용하는 것은 일도 아님, 그래서 등장한 것이 **역전파**
- 수치 미분은 구현이 쉽고 거의 정확한 값을 얻을 수 있음, 이에 비해 역전파는 복잡한 알고리즘이라서 버그가 섞여 들어가기 쉬움
- 정확히 구현했는지 확인을 위해 수치미분 결과를 이용, 이를 **기울기 확인**(gradient checking)
---

</details>

<details>
<summary>step05</summary>

---
## 5.1 연쇄 법칙
- 역전파(backpropagation, 오차역전파법)를 이해하는 열쇠는 **연쇄 법칙**(chain rule)
- 연쇄 법칙에 따르면 합성 함수(여러함수가 연결된 함수)의 미분은 구성 함수 각각을 미분한 후 곱한 것과 같다고 함
- $a = A(x)$, $b = B(a)$, $y = C(b)$ $\Rightarrow$ $y = C(B(A(x)))$
$$\frac{dy}{dx} = \frac{dy}{db}\frac{db}{da}\frac{da}{dx}$$
$$\frac{dy}{dx} = \frac{dy}{dy}\frac{dy}{db}\frac{db}{da}\frac{da}{dx}$$
- $\frac{dy}{dy}$는 1

## 5.2 역전파 원리 도출
- 출력에서 입력 방향으로(즉, 역방향으로) 순서대로 계산
$$\frac{dy}{dx} = ((\frac{dy}{dy}\frac{dy}{db})\frac{db}{da})\frac{da}{dx}$$
- 미분값이 오른쪽에서 왼쪽으로 전파되는 것을 알 수 있음, 역전파 

## 5.3 계산 그래프로 살펴보기
- 순전파
$$x \rightarrow A \rightarrow a \rightarrow B \rightarrow b \rightarrow C \rightarrow y$$
- 역전파
$$\frac{dy}{dx} \leftarrow A'(x) \leftarrow \frac{dy}{da} \leftarrow B'(a) \leftarrow \frac{dy}{db} \leftarrow C'(b) \leftarrow \frac{dy}{dy}$$
- 위의 식을 잘 보면 역전파 시에는 순전파 시 이용한 데이터가 필요하다는 것을 알 수 있음 
- 따라서, 역전파를 구하려면 먼저 순전파를 하고 각 합수의 입력변수(x, a, b)의 값을 꼭 기억해둬야함 
---

</details>

<details>
<summary>step06</summary>

---
## 6.1 Variable 클래스 추가 구현
- 미분값도 저장하도록 확장

## 6.2 Function 클래스 추가 구현
- 미분을 계산하는 역전파(backward 메서드) 추가
- forward 메서드 호출 시 건네받은 Variable 인스턴스 유지 추가

## 6.3 Square 과 Exp 클래스 추가 구현
- $y=x^2$, $y=e^x$ 의 각 미분값을 곱해주는 함수 추가

## 6.4 역전파 구현
- 역전파는 $\frac{dy}{dy}=1$ 에서 시작, 출력 y의 미분값을 np.array(1.0)
---

</details>

<details>
<summary>step07</summary>

---
## 7.1 역전파 자동화의 시작
- 역전파 자동화는 변수와 함수의 '관계'를 이해하는 데서 출발
- 함수의 입장에서 변수는 입력 변수(input)와 출력 변수(output)
- 변수의 입장에서 함수는 창조자(creator), 함수에 의해 변수가 만들어짐
- 동적 계산 그래프는 실제 계산이 이루어질 때 변수에 관련 '연결'을 기록하는 방식으로 만들어짐
- Define-by-Run : 데이터를 흘려보냄으로써(Run 함으로써) 연결이 규정된다(Define 된다)는 뜻
- $x \longrightarrow A \longrightarrow a \longrightarrow B \longrightarrow b \longrightarrow C \longrightarrow y$
- $x\xleftarrow{\text{input}}A\xleftarrow{\text{creator}}a\xleftarrow{\text{input}}B\xleftarrow{\text{creator}}b\xleftarrow{\text{input}}C\xleftarrow{\text{creator}}y$

## 7.2 역전파 도전!
```python
y.grad = np.array(1.0)

C = y.creator # 1. 함수를 가져온다.
b = C.input # 2. 함수의 입력을 가져온다.
b.grad = C.backward(y.grad) # 3. 함수의 backword 메서드를 호출한다.
```
```python
B = b.creator # 1. 함수를 가져온다.
a = B.input # 2. 함수의 입력을 가져온다.
a.grad = B.backward(b.grad) # 3. 함수의 backword 메서드를 호출한다.
```
```python
A = a.creator # 1. 함수를 가져온다.
x = A.input # 2. 함수의 입력을 가져온다.
x.grad = A.backward(a.grad) # 3. 함수의 backword 메서드를 호출한다.
```

## 7.3 backward 메서드 추가
(step07.py)

---

</details>

<details>
<summary>step08</summary>

---

## 8.1 현재의 Variable 클래스
- 현재 backward 함수는 이전의 backward를 호출하는 함수
- self.creator 가 None인 변수를 찾을 때까지 backward에서 backward를 호출하고 또 호출하고... 계속 반복 $\rightarrow$ 재귀 구조

## 8.2 반복문을 이용한 구현
(step08.py)

## 8.3 동작 확인
- 재귀보다 반복문이 더 효율적인 이유? 
    - 재귀는 함수를 재귀적으로 호출할 떄 마다 중간 결과를 메모리에 유지하면서(스택을 쌓으면서) 처리를 이어감. 그러나 요즘 컴퓨터의 메모리가 넉넉한 편이여서 큰 문제가 되지 않음.

---

</details>

<details>

<summary>step09</summary>

---
## 9.1 파이썬 함수로 이용하기 
- 기존 방식의  불편한 점 : Square 클래스의 인스턴스 생성 후 호출 
    - 해결방법 : 파이썬 함수 (step09.py)
```
x = Variable(np.array(0.5))
f = Square()
y = f(x)
```

## 9.2 backward 메서드 간소화
- 역전파할 때마다 작성했던 `y.grad = np.array(1.0)` 생략
    - 해결방법 : `np.ones_like()` ( + data, grad 데이터 타입을 같게 하기 위함)

## 9.3 ndarray 만 취급하기
- float, int 타입을 사용하지 않도록 하기 위한 처리
- numpy 의 독특한 관례?!
```
# 예상한 결과
x = np.array([1.0])
y = x ** 2
print(type(x), x.ndim) -> <class 'numpy.ndarray'> 1
print(type(y))         -> <class 'numpy.ndarray'>
```

```
# 문제의 결과 : 출력이 float이 나와버림 (오호...)
x = np.array(1.0)
y = x ** 2
print(type(x), x.ndim) -> <class 'numpy.ndarray'> 0
print(type(y))         -> <class 'numpy.float64'>
```
- 스칼라 값을 걸러내기 위해서 $\rightarrow$ `np.isscalar()`

---

</details>

<details >

<summary>step10</summary>

---
## 10.1 파이썬 단위 테스트
- 파이썬 테스트 때는 표준 라이브러리에 unittest 를 사용하면 편함
- unittest.TestCase를 상속한 SquareTest 클래스 구현
- 기억할 규칙
    - 테스트할 때는 test로 시작하는 메서드를 만들고 그 안에 테스트할 내용을 적음
    - 출력과 기대값이 같은지 확인

```
$ python -m unittest steps/step10.py
``` 
- `-m unittest` 인수를 제공하면 테스트 모드로 실행 가능
- OR python 파일에 `unittest.main()` 추가

## 10.2 square 함수의 역전파 테스트
- (step10.py)

## 10.3 기울기 확인을 이용한 자동 테스트
- 미분의 기댓값을 손으로 입력했으나 이부분은 자동화할 방법이 있다 : 기울기 확인
    - 기울기 확인 : 수치 미분으로 구한 결과와 역전파로 구한 결과를 비교 
- `np.allclose(a, b, rtol=1e-05, atol=1e-08)` : a, b 값이 가까운지 판정, `|a - b| <= (atol + rtol * |b|)` 조건을 만족하면 True

## 10.4 테스트 정리 
- 테스트 파일들은 하나의 장소에 모아 관리하는 것이 일반적
```
$ python -m unittest discover tests
```
- discover 하위 명령을 사용하면 지정한 디렉터리에서 테스트 파일이 있는지 검색하고 발견한 모든 파일을 실행함
- 기본적으로 지정한 디렉터리에서 이름이 test*.py 형태인 파일을 테스트 파일로 인식함(변경할 수 있음)
- DeZero의 깃허브는 트래비스 CI 라는 지속적 통합(CI) 서비스를 연계해둠
    - push -> pull request 병합 -> 매시간 자동으로 테스트 실행 되도록 설정
    - 배지까지 표기하면 소스 코드의 신뢰성을 줄 수 있음

---

</details>

<details>

<summary>step11</summary>

---
## 11.1 Function 클래스 수정 
- 이전 Function 클래스는 '하나의 인수'만 받고 '하나의 값'만 반환
- 지금 인수와 반환값의 타입을 리스트로 바꿈

## 11.2 Add 클래스 구현
- (step11.py)

---

</details>

<details>
<summary>step12</summary>

---
## 12.1 첫 번쨰 개선 : 함수를 사용하기 쉽게
- '정의'할 떄 인수에 별표를 붙이면 호출할 떄 넘긴 인수들을 별표에 붙인 인수 하나로 모아서 받을 수 있음

## 12.2 두 번째 개선 : 함수를 구현하기 쉽도록
- `paking` : 여러개의 객체를 하나의 객체로 합쳐줌 (매개변수에서 *)
- `unpacking` : 여러개의 객체를 포함하고 있는 하나의 객체를 풀어줌 (인자 앞에 *)

## 12.3 add 함수 구현
- (step12.py)
---

</details>

<details>
<summary>step13</summary>

---
## 13.1 가변 길이 인수에 대응한 Add 클래스의 역전파
- 덧셈의 역전파는 출력쪽에서 전해지는 미분값에 1을 곱한값, 상류에서 흘러오는 미분값을 그대로 흘려보내는 것 

## 13.2 Variable 클래스 수정
- (step13.py)

## 13.3 Square 클래스 구현
- (step13.py)
- 단수형인 input에서 복수형인 inputs로 받을 수 있게 수정

---

</details>

<details>
<summary>step14</summary>

---
## 14.1 문제의 원인
```
x = Variable(np.array(3.0))
y = add(x, x)
print('y', y.data)

y.backward()
print('x.grad', x.grad)

>>> y 6.0
>>> x.grad 1
```
- 왜 미분값이 틀렸을까...? 
    - -> 출력쪽에서 전해지는 미분값을 그래도 대입하여 같은 변수를 반복해서 사용할 경우 전파되는 미분값이 덮어쓰여지는 것!!

## 14.2 해결책
- (step14.py)

## 14.3 미분값 재설정
- (step14.py)

---

</details>

<details>
<summary>step15 : 복잡한 계산 그래프(이론 편)</summary>

---
## 15.1 역전파의 올바른 순서
- 같은 변수를 반복해서 사용하거나 여러 변수를 입력받는 함수를 사용하는 계산도 할 수 있어야함
- 현재 복잡한 연결의 역전파는 불가능
- 이를 구현하기 위해 비교적 간단한 그래프에서 시작하려고 함

## 15.2 현재의 DeZero
- 아래의 코드를 보면 처리할 함수는 리스트의 끝에 추가하고 끝에서 꺼낸다
```
class Variable:
...
    def backward(self):
        ...
        while funcs:
            f = funcs.pop()
            ...
            for x, gx in zip(f.input, gxs):
                ...
                funcs.append(x.creator)

```
- 그동안 다뤘던 함수는 항상 하나의 함수를 꺼냈기 때문에 순서를 고려하지 않음

## 15.3 함수 우선순위
- funcs 리스트에 있는 함수 중 출력 쪽에 더 가까운 함수를 꺼낼 수 있어야함
- 변수-함수 를 하나의 세대로 묶어서 입력쪽에 있을 수록 0세대, 출력쪽에 있을 수록 4세대로 먼저 계산해야할 함수를 알 수 있다

---

</details>

<details>
<summary>step16 : 복잡한 계산 그래프(구현 편)</summary>

---
## 16.1 세대 추가
- Variable, Function 클래스에 변수 generation 추가 (step16.py)

## 16.2 세대 순으로 꺼내기 
- sort 메서드를 통해서 generation을 오름차순으로 정렬

## 16.3 Variable 클래스의 backward
- 중첩 합수 : 메서드 안에 메서드 
    - 감싸는 메서드 안에서만 사용한다
    - 감싸는 메서드에 정의된 변수를 사용해야 한다

## 16.4 동작 확인
- (step16.py)

---

</details>


<details>

<summary>step17 : 메모리 관리와 순환 참조</summary>

---
## 17.1 메모리 관리
- 파이썬은 필요 없어진 객체를 메모리에서 자동으로 삭제함
- 코드를 제대로 작성할지 않으면 메모리 누수 또는 메모리 부족 등의 문제가 발생
- 특히 신경망에서는 큰 데이터를 다루는 경우가 많아서 메모리 관리를 제대로 하지 않으면 실행 시간이 오래걸리는 (GPU의 경우 실행할 수조차 없는) 일이 자주 발생함
- 파이썬의 메모리 관리 방식
    1. 참조(reference)수를 세는 방식 : 참조 카운트
    2. 세대(generation)를 기준으로 쓸모없어진 객체(garbage)를 회수(collection)하는 방식 : GC(garbage collection)

## 17.2 참조 카운트 방식의 메모리 관리
- 모든 객체는 카운트가 0인 상태로 생성되고 다른 객체가 참조할 떄 마다 1씩 증가 
- 반대로 객체에 참조가 끊길 때 1씩 감소하다가 0이 되면 파이썬 인터프리터가 회수
- 참조 카운트가 증가하는 경우
    - 대입 연산자를 사용할 떄
    - 함수에 인수로 전달할 때
    - 컨테이너 타입 객체(리스트, 튜플, 클래스 등)에 추가할 때
    ```
    a = obj() # 변수에 대입 : 참조 카운트 1
    f(a) # 함수에 전달 : 함수 안에서는 참조 카운트 2
    # 함수 완료 : 빠져나오면 참조 카운트 1
    a = None # 대입 해제 : 참조 카운트 0
    ```
- 참조 카운트로 해결할 수 없는 문제 -> 순환참조

## 17.3 순환 참조
- 순환 참조 (circular reference) : 참조 카운트가 0이 되지 않고 삭제가 되지 않음
```
a = obj()
b = obj()
c = obj()

a.b = b
b.c = c
c.a = a

a = b = c = None
```
- GC 는 참조 카운트보다 더 영리한 방법으로 불필요한 객체를 찾아냄
- GC 는 메모리가 부족해지는 시점에 파이썬 인터프리터에 의해 자동으로 호출, 물론 명시적 호출도 가능 (gc 모듈 임포트해서 gc.collect() 실행)
- 순환 참조를 만들지 않는 것이 좋음 

## 17.4 weakref 모듈
- weakref.ref 함수를 사용하여 약한 참조를 만들 수 있음
- 약한 참조란 다른 객체를 참조하되 참조 카운트는 증가시키지 않음 

## 17.5 동작 확인
- (step17.py)

---

</details>

<details>

<summary>step18 : 메모리 절약 모드</summary>

---
## 18.1 필요 없는 미분값 삭제
- 모든 변수의 미분값이 생기는 것을 정하는 옵션을 추가 (step18.py)

## 18.2 Function 클래스 복습
- 미분을 하려면 순전파를 수행한 뒤 역전파를 해주면 된다
- 그리고 역전파 시에는 순전파의 결과가 필요하기 떄문에 기억해둔다
- 역전파를 구하지 않는 경우라면 순전파의 모든 결과를 기억할 필요는 없다
- 학습의 경우 미분값을 구해야하지만 추론 시에는 단순히 순전파만 하기 때문에 중간 결과가 필요없다

## 18.3 Config 클래스를 활용한 모드 전환
- 역전파 활성 모드 / 비활성 모드 전환 구조를 위해 `Config` 클래스 선언
- [CAUTION] 설정 데이터는 단 한군데만 존재하는 것이 좋음. 그래서 Config 클래스는 인스턴스화 하지않고 '클레스' 상태로 이용. 인스턴스는 여러 개 생성할 수 있지만 클래스는 항상 하나만 존재하기 때문임

## 18.4 모드 전환
- (step18.py)

## 18.5 with 문을 활용한 모드 전환
- `with` 후처리를 자동으로 수행할 떄 사용하는 구문 예. open, close
- contextlib.contextmanager 데코레이터를 활용해서 config를 설정함 (step18.py)
- getattr 클래스의 값을 꺼내옴, setattr 새로움 값으로 설정가능

---

</details>

<details>

<summary>step19 : 변수 사용성 개선</summary>

---
## 19.1 변수 이름 지정
- 많은 변수를 처리할 것이기에 이름을 붙이기로 함 
- 이후 시각화때도 사용할 수 있음

## 19.2 ndarray 인스턴스 변수
- Variable은 데이터를 담는 '상자' -> 중요한 것은 상자가 아니라 그 안의 '데이터'
- ndarray 의 인스턴스 변수를 사용할 수 있도로고 확장 -> `@property`

## 19.3 len 함수와 print 함수
- (step19.py)

---

</details>

<details>

<summary>step20 : 연산자 오버로드(1)</summary>

---
## 20.1 Mul 클래스 구현
- (step20.py)

## 20.2 연산자 오버로드
- 곱셈 연산자 *를 오버로드 -> `__mul__(self, other)`
- (step20.py)

---

</details>

<details>

<summary>step21 : 연산자 오버로드(2)</summary>

---
## 21.1 ndarray와 함께 사용하기
- `a * np.array(2.0)` 가능하게 하기 위해 as_variable 함수 준비 

## 21.2 float, int와 함께 사용하기
- (step21.py)

## 21.3 문제점 1 : 첫 번째 인수가 float나 int인 경우
- `2.0 * x` 가능하려면 -> `__rmul__(self, other)` 필요함
- (step21.py)

## 21.4 문제점 2 : 좌항이 ndarray 인스턴스인 경우 
- `ndarray + Variable` -> ndarray의 `__add__` 메서드가 호출됨
- 연산자 우선순위를 정해야함 -> `__array_priority__`
    - 우선순위를 정하지 않을 경우 -> `[varivale([3.])]`
    - 우선순위를 정했을 경우 -> `varivale([3.])`

---

</details>

<details>

<summary>step22 : 연산자 오버로드(3)</summary>

---
## 22.1 음수(부호 변환)
- 새로운 연산자를 추가하는 순서
    1. Function 클래스를 상속하여 원하는 함수 클래스 구현(예. Mul 클래스)
    2. 파이썬 함수로 사용할 수 있도록 함(예. mul 함수)
    3. Variable 클래스의 연산자를 오버로드 함(예. Variable.__mul__=mul)
- 위 순서를 반복함 (step22.py)

## 22.2 뺄셈
- 덧셈과 곱셈은 좌항과 우항의 순서를 바꿔도 결과가 같기 때문에 둘을 구별할 필요가 없었으나 뺄셈에서는 구별해야함(`x0 - x1`과 `x1 - x0`의 값은 다름) 따라서 우항을 대상으로 했을 떄 적용할 함수인 rsub(x0, x1)을 별도로 준비해야함

## 22.3 나눗셈
- (step22.py)

## 22.4 거듭제곱
- (step22.py)

---

</details>

<details>

<summary>step23 : 패키지로 정리</summary>

---
## 23.1 파일 구성
- dezero 라는 공통의 디렉터리를 만듬 -> 패키지
```
# 최종 파일 구성
.
├── dezero
│   ├── __init__.py
│   ├── core_simple.py
│   ├── ...
│   └── utils.py
│
├── steps
│   ├── step01.py
│   ├── step02.py
│   ├── ...
│   └── step60.py
│
```

## 23.2 코어 클래스 옮기기
- steps/step22.py -> dezero/core_simple.py

## 23.3 연산자 오버로드
- (dezero/core_simple.py, dezero/__init__.py)

## 23.4 실제 __init__.py 파일
- (dezero/__init__.py)

## 23.5 dezero 임포트하기
- (step23.py)
- `if '__file__' in globals():` -> __file__ 이라는 전역변수가 정의되어 있는지 확인
    - (참고) `__file__` 변수는 파이썬 인터프리터의 인터랙티브 모드와 구글 콜랩에서 실행하는 경우 정의되어있지 않음

---

</details>

<details>

<summary>step24 : 복잡한 함수의 미분</summary>

---
## 24.1 Sphere 함수
- $z=x^2 + y^2$

## 24.2 matyas 함수
- $z= 0.26(x^2 + y^2) - 0.48xy$

## 24.3 Goldstein-Price 함수
$$
\begin{aligned}
f(x, y)=&\ [1+(x+y+1)^2(19-14x+3x^2-14y+6xy+3y^2)]\\
&\ [30+(2x-3y)^2(18-32x+12x^2+48y-36xy+27y^2)]
\end{aligned}
$$ 

## [칼럼] Define-by-Run
- 딥러닝 프레임워크는 동작 방식에 따라 크게 두 가지로 나뉨
    1. 정적 계산 그래프, Define-and-Run
    2. 동적 계산 그래프, Define-by-Run
### 1. Define-and-Run(정적 계산 그래프 방식)
- 직역하면 '계산 그래프를 정의한 다음 데이터를 흘려보낸다'
- 계산 그래프 정의는 사용가자 제공하고, 프레임워크는 주어진 그래프를 컴퓨터가 처리할 수 있는 형태로 변환(컴파일)하여 데이터를 흘려보내는 식
```
# 가상의 Define-and-Run 방식 프레임워크용 코드 예

# 계산 그래프 정의
a = Variable('a')
b = Variable('b')
c = a * b
d = c + Constant(1)

# 계산 그래프 컴파일
f = compile(d)

# 데이터 흘려보내기
d = f(a=np.array(2), b=np.array(3))
```
- 위 정의한 계싼 그래프 4줄의 코드는 실제 계산이 이루어지지 않음, 실제 '수치'가 아닌 기호를 대상으로 프로그래밍 됐음 -> 기호프로그래밍
- 추상적인 계산 절차를 코딩해야함
- 도메인 특화 언어(Domain-Specific-Language, DSL) 사용
    - 프레임워크 자체의 규칙들로 이루어진 언어 
    - 예. 상수는 Constant에 담아라 라는 규칙, 조건에 따라 분기하고 싶다면 if문 \
    (텐서플로에서는 if문의 역할로 tf.cond 연산을 사용)
    ```python
    import tensorflow as tf

    flg = tf.placeholder(dtype=tf.bool)
    x0 = tf.placeholder(dtype=tf.float32)
    x1 = tf.placeholder(dtype=tf.float32)
    y = tf.cond(flg, lambda: x0+x1, lambda: x0*x1) # if문 역할
    ```
    - 딥러닝 여명기에는 Define-and-Run 방식 프레임워크가 대부분
    - 대표적으로 텐서플로, 카페, CNTK (텐서플로 2.0부터는 Define-by-Run 방식도 도입)
    - 다음 세대로 등장한 것이 DeZero에도 채용할 Define-by-Run

### 2. Define-by-Run(동적 계산 그래프 방식)
- '데이터를 흘려보냄으로써 계산 그래프가 정의된다' 
- '데이터 흘려보내기'와 '게산 그래프 구축' 동시에 이루어지는 것이 특징
- DeZero의 경우 사용자가 데이터를 흘려보낼 때 자동으로 계산 그래프를 구성하는 '연결(참조)'를 만듬 \
이 연결이 DeZero의 계산 그래프에 해당함 \
구현 수준에서는 연결리스트로 표현되는데 이를 사용하면 계산이 끝난 후 역방향으로 추적할 수 있음
- Define-by-Run 방식 프레임워크는 넘파이를 사용하는 일반적인 프로그래밍과 같은 형태로 코딩이 가능함
```python
import numpy as np
from dezero import Variable

a = Variable(np.ones(10))
b = Variable(np.ones(10)*2)
c = b * a
d = c + 1
print(d)
```
- 2015년 체이너에 의해 처음 제창되고 이후 많은 프레임워크에 채용되고 있음
- 대표적으로 파이토치, MXNet, DyNet, 텐서플로(2.0 이상에서는 기본값)

### 정적/동적 계산 그래프 방식 장점
||Define-and-Run(정적 계산 그래프)|Define-by-Run(동적 계산 그래프)|
|:--:|:--|:--|
|장점|- 성능이 좋다 <br> - 신경망 구조를 최적화하기 쉽다 <br> - 분산 학습 시 더 편리하다|- 파이썬으로 계산 그래프를 제어할 수 있다 <br> - 디버깅이 쉽다 <br> - 동적인 계산 처리에 알맞다 |
|단점|- 독자적인 언어(규칙)을 익혀야한다 <br> - 동적 계산 그래프를 만들기 어렵다 <br> - 디버깅하기 매우 어려울 수 있다| - 성능이 낮을 수 있다|
- 성능이 중요할 떄는 Define-and-Run
- 사용성이 중요할 떄는 Define-by-Run

---

</details>

<details>

<summary>step 25 : 계산 그래프 시각화(1)</summary>

---
## 25.1 Graphviz 설치하기
- Graphviz 그래프를 시각화 해주는 도구(계산 그래프와 같이 노드와 화살표로 이뤄진 데이터 구조)
    ```
    $ brew install graphviz
    ```
- dot 명령
    ```
    $ dot sample.dot -T png -o sample.png
    ```
    - -o 파일 이름 
    - -T 출력 파일의 형식을 지정 

## 25.2 DOT 언어로 그래프 작성하기
- (dot/sample_1.dot, dot/sample_1.png)

## 25.3 노드에 속성 지정하기 
- 노드의 '색'과 '모양'을 지정할 수 있음
- 각 줄의 '1', '2' 는 노드의 ID를 나타냄, 0 이상의 정수이며 중복되면 안됨
- 해당 ID에 부여할 속성을 대괄호 []안에 적음
- (dot/sample_2.dot, dot/sample_2.png)
- (dot/sample_3.dot, dot/sample_3.png)

## 25.4 노드 연결하기
- ID를 '->' 로 연결하면 됨
- (dot/sample_4.dot, dot/sample_4.png)
---

</details>

<details>

<summary>step 26 : 계산 그래프 시각화(2)</summary>

---
## 26.1 시각화 코드 예
- 계산 그래프를 시각화하는 함수 `get_dot_graph` 라는 이름으로 `dezero/utils.py` 에 구현
- (`get_dot_graph` 구현 후 예시를 코드로 보여줌)

## 26.2 계산 그래프에서 DOT 언어로 변환하기
- (dezero/utils.py)

## 26.3 이미지 변환까지 한번에
- (dezero/utils.py)
---

</details>

<details>

<summary>step 27 : 테일러 급수 미분</summary>

---
## 27.1 sin 함수 구현
- $y=\sin(x)$ 일 때 그 미분은 $\frac{\partial{y}}{\partial{x}}=\cos(x)$
- (steps/step27.py)

## 27.2 테일러 급수 이론
- 테일러 급수란 어떤 함수를 다항식으로 근사하는 방법
- 점 a 에서 f(x)의 테일러 급수. a는 임의의 값, $f'$는 1차 미분, $f''$는 2차 미분, $f'''$는 3차 미분을 뜻함. !기호는 계승(factorial)을 뜻하며 5! = 5 x 4 x 3 x 2 x 1 =120 
    $$f(x)=f(a)+f'(a)(x-a)+\frac{1}{2!}f''(a)(x-a)^2+\frac{1}{3!}f'''(a)(x-a)^3+\cdots$$
- a = 0일 떄의 테일러 급수를 매클로린 전개(Maclaurin's series)라고도 함
    $$f(x)=f(0)+f'(0)x+\frac{1}{2!}f''(0)x^2+\frac{1}{3!}f'''(0)x^3+\cdots$$
- $f(x)=\sin(x)$를 적용해보자
    $$\sin(x)=\frac{x}{1!}-\frac{x^3}{3!}+\frac{x^5}{5!}-\cdots = \displaystyle\sum_{i=0}^{\infty}{\frac{x^{2i+1}}{(2i+1)!}}$$

## 27.3 테일러 급수 구현
- (steps/step27.py)

## 27.4 계산 그래프 시각화
- (steps/step27.py)

---

</details>

<details>

<summary>step 28 : 함수 최적화</summary>

---
## 28.1 로젠브록 함수
- 최적화란 어떤 함수가 주어졌을 때 그 최솟값(또는 최댓값)을 반환하는 '입력(함수의 인수)'을 찾는 일
- 로젠브록 함수(Rosenbrock function)
    $$y=100(x_1-x_0^2)^2+(1-x_0)^2$$
- 답부터 이야기하면 최솟값이 되는 지점은 $(x_0, x_1)=(1, 1)$ 

## 28.2 미분 계산하기
- $(x_0, x_1)=(0.0, 2.0)$ 에서의 미분을 구해보자

## 28.3 경사하강법 구현
- (steps/step28.py)

---

</details>

<details>

<summary>step 29 : 뉴턴 방법으로 푸는 최적화(수동 계산)</summary>

---
## 29.1 뉴턴 방법을 활용한 최적화 이론

$$f(x)=f(a)+f'(a)(x-a)+\frac{1}{2!}f''(a)(x-a)^2+\frac{1}{3!}f'''(a)(x-a)^3+\cdots$$
- 위의 테일러 급수는 1차 미분, 2차 미분, ... 형태로 증가하는데 이를 어느 시점에 중단하면 아래와 같이 $f(x)$를 근사적으로 나타낼 수 있음
    $$f(x)\simeq f(a)+f'(a)(x-a)+\frac{1}{2!}f''(a)(x-a)^2$$
- 근사한 2차 함수는 $y = f(x)$ 에 접하는 곡선으로 2차 함수의 최솟값은 미분 결과가 0인 위치를 확인 하면 됨
    $$
    \begin{aligned}
    \frac{d}{dx}(f(a)+f'(a)(x-a)+\frac{1}{2!}f''(a)(x-a)^2) &= 0 \\
    f'(a) + f''(a)(x-a) &= 0 \\
    x &= a - \frac{f'(a)}{f''(a)}
    \end{aligned}
    $$
- 경사하강법 $x \leftarrow x - \alpha f'(x)$ : 스칼라 값인 $\alpha$를 통해 갱신
- 뉴턴 방법 $x \leftarrow - \frac{f'(x)}{f''(x)}$ : 2차 미분을 활용하여 갱신

## 29.2 뉴턴 방법을 활용한 최적화 구현
- 2차 미분은 아직 자동이 안되므로 수동으로 구하면
    $
    \begin{align}
    y &= x^4 - 2x^2 \\
    \frac{\partial{y}}{\partial{x}} &= 4x^3 - 4x \\
    \frac{\partial^2{y}}{\partial{x^2}} &= 12x^2 - 4
    \end{align}
    $

---

</details>

<details>

<summary>step 30 : 고차 미분(준비편)</summary>

---
## 30.1 확인 1 : Variable 인스턴스 변수
- Variable 클래스의 인스턴스 변수에 대한 복습
```python
class Variable:
    def __init__(self, data, name=None):
        if data is not None:
             if not isinstance(data, np.ndarray):
                  raise TypeError('{}은(는) 지원하지 않습니다.'.format(type(data)))

        self.data = data
        self.name = name
        self.grad = None
        self.creator = None
        self.generation = 0 
```
- 주의할 것은 `data`, `grad` 모두 `ndarray` 인스턴스로 저장한다는 사실
- x = Variabel(np.array(2.0)) -> x.data = object
- x.backward(), x.grad = np.array(1.0) -> x.grad = object

## 30.2 확인 2 : Function 클래스
- Function 클래스의 __call__ 메서드 복습
```python
class Function:
    def __call__(self, *inputs):
        inputs = [as_variable(x) for x in inputs]
        # 순전파 메인 처리
        xs = [x.data for x in inputs]
        ys = self.forward(*xs) 
        if not isinstance(ys, tuple): 
          ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]
        
        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs]) 
            # '연결'을 만듦
            for output in outputs:
                output.set_creator(self)
            self.inputs = inputs 
            self.outputs = [weakref.ref(output) for output in outputs]
        return outputs if len(outputs) > 1 else outputs[0]
```
- 순전파의 메인 처리에서 inputs의 data를 꺼내 리스트에 모으고 forward를 계산
- 연결에서 Variabel과 Function의 관계가 만들어짐, 변수에서 함수로의 연결은 set_creator 메서드에서 만들어짐, 또한 함수의 inputs와 outputs 인스턴스 변수에 저장하여 연결을 유지

## 30.3 확인 3 : Variable 클래스의 역전파 
- backward 메서드 복습
```python
class Variable:
    ...

    def backward(self, retain_grad=False):
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
           f = funcs.pop() 

           # 역전파 계산(메인 처리)
           gys = [output().grad for output in f.outputs] # 1
           gxs = f.backward(*gys) # 2
           if not isinstance(gxs, tuple): 
                gxs = (gxs,) 
            
           for x, gx in zip(f.inputs, gxs): # 3
                if x.grad is None:
                      x.grad = gx 
                else:
                      x.grad = x.grad + gx 

                if x.creator is not None:
                     add_func(x.creator)
           if not retain_grad:
                for y in f.outputs:
                     y().grad = None 
```
- 1에서 인스턴스 변수인 grad를 리스트로 모음
- 2에서 backward 메서드에는 ndarray 인스턴스가 담긴 리스트가 전달
- 3에서 출력쪽에서 전파하는 미분값(gxs)을 함수의 입력변수(f.inputs)의 grad로 설정
---

</details>

<details open>

<summary>step 31 : 고차 미분(이론편)</summary>

---
## 31.1 역전파 계산 
- 계산의 '연결'은 `Function` 클래스의 `__call__` 메서드에서 만들어짐이 중요
```python
class Sin(Function):
    ...

    def backward(self, gy):
        x = self.inputs[0].data
        gx = gy * np.cos(x) # 구체적인 게산이 이루어짐
        return gx
```
- 만약 역전파를 계산할 때도 '연결' 이 만들어진다면 고차미분을 자동으로 계산할 수 있음!!
- gx.backward()를 호출함으로써 x에 대한 미분이 한 번 더 이루어짐. 즉, x의 2차 미분!

## 31.2 역전파로 계산 그래프 만들기
- DeZero는 Variable 인스턴스를 사용해서 일반적인 계산(순전파)을 하는 시점에 '연결'이 만들어짐
- 즉, 함수의 backward 메서드에서도 ndarray 인스턴스가 아닌 Variable 인스턴스를 사용하면 '연결'이 만들어짐

---

</details>

<details open>

<summary>step 32 : 고차 미분(구현편)</summary>

---
## 32.1 새로운 DeZero로!
- 

## 32.2 함수 클래스의 역전파
- 

## 32.3 역전파를 더 효율적으로(모드 추가)
-

## 32.4 __init__.py 변경
-

---

</details>