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

<details>

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

<details>

<summary>step 32 : 고차 미분(구현편)</summary>

---
## 32.1 새로운 DeZero로!
- 가장 중요한 변화는 Variable 클래스의 인스턴스 변수인 `grad`
- (`dezero/core.py`)

## 32.2 함수 클래스의 역전파
- 역전파 계산은 Variable 인스턴스에 대해 이루어짐
- (`dezero/core.py`)

## 32.3 역전파를 더 효율적으로(모드 추가)
- '역전파 비활성 모드'로 전환하여 역전파 처리를 생략
- 2차 미분이 필요할 경우 `create_graph`를 `True`로 설정

## 32.4 __init__.py 변경
- (`dezero/__init__.py`)

---

</details>

</details>

<details>

<summary>step 33 : 뉴턴 방법으로 푸는 최적화(자동 계산)</summary>

---
## 33.1 2차 미분 계산하기
- (`steps/step33.py`)

## 33.2 뉴턴 방법을 활용한 최적화
- (`steps/step33.py`)

---

</details>

<details>

<summary>step 34 : sin 함수 고차 미분</summary>

---
## 34.1 sin 함수 구현
- (`dezero/functions.py`)

## 34.2 cos 함수 구현
- (`dezero/functions.py`)

## 34.3 sin 함수 고차 미분
- (`steps/step34.py`)

---

</details>

<details>

<summary>step 35 : 고차 미분 계산 그래프</summary>

---
## 35.1 tanh 함수 미분
- `tanh` 는 쌍곡탄젠트 혹은 하이퍼볼릭 탄젠트 라고 읽음
$$y = \tanh(x)=\frac{e^x-e^{-x}}{e^x+e^{-x}}$$
- 입력을 `-1 ~ 1` 사이의 값으로 변환
- 분수 함수의 미분 공식
$$\bigg( \frac{f(x)}{g(x)} \bigg)' = \frac{f'(x)g(x)-f(x)g'(x)}{g(x)^2}$$
- 자연로그에 대해 $\frac{\partial e^x}{\partial x}=e^x$ 이고 $\frac{\partial e^{-x}}{\partial x}=-e^{-x}$ 인 점을 이용하면 아래와 같은 식을 얻음

$$\begin{align}
   \frac{\partial \tanh(x)}{\partial x}&=\frac{(e^x+e^{-x})(e^x+e^{-x})-(e^x-e^{-x})(e^x-e^{-x})}{(e^x+e^{-x})^2} \\
   &= 1-\frac{(e^x-e^{-x})(e^x-e^{-x})}{(e^x+e^{-x})^2} \\
   &= 1-\tanh(x)^2 \\
   &= 1-y^2
\end{align}$$

## 35.2 tanh 함수 구현
- (`dezero/functions.py`)

## 35.3 고차 미분 계산 그래프 시각화
- (`steps/step35.py`)

---

</details>

<details>

<summary>step 36 : 고차 미분 이외의 용도</summary>

---
## 36.1 double backprop 의 용도
- 다음의 두 식이 주어졌을 때 x = 2.0에서 z의 미분 $\frac{\partial z}{\partial x}$ 를 구하라
$$y = x^2$$
$$z = (\frac{\partial y}{\partial x})^3 + y $$

- 손으로 계산했을 때
$$\frac{\partial y}{\partial x} = 2x$$
$$z = (\frac{\partial y}{\partial x})^3 + y = 8x^3 + x^2$$
$$\frac{\partial z}{\partial x} = 24x^2 + 2x$$

- gx = x.grad 는 단순한 변수가 아니라 계산 그래프(식)  
따라서 x.grad 계산 그래프에 대해 추가로 역전파할 수 있음

- (`steps/step36.py`)

## 36.2 딥러닝 연구에서의 사용 예
- WGAN-GP 논문에서 최적화 식에 기울기가 들어있음  
이 기울기는 첫 번째 역전파에서 구할 수 있고 이 기울기를 사용해서 함수를 계산하고 이를 최적화하기 위해 두 번쨰 역전파를 함 -> double backprop

---

</details>

<details>

<summary>step 37 : 텐서를 다루다</summary>

---
## 37.1 원소별 계산
- add, mul, div, sin 등 DeZero의 함수는 입력 출력이 `스칼라` 라고 가정했음
- (`steps/step37.py`)


## 37.2 텐서 사용 시의 역전파
- 텐서를 사용한 역전파를 적용하여면 무엇을 바꿔야할까?  
사실 `텐서`를 사용해도 역전파 코드가 문제없이 작동한다. 그 이유는
    - 우리는 그동안 `스칼라`를 대상으로 역전파를 구현
    - 지금까지 구현한 DeZero 함수에 `텐서`를 건네면 텐서의 원소마다 `스칼라`로 계산
    - 텐서의 원소별 `스칼라`계산이 이루어지면 `스칼라`를 가정해 구현한 역전파는 `텐서`의 원소별 계산에서도 성립


---

</details>

<details>

<summary>step 38 : 형상 변환 함수</summary>

---
## 38.1 reshape 함수 구현
- reshape 함수는 단순히 형상만 변환함. 다시 말해, 구체적인 계산을 하지 않음
- (`steps/step38.py`)
- (`dezero/functions.py`)

## 38.2 Variable 에서 reshape 사용하기
- numpy의 reshape를 ndarray 인스턴스의 메서드로 사용할 수 있고 x.reshape(2, 3) 과  같이 가변 인수도 받음 -> DeZero에서도 같은 방법을 제공하고 싶음
- (`dezero/core.py`)

## 38.3 행렬의 전치
- transpose 구현
- (`dezero/functions.py`)
- (`dezero/core.py`)

---

</details>

<details>

<summary>step 39 : 합계 함수</summary>

---
## 39.1 sum 함수의 역전파
- 덧셈의 미분은 $y=x_0+x_1$ 일 때 $\frac{\partial y}{\partial x_0}=1$, $\frac{\partial y}{\partial x_1}=1$
- 따라서 역전파는 출력쪽에서 전해지는 기울기를 그대로 입력 쪽으로 흘려보내기만 하면 됨
- 원소가 2개 이상인 벡터의 합에 대한 역전파는 출력쪽에서 전해준 값인 1 을 입력변수의 형상과 같아지도록함

## 39.2 sum 함수 구현
- DeZero의 sum함수 역전파에서는 입력변수의 형상과 같아지도록 기울기의 원소를 복사 
- (`steps/step39.py`)

## 39.3 axis와 keepdims
- 합계를 구할 떄 '축'을 지정할 수 있음 -> `axis`
- 입력과 출력의 차원 수(축 수)를 똑같게 유지할지 정하는 플래그 -> `keepdims`
- (`dezero/core.py`)
- (`dezero/functions.py`)
- (`steps/step39.py`)

---

</details>

<details>

<summary>step 40 : 브로드캐스트 함수</summary>

---
## 40.1 broadcast_to 함수와 sum_to 함수(넘파이 버전)
- 넘파이의 `np.broadcast_to(x, shape)` 함수는 ndarray 인스턴스인 `x`의 원소를 복제하여 `shape` 인수로 지정한 형상이 되도록 해줌
- (`steps/step40.py`)
- 브로드캐스트(원소 복사)가 수행된 후 역전파는?
    - 기울기의 `합`을 구함
- broadcast_to 의 역전파는 sum_to
- `sum_to(x, shape)` 함수는 `x`의 원소 합을 구해 `shape` 형상으로 만들어주는 함수
- sum_to 의 역전파는 broadcast_to

## 40.2 broadcast_to 함수와 sum_to 함수(DeZero 버전)
- (`dezero/functions.py`)

## 40.3 브로드캐스트 대응
- (`steps/step40.py`)
- (`dezero/core.py`)

---

</details>

<details>

<summary>step 41 : 행렬의 곱</summary>

---
## 41.1 벡터의 내적과 행렬의 곱
- 벡터의 내적 : 벡터 $\textbf{a} = (a_1, \cdots , a_n)$, $\textbf{b} = (b_1, \cdots , b_n)$ 가 있다고 가정했을 때, 두 벡터의 내적은 아래와 같음. 두 벡터 사이의 대응 원소의 곱을 모두 더한 값 $$\textbf{ab} = a_1 b_1 + a_2 b_2 \cdots + a_n b_n$$
- 행렬의 곱 : 왼쪽 행렬의 '가로 방향 벡터'와 오른쪽 행렬의 '세로 방향 벡터' 사이의 내적을 계산
- (`steps/step41.py`)

## 41.2 행렬의 형상 체크
- 행렬과 벡터를 사용한 계산에서는 '형상'에 주의해야 함! 
- 3 X 2 행렬 $\textbf{a}$ 와 2 X 4 행렬 $\textbf{b}$ 를 곱하여 3 X 4 행렬 $\textbf{c}$ 가 만들어질 때,  
행렬 $\textbf{a}$ 와 행렬 $\textbf{b}$ 의 대응하는 차원(축)의 원소 수가 일치해야 함

## 41.3 행렬 곱의 역전파
- DeZero는 행렬 곱 계산을 MatMul 클래스와 matmul 함수로 구현함
- matmul 은 matrix multiply 약자 
- (어렵다... 이해가 어려워...)
- (`dezero/functions.py`)

---

</details>

<details>

<summary>step 42 : 선형 회귀</summary>

---
## 42.1 토이 데이터셋 
- 100개의 토이 데이터셋 준비 
```python
import numpy as np

np..random.seed(0) # 시드값 고정
x = np.random.rand(100, 1)
y = 5 + 2 * x + np.random.rand(100, 1) # y에 무작위 노이즈 추가
```

## 42.2 선형 회귀 이론
- x로부터 실숫값 y를 예측하는 것 : 회귀(regressions)
- 선형 관계를 가정하기에 $y=Wx+b$ 식으로 표현 ($W$는 스칼라값)
- 우리의 목표인 $y=Wx+b$을 찾으려고 할 때, 예측치의 차이인 잔차(residual)을 최소화 해야함
- 모델의 성능이 얼마나 `나쁜가`를 평가하는 함수 -> 손실함수(loss function) 
- 선형 회귀는 손실함수로 평균 제곱 오차(mean squared error)를 이용
$$L = \frac{N}{1}\sum^{N}_{i=1}(f(x_i)-y_i)^2$$
- 손실 함수의 출력이 최소화하는 $W$와 $b$를 찾는 것

## 42.3 선형 회귀 구현
- (`steps/step42.py`)

## 42.4 [보충] DeZer의 mean_squared_error 함수
- 이전 코드
```python
def mean_squared_error(x0, x1):
    diff = x0 - x1
    return F.sum(diff ** 2) / len(diff)
```
- 이름 없는 변수 3가지 -> 메모리 잡아먹는 친구들
    - x0 - x1 = ?
    - ? ** 2 = ??
    - sum(??) = ???
- 조금 더 나은 방식 도입 (`dezero/functions.py`)


---

</details>


<details>

<summary>step 43 : 신경망</summary>

---
## 43.1 DeZero의 linear 함수
- 선형회귀를 수행한 계산은 '행렬의 곱' 과 '덧셈'
```
y = F.matmul(x, W) + b
```
- 이 변환을 선형 변환(linear transformation) or 아핀 변환(affine transformation)
- 클래스를 상속하여 함수 클래스를 선언하는 방식이 메모리를 더 효율적으로 사용함 (참고. 42.4)
- 그러나 클래스를 선언하지 않고 메모리를 효율적으로 사용할 수 있는 방법도 있음!!
    - 위 선형회귀 식에서 역전파를 구할 때, $x+W$를 담은 변수는 역전파시 사용하지 않음
    - (`dezero/functions.py`)
- 신경망에서 메모리으 대부분을 차지하는 것이 중간 계산 결과인 텐서(ndarray 인스턴스)이다. 특히 큰 텐서를 취급하는 경우 ndarray 인스턴스가 거대해지므로 불필요한 ndarray 인스턴스는 즉시 삭제하는 것이 바람직하다. 

## 43.2 비션형 데이터셋
- (`steps/step43.py`)
- 비선형 함수인 sine 함수 사용

## 43.3 활성화 함수와 신경망
- 선형 변환은 입력 데이터를 선형으로 변환
- 신경망은 출력에 비선형 변환을 수행함 -> 활성화 함수(activation function)
- 대표적으로 ReLU함수와 sigmoid function 등이 있음
$$y = \frac{1}{1 + \exp(-x)}$$
- (`dezero/functions.py`)

## 43.4 신경망 구현
- 2층 신경망은 다음처럼 구현할 수 있음
```python
W1, b1 = Variable(...), Variable(...)
W2, b2 = Variable(...), Variable(...)

def predict(x):
    y = F.linear(x, W1, b1) # 또는 F.linear_simple(...)
    y = F.sigmoid(y) # 또는 F.sigmoid_simple(y)
    y = F.linear(y, W2, b2)
    return y
```
- (`steps/step43.py`)
- 추론을 제대로 하려면 `학습`이 필요하고 학습에서는 추론을 처리한 후 손실함수를 추가하고 손실함수의 출력을 최소화하는 매개변수를 찾느다. 

---

</details>

<details>

<summary>step 44 : 매개변수를 모아두는 계층</summary>

---
## 44.1 Parameter 클래스 구현
- 매개변수는 경사하강법 등의 최적화 기법에 의해 갱신되는 변수. 가중치와 편향이 매개변수에 해당함
- 매개변수 자동화를 위한 클래스 선언(Parameter, Layer)
- (`steps/step44.py`)
- Parameter 인스턴스와 Variable 인스턴스를 조합하여 계산할 수 있고 isinstance 함수로 구분할 수 있음

## 44.2 Layer 클래스 구현
- Layer는 DeZero의 Function 클래스와 마찬가지로 변수를 변환하는 클래스  
그러나 매개변수를 유지한다는 점에서 다름, 매개변수를 유지하고 매개변수를 사용해 변환함
- (`dezero/layers.py`)
    - `__setattr__` 인스턴스 변수를 설정할 떄 호출하는 특수 메서드
    - `__setattr__(self, name, value)`는 이름이 name인 인스턴스 변수에 값으로 value로 전달해줌 
    - value가 Parameter 인스턴스라면 self._params 에 name을 추가함 
- (`steps/step44.py`)
- (`dezero/layers.py`)
    - `__call__` 메서드는 입력받은 인수를 건네 forward 메서드를 호출. forward 메서드는 자식 클래스에서 구현할 것
    - `yield` 는 return 처럼 사용할 수 있음. 
        - return 은 처리를 종료하고 값을 반환
        - `yield` 는 처리를 일시중지(suspend)하고 값을 반환 -> 작업을 재개(resume)할 수 있음

## 44.3 Linear 클래스 구현
- 함수로서의 Linear 클래스가 아니라 계층으로서의 Linear 클래스 구현
- (`steps/step44.py`)
    - DeZero의 linear 함수(linera_sample)를 호출할 뿐
- (`dezero/layers.py`)
    - 더 나은 방법이 있었으니, 가중치 $W$를 생성하는 시점을 늦추는 방식
    - 구체적으로 가중치를 forward 메서드에서 생성함으로써 Linear 클래스의 입력크기를 자동으로 결정할 수 있음
    - 주목할 점은 in_size를 지정하지 않아도 된다는 것 

## 44.4 Layer를 이용한 신경망 구현
- sin 함수의 데이터셋에 대한 회귀 문제를 다시 풀어보자

---

</details>

<details>

<summary>step 45 : 계층을 모아두는 계층</summary>

---
## 45.1 Layer 클래스 확장
- Layer 클래스를 사용하면 매개변수를 직접 다루지 않아도 되서 편하지만 Layer 인스턴스 자체를 관리해야함 -> Layer 클래스 확장 
- Layer 안에 다른 Layer 가 들어가는 구조 
- (`dezero/layers.py`)
    - Layer 인스턴스 이름도 params에 추가
    - params 매서드는 _params 에서 name을 꺼내 해당하는 객체를 obj로 꺼냄. 이떄 obj 가 Layer 라면 obj.params()를 호출
    - `yield` 를 사용한 함수를 제너레이터(generator) 라고 함.  제너레이터를 사용하여 또 다른 제너레이터를 만들고자 할 때는 `yield from`을 사용 
- (`steps/step45.py`)
    - Layer 클래스 하나로 신경망의 매개변수를 한꺼번에 관리할 수 있음
    - Layer 클래스를 상속하여 모델 전체를 클래스로 정의할 수 있음

## 45.2 Model 클래스
- 모델은 '사물의 본질을 단순하게 표현한 것'이라는 뜻
- 머신러닝에서도 마찬가지. 복잡한 패턴이나 규칙이 숨어 있는 현상을 수식을 사용하여 단순하게 표현한 것. 신경망도 수식으로 표현할 수 있는 함수 -> `모델`
- (`dezero/layers.py`)
    - Layer 클래스 기능을 이어받으며 시각화 메서드 하나 추가

## 45.3 Model을 사용한 문제 해결
- (`steps/step45.py`)
    - sin 함수로 생성한 데이터셋 회귀 문제를 Model 클래스를 이용하여 다시 풀어보자 

## 45.4 MLP 클래스
- (`dezero/models.py`)
    - fc(full connect)의 약자로 `fc_output_sizes`에 신경망을 구성하는 출력크기를 튜플 또는 리스트로 전달 
    - (10, 1)를 넣으면 2개의 Linear 계층을 만듬
    - MLP (Multi-Layer Perception)의 약자로 다층 퍼셉트론이라고 함. 완전연결계층 신경망의 별칭으로 흔히 쓰임

---

</details>

<details>

<summary>step 46 : Optimizer로 수행하는 매개변수 갱신</summary>

---
## 46.1 Optimizer 클래스
- 매개변수 갱신을 위한 기반 클래스는 Optimizer 
- 구체적인 최적화 기법은 Optimizer 클래스를 상속한 곳에서 구현 
- (`dezero/optimizer.py`)
    - 초기화 메서드 : target, hooks 인스턴스 초기화
    - setup 메서드 : 매개변수를 갖는 클래스를 target 인스턴스로 설정
    - update 메서드 : 매개변수 갱신, None 은 건너뜀
    - update_one 메서드 : 구체적인 매개변수 갱신을 재정의하는 곳
    - add_hook : 매개변수 갱신 전 전처리 해주는 기능

## 46.2 SGD 클래스 구현
- 경사하강법으로 매개변수를 갱신하는 클래스 구현 
- (`dezero/optimizers.py`)
    - `SGD` 는 확률적경사하강법(Stochastic Gradient Descent)의 약자
    - 확률적이라는 말은 데이터 중 무작위(확률적으로) 선별한 데이터에 대해 경사하강법을 수행하는 것 

## 46.3 SGD 클래스를 사용한 문제 해결
- (`steps/step45.py`)

## 46.4 SGD 이외의 최적화 기법
- 대표적인 최적화 기법 Momentum, AdaGrad, AdaDelta, Adam 등등
- Optimizer 클래스를 도입한 첫번째 목표는 이처럼 다양한 최적화 기법을 손쉽게 전환하기 위함 
- Momentum 구현 (`dezero/optimizers.py`)
    - $W \leftarrow W+v$ 여기서 $v$는 물리학에서의 속도를 의미
    - $v \leftarrow \alpha v - \eta \frac{\partial L}{\partial W}$ $W$는 갱신할 매개변수,   
    $\frac{\partial L}{\partial W}$ 는 기울기($W$에 관한 손실함수 $L$의 기울기), $\eta$는 학습률,  
    $\alpha v$는 물체가 아무 힘도 받지 않을 때 서서히 감속시키는 역할($\alpha$ 는 0.9로 설정)
    - 속도에 해당하는 데이터들은 `self.vs` 에 유지
    - `update_one`이 호출될 때 매개변수와 같은 타입의 데이터를 생성

---

</details>

<details>

<summary>step 47 : 소프트맥스 함수와 교차 엔트로피 오차</summary>

---
## 47.1 슬라이스 조작 함수
- 다중 클래스 분류(multi-class classification) 도전
- 사전 준비로 `get_item` 이라는 편의 함수를 하나 추가한다.
- __구현은 부록 B__ (`dezero/functions.py`)
```python
import numpy as np
import dezero.functions as F
from dezero import Variable

x = Variable(np.array([[1,2,3], [4,5,6]]))
y = F.get_item(x, 1)
print(y)
```
```
# 결과
variable([4 5 6])
```
- 다차원 배열 중 일부를 슬라이스하여 뽑아준다.
- 역전파도 수행한다. 
```python
y.backward()
print(x.grad)
```
```
variable([[0. 0. 0.]
          [1. 1. 1.]])
```
- 슬라이스로 인한 계산은 데이터 일부를 수정하지 않고 전달하는 것이다.
- 따라서 그 역전파는 원래의 다차원 배열에서 데이터가 추출된 위치에 해당 기울기를 설정하고 그 외에는 0으로 설정한다. 

## 47.2 소프트맥스 함수
- (`steps/step47.py`)
    - x의 형상은 (1, 2)
    - 신경망 출력은 (1, 3) -> 3개의 클래스
    - 이 신경망의 출력은 단순한 '수치'인데, 이 수치를 '확률'로 변환할 수 있다.  
-> 소프트맥스 함수(softmax function)
    $$p_k = \frac{\exp (y_k)}{\sum^{n}_{i=1}(\exp (y_i))}$$ 
    - n은 클래수 수, k 번째 출력 $p_k$를 구하는 식
    - 분자는 입력 $y_k$의 지수함수, 분모는 모든 입력의 지수함수의 총합
- 배치(batch) 데이터에도 소프트맥스 함수를 적용할 수 있도록 확장한다.
- (`dezero/functions.py`)
    - x는 2차원이라고 가정한다.

## 47.3 교차 엔트로피 오차
- 선형 회귀에서는 손실 함수로 평균 제곱 오차(mse)
- 다중 클래스 분류에서 손실 함수는 교차 엔트로피 오파(cross entropy error)
$$L = -\sum_k(t_k \log{p_k})$$
- $t_k$는 정답 데이터의 k 차원째 값을 나타내며, 정답이면 1 아니면 0으로 기록된다.
- 이러한 표현 방식을 원핫 벡터(one-hot vector)라고 한다.
    - 벡터를 구성하는 여러 원소 중 하나만 핫(hot, 값이 1이다)하다는 뜻
- 예로 $t = (0, 0, 1)$, $p = (p_0, p_1, p_2)$ 인 경우 대입하면 $L=-\log{p_2}$ 이다. 그러므로 아래와 같이 간단하게 표현이 가능하다.
$$L=-\log{ p [t]}$$

---

</details>

<details>

<summary>step 48 : 다중 클래스 분류</summary>

---
## 48.1 스파이럴 데이터셋
- 스파이럴 : 나선형 혹은 소용톨이 모양이라는 뜻
- (`steps/step48.py`, `steps/step48_plot.png`)

## 48.2 학습 코드
- (`steps/step48.py`)
    - `softmax_cross_entropy` 클래스 선언이 필요함
- (`steps/step48_loss_plot.png`)
- (`steps/step48_plot.png`)
- (`steps/step48_results.png`)

---

</details>

<details>

<summary>step 49 : Dataset 클래스와 전처리</summary>

---
## 49.1 Dataset 클래스 구현
- 100만개와 같은 거대한 데이터셋을 하나의 ndarray 인스턴스로 처리하면 모든 원소를 한꺼번에 메모리에 올려야하는 문제가 있다.
- Dataset 클래스는 기반 클래스로서의 역할을 하고 사용자가 실제로 사용하는 데이터셋은 이를 상속하여 구현하게 할 것이다. 
- (`dezero/datasets.py`)
    - train 인수는 '학습'이나 '테스트'이나를 구별하기 위한 플래그
    - 인스턴스 변수 data와 label에는 각각 입력 데이터와 레이블을 보관한다.
    - 자식 클래스에서는 prepare 메서드가 데이터 준비 작업을 하도록 구현해야한다. 
    - 중요한 메서드는 `__getitem__`과 `__len__` 이다.
    - 두 메서드(인터페이스)를 제공해야지만 'DeZero 데이터셋'이라고 할 수 있기 때문이다.
        - `__getitem__` 은 파이썬의 특수 메서드로 x[0], x[1]처럼 괄호를 사용해 접근할 때 동작을 정의한다.
        - 단순히 지정된 인덱스에 위치하는 데이터를 꺼냅니다.
        - 레이블 데이터가 없다면 None을 반환한다.
        - `__len__` len 함수를 사용할 때 호출된다.  

## 49.2 큰 데이터셋의 경우
- 데이터셋이 훨씬 크다면..?
```python
class BigData(Dataset):
    def __getitem__(index):
        x = np.load('data/{}.npy'.format(index))
        t = np.load('label/{}.npy'.format(index))
        return x, t

    def __len__():
        return 1000000
```
- 각 100만개씩 데이터가 저장되어 있다고 가정했을 때, BigData 클래스를 초기화 할 때는 데이터를 읽지 않고, 데이터에 접근할 때 비로소 읽게 하는 것이다.
- 'DeZero 데이터셋'이 되기 위한 요건은 `__getitem__`, `__len__` 두 메서드를 구현 하는 것이다.

## 49.3 데이터 이어 붙이기
- 신경망을 학습시킬 때는 데이터셋 중 일부를 미니배치로 꺼낸다.
```python
train_set = dezero.datasets.Spiral()

batch_index = [0, 1, 2]
batch = [train_set[i] for i in batch_index]
# batch = [(data_0, label_0), (data_1, label_1), (data_2, label_2)]
```

## 49.4 학습 코드
- (`steps/step49.py`)
- batch로 불러와서 학습 하는 코드

## 49.5 데이터셋 전처리
- 모델에 데이터를 입력하기 전에 데이터를 특정한 형태로 가공하는 전처리가 많다.
- 예로 이미지 회전, 좌우 반전 등등 인위적으로 늘리는 기술 -> 데이터 확장(data augmentation)
- (`dezero/datasets.py`)
```python
def f(x):
    y = x / 2.0
    return y

train_set = dezero.datasets.Spiral(trainsform=f)
```
- 입력 데이터의 1/2로 스케일 변환하는 전처리 예
- noramlize 하는 코드, 순서대로 전처리를 하는 코드 -> 파일을 참조..
---

</details>

<details>

<summary>step 50 : 미니배치를 뽑아주는 DataLoader</summary>

---
## 50.1 반복자란
- 반복자(iterator)는 원소를 반복해서 꺼내준다.
- (`steps/step50.py`)
- (`dezero/dataloaders.py`)
    - dataset : Dataset 인터페이스를 만족하는 인스턴스
    - batch_size : 배치 크기
    - shuffle : 에포크별로 데이터셋을 뒤섞을지 여부 

## 50.2 DataLoader 사용하기
- (`steps/step50.py`)

## 50.3 accuracy 함수 구현하기
- (`dezero/functions.py`)

## 50.4 스파이럴 데이터셋 학습 코드
- (`steps/step50.py`)

---

</details>

<details>

<summary>step 51 : MNIST 학습</summary>

---
## 51.1 MNIST 데이터셋
- MNIST 데이터셋 가져오고 살펴보기
- (`dezero/datasets.py`) -> `MNIST` 클래스 가져옴..
- (`dezero/transform.py`) -> 전부 가져옴..
- (`dezero/utils.py`) -> `download function` 메서드들 가져옴..
    - **다시 자세히 살펴봐야함..!!**
- (`steps/step51.py`)

## 51.2 MNIST 학습하기
- (`steps/step51.py`)

## 51.3 모델 개선하기
- 현재 구현된 MLP의 활성화 함수는 시그모이드(sigmoid)
- 역사가 깊은 활성화 함수지만 최근에는 ReLU(rectified linear uniut)가 대세
$$h(x) = \begin{cases}x & (x > 0)\\0 & (x \leq 0)\end{cases} $$
- (`dezero/functions.py`)
    - `forward` 에서 `np.maximum(x, 0.0)` 에 의해 x의 원소와 0.0 중 큰 쪽을 반환한다.
    - `backward` 에서 0 이하일 경우 기울기를 0으로 설정해야하기에 mask 를 이용해 출력에서 전해지는 기울기를 통과시킬지 정한다. 
- (`steps/step51.py`)
    - 3층 신경망 사용
    - 활성화 함수 ReLU 사용
- (칼럼 내용은... 추후에 정리해서 추가하자...)

---

</details>

<details>

<summary>step 52 : GPU 지원</summary>

---
## 52.1 쿠파이 설치 및 사용 방법
- 딥러닝 계산은 '행렬의 곱'이 대부분
- 행렬의 곱은 곱셈과 덧셈으로 병렬로 계산하는게 가능하고, GPU가 훨씬 뛰어난다.
```bash
$ pip install cupy
```
- 쿠파이의 장점은 넘파이와 API가 거의 같다는 것이다.
```python
import cupy as cp

x = cp.arange(6).reshape(2, 3)
print(x)

y = x.sum(axis=1)
print(y)
```
- 쿠파이로 바꾸기 위해서는 두가지를 알아야 한다.
```python
import numpy as np
import cupy as cp
# 넘파이 -> 쿠파이
n = np.array([1, 2, 3])
c = cp.asarray(n)
assert type(c) == cp.ndarray

# 쿠파이 -> 넘파이
c = cp.array([1, 2, 3])
n = cp.asnumpy(c)
assert type(n) == np.ndarray
```
- 첫번째 : 서로 전환이 가능하나 이는 최소화 해야한다. 왜냐하면, 메인 메모리에서 GPU 메모리로 전송되는 과정에서 다량의 데이터의 경우 병목이 발생하기 때문이다.
```python
# x가 넘파이 배열인 경우 
x = np.array([1, 2, 3])
xp = cp.get_array_module(x)
assert xp == np

# x가 쿠파이 배열인 경우
x = cp.array([1, 2, 3])
xp = cp.get_array_module(x)
assert xp == cp
``` 
- 두번째 : `get_array_module(x)`은 x 배열에 적합한 모듈을 돌려준다.
### 설치가 안되는데...? ㅎㅎㅎ 우선 코드는 작성해보자...

## 52.2 쿠다 모듈
- (`dezero/cuda.py`)

## 52.3 Variable/Layer/DataLoader 클래스 구현
- (`dezero/core.py`)
- (`dezero/layers.py`)
- (`dezero/dataloader.py`)

## 52.4 함수 수가 구현
- (`dezero/functions.py`)
- (`dezero/core.py`)

## 52.5 GPU로 MNIST 학습하기
- (`steps/step52.py`)

---

</details>

<details>

<summary>step 53 : 모델 저장 및 읽어오기</summary>

---
## 53.1 넘파이의 save 함수와 load 함수
- 모델이 가지는 매개변수를 외부 파일로 저장하고 다시 읽어오는 기능 구현
- 학습 중인 모델의 '스냅샷'을 저장하거나 이미 학습된 매개변수를 읽어와서 추론만 수행할 수 있다.
- DeZero의 매개변수는 Parameter 클래스로 구현되어 있고, Parameter의 데이터는 인스턴스 변수 data 에 ndarray 인스턴스로 보관되므로 이 인스턴스를 외부 파일로 저장하는 것
- (`steps/step53.py`)

## 53.2 Layer 클래스의 매개변수를 평평하게
- Layer 클래스는 계층의 구조를 표현한다. 계층은 Layer 안에 다른 Layer 가 들어가는 중첩 형태의 구조를 취한다.
```python
layer = Layer()

l1 = Layer()
l1.p1 = Parameter(np.array(1))

layer.l1 = l1
layer.p2 = Parameter(np.array(2))
layer.p3 = Parameter(np.array(3))
```
- 위와 같은 계층구조를 하나의 평평한 딕셔너리 로 뽑아내기위해 Layer 클래스에 _flatten_params 메서드를 추가한다.
```python
params_dict = {}
lay._flatten_params(params_dict)
print(params_dict)

>>> {'p2':variable(2), 'l1/p1':variable(1), 'p3':variable(3)}
```
- (`dezero/layers.py`)

## 53.3 Layer 클래스의 save 함수와 load 함수 
- (`dezero/layers.py`)
    - `save_weights` 메서드는 먼저 `self.to_cpu()`를 호출하여 데이터가 메인 메모리에 존재함을 보장한다.
    - ndarray 로 이뤄진 딕셔너리를 만들고 `np.savez_compressed` 함수를 호출하여 데이터를 외부 파일로 저장한다.
    - `load_weights`메서드는 `np.load` 함수로 데이터를 읽은 후 대응하는 키 데이터를 매개변수로 설정한다. 

---

</details>

<details>

<summary>step 54 : 드롭아웃과 테스트 모드</summary>

---
## 54.1 드롭아웃이란 
- 신경망 학습에서는 과대적합이 자주 문제이고 그 원인과 대안은
    - 훈련 데이터가 적음 -> 데이터를 확보하거나 인위적으로 늘리는 데이터 확장을 이용
    - 모델의 표현력이 지나치게 높음 -> 가중치 감소(Weight Decay), 드롭아웃(Dropout), 배치 정규화(Batch Normalization) 등이 유효함
- 드롭아웃을 적용하려면 학습과 테스트를 구분해서 처리해야한다.
- 드롭아웃은 뉴런을 임의로 삭제(비활성화)하면서 학습하는 방법이다.
- 학습 시 은닉층 뉴런을 무작위로 골라 삭제한다.
- (`steps/step54.py`)
- 앙상블과 dropout은 가까운 관계. 
    - 앙상블은 독립적으로 여러 모델을 학습한 후 테스트 시 출력한 값들의 평균을 결과로 낸다.
    - dropout은 학습 시 임의로 뉴런을 삭제하는데, 이를 매번 다른 모델을 학습하고 있다고 해석할 수 있기 때문이다.

## 54.2 역 드롭아웃
- 역 드롭아웃은 스케일 맞추기를 '학습할 때' 수행한다.
- 앞 절에서 스케일을 맞추기 위해 '테스트할 때' scale을 곱했다. 
- 그래서 이번에는 학습할 때 미리 1/scale 을 곱해두고 테스트 때는 아무런 동작도 하지 않는다. 
- 역 드롭아웃도 보통의 드롭아웃과 원리는 같지만 나름의 장점이 있다.
    - 테스트 시 아무런 처리를 하지 않기 떄문에 속도가 살짝 향상
    - 학습할 때 dropout_ratio를 동적으로 변경할 수 있다.
    - 예로 첫 학습때는 0.433 두번째 학습때는 0.563 등등

## 54.3 테스트 모드 추가
- 드롭아웃을 사용하려면 학습 단계인지 테스트 단계인지 구분해야 한다.
- (`dezero/core.py`)
    - `Config` 에 train 변수를 추가
    - 그 다음 `test_mode` 함수를 추가. with 문과 함께 사용하면 with 블록 안에서만 `Config.train`이 False 로 전환된다.
- (`dezero/__init__.py`) 
    - `Config` 와 `test_mode` 추가

## 54.4 드롭아웃 구현
- (`dezero/functions.py`)


---

</details>

<details>

<summary>step 55 : CNN 메커니즘(1)</summary>

---
## 55.1 CNN 신경망의 구조
- CNN, 합성곱 신경망, Convolutional Neural Network
- 이미지 인식, 음성 인식, 자연러 처리 등 다양한 분야에서 사용된다.
- CNN도 지금까지 다룬 신경망처럼 계층을 조합해서 만든다.
- 합성곱층(convolution layer), 풀링층(pooling layer) 등장
- 지금까지의 `Linear -> ReLU` 가 `Conv -> ReLU -> (Pool)` 로 대체되었다고 생각하면 된다.

## 55.2 합성곱 연산
- CNN에는 합성곱층이 사용된다.
- 이미지 처리에서 말하는 '필터 연산'에 해당된다.
- 입력 데이터에 필터 윈도(window)를 일정 간격으로 이동시키며 적용한다. 
- 필터와 입력의 해당 원소를 곱하고 총합을 구한다.
- `입력 (4, 4) * 필터 (3, 3) -> 출력 (2, 2)`
- 필터(filter)는 커널(kernel)이라고도 쓰인다. 
- 합성곱층도 완전연결계층처럼 '편향'이 존재한다. 
- 편향은 필터링 후 더해준다. 여기서 편향은 하나뿐. 브로드캐스트 되어 더해진다.

## 55.3 패딩
- 패딩 처리 : 합성곱층의 주요 처리 전 입력 데이터 주위에 고정값(가령 0등)을 채우는 것
- `입력 (4+2, 4+2) * 필터 (3, 3) -> 출력 (4, 4)`
- 사용하는 주된 이유는 출력 크기를 조정하기 위해서이다.
- 합성곱 연산을 거칠 때 공간이 축소되다보면 어느 순간 합성곱 연산을 할 수 없게 되기 떄문이다.


## 55.4 스트라이드
- 스트라이드(stride, 보폭) : 필터를 적용하는 위치의 간격
- `입력 (7, 7) * 필터 (3, 3) -> (스트라이드 2) -> (3, 3)`

## 55.5 출력 크기 계산 방법
- 패딩을 늘리면 출력 데이터의 크기가 커지고, 스트라이드를 크게하면 반대로 작아진다.
- 출력 크기에 영향을 주는 패딩과 스트라이드를 활용해 출력 크기를 계산하는 식은 다음과 같다.
- (`steps/step55.py`)
- (`dezero/utils.py`)

---

</details>

<details>

<summary>step 56 : CNN 메커니즘(2)</summary>

---
## 56.1 3차원 텐서
- 사진에서 가로/세로 뿐 아니라 RGB처럼 '채널 방향'으로도 데이터가 쌓여있기 때문에 3차원 데이터(3차원 텐서)를 다뤄야 한다.
- 합성곱 연산 절차는 2차원 텐서일 때와 똑같다. 깊이 방향으로 데이터가 늘어난 것을 제외하면 필터가 움직이는 것, 계산 방법도 동일하다.
- 주의점은 '채널'수를 똑같이 맞춰줘야한다는 것이다.

## 56.2 블록으로 생각하기
- 3차원 텐서에 대한 합성곱 연산은 직육면체 블록을 생각하면 이해가 쉽다.
- 데이터가 (채널(channel), 높이(height), 너비(width)) 순서로 정렬되어 있다고 가정하자.
- `입력 (C, H, W) * 필터 (C, KH, KW) -> 출력 (1, OH, OW)`, 특징 맵(feature map)
- 여러개의 특징맵을 갖고 싶다면, 다수의 필터(가중치)를 사용하면 된다.
- 입력 (C, H, W) * 필터 (OC, KH, KW) -> 출력 (OC, OH, OW)
    - 위처럼 합성곱 연산에서는 필터 수도 고려해야 한다. 따라서 필터의 가중치 데이터는 4차원 텐서인 (output_channel, input_channel, height, width) 형상으로 관리한다. 
    - 예로 채널수가 3, 가로세로가 (5, 5)인 필터가 20개 있다면, (20, 3, 5, 5)
- 여기서도 편향이 존재한다. 편향까지 추가한다면,  
`입력 (C, H, W) * 필터 (OC, C, KH, KW) -> (OC, OH, OW) + 편향 (OC, 1, 1) -> 출력 (OC, OH, OW)`

## 56.3 미니배치 처리
- 여러 개의 입력 데이터를 하나의 단위(미니 배치)로 묶어 처리한다.
- 미니 배치 처리를 위해서는 각 층을 흐르는 데이터를 4차원 텐서로 취급한다.
- 예로 N개의 데이터로 이루어진 미니 배치에 합성곱 연산을 수행하면,  
`입력 (N, C, H, W) * 필터 (OC, C, KH, KW) -> (N, OC, OH, OW) + 편향 (OC, 1, 1) -> 출력 (N, OC, OH, OW)`

## 56.4 풀링층
- 풀링은 가로, 세로 공간을 작게 만드는 연산이다.
- 2 X 2 Max 풀링(최대값을 취하는 연산)을 스트라이드 2로 수행하는 경우,  
`입력 (4, 4) -> (2 X 2) Max 풀링 & 스트라이드 2 -> 출력 (2, 2)`
- 일반적으로 풀링 윈도 크기와 스트라이드 크기는 같은 값으로 설정한다.
- 풀링층의 주요 특징
    - 학습하는 매개변수가 없다.
    - 채널 수가 변하지 않는다.  
        - 계산은 채널마다 독립적으로 이루어진다.
    - 미세한 위치 변화에 영향을 덜 받는다.
        - 입력 데이터의 차이가 크지 않으면 풀링 결과도 크게 달라지지 않는다.
        - 이를 입력 데이터의 미세한 차이에 강건하다고도 한다.

---

</details>


<details>

<summary>step 57 : conv2d 함수와 pooling 함수</summary>

---
## 57.1 im2col에 의한 전개
- 합성곱 연산을 곧이곧대로 구현하면 for 문이 겹겹이 중첩된 코드가 될 것이다.
- 속도도 느려진다. 
- for 문을 사용하지 않고 im2col 이라는 편의 함수를 사용하여 간단히 구현하고자 한다.
- im2col은 데이터를 한줄로 '전개'하는 함수로, 합성곱 연산 중 커널 계산에 편리하도록 입력데이터를 펼쳐준다.
- 커널을 적용할 영역을 꺼낸 다음 한줄로 형상을 바꿔 최종적으로는 '행렬(2차원 텐서)'로 변환한다. 

## 57.2 conv2d 함수 구현
- DeZero 의 im2col 함수를 블랙박스 처럼 사용한다고 가정한다.(상세구현은 신경쓰지 않는다.)
- DeZero의 im2col 함수의 인터페이스는 다음과 같다.
```
im2col(x, kernel_size, stride=1, pad=0, to_matrix=True)
```
|인수| 데이터 타입| 설명|
|:--:|:--:|:--:|
|x|Variable 또는 ndarrya|입력데이터|
|kernel_size|int 또는 (int, int)|커널 크기|
|stride|int 또는 (int, int)|스트라이드|
|pad|int 또는 (int, int)|패딩|
|to_matrix|bool|행렬로 형상 변환 여부|
- (`dezero/functions_conv.py`) -> 가져옴...  
> ~~함수가 안불러와지네..??~~  
-> 내 실수.. 확장자가 없었음... `dezero/functions_conv`
- reshape의 마지막 인수를 -1 로 지정하면 그 앞의 인수들로 정의한 다차원 배열에 전체 원소들을 적절히 분배해준다.
> TypeError: transpose() takes 1 positional argument but 5 were given  
-> 해결하려면 `dezero/core.py`, `dezero/functions.py`의 transpose 관련 함수를 가져와야함!!!

## 57.3 Conv2d 계층 구현
- 계층으로서의 Conv2d 구현
- (`dezero/layers.py`)  

|인수| 데이터 타입| 설명|
|:--:|:--:|:--:|
|out_channels|int|출력 데이터의 채널 수|
|kernel_size|int 또는 (int, int)|커널 크기|
|stride|int 또는 (int, int)|스트라이드|
|pad|int 또는 (int, int)|패딩|
|nobias|bool|편향 사용 여부|
|dtype|numpy.dtype|초기화할 가중치의 데이터 타입|
|in_channels|int 또는 None|입력 데이터의 채널 수|

## 57.4 pooling 함수 구현
- im2col을 사용하여 입력 데이터를 전개
- 풀링은 채널 방향과는 독립적이라는 점이 합성곱층과 다르다.
- 채널마다 독립적으로 전개한다.
- (책의 그림을 같이 봐야함)

---

</details>

<details open>

<summary>step 58 : 대표적인 CNN(VGG16)</summary>

---
## 58.1 VGG16 구현
- VGG는 2014년 ILSVRC 대회에서 준우승한 모델
- '3 X 3 conv 64' 는 커널 크기가 3 X 3 이고 출력 채널 수가 64개라는 뜻
- 'pool/2' 는 2 X 2 풀링
- 'Linear 4096' 은 출력 크기가 4096 인 완전연결계층
- VGG16의 특징
    - 3 X 3 합성곱층 사용(패딩은 1 X 1)
    - 합성곱층의 채널 수는 (기본적으로) 풀링하면 2배로 증가 (64 -> 128 -> 256 -> 512)
    - 완전연결계층에서는 드롭아웃 사용
    - 활성화 함수로는 ReLU 사용
- (`dezero/models.py`)
    - 입력 데이터의 채널 수는 지정하지 않는다. 순전파 시 흐르는 데이터로부터 얻는다. 
    - fc 레이어에서도 출력크기만 정한다. 
    - 합성곱층에서 완전연결계층으로 전환하기 위해 데이터 형상을 변환한다.

## 58.2 학습된 가중치 데이터
- VGG16은 ImageNet의 거대한 데이터셋으로 학습한다. 학습이 완료된 가중치 데이터가 공개되어있다. 
- 학습된 가중치를 읽어오는 기능을 추가한다.
- (`dezero/models.py`)

## 58.3 학습된 VGG16 사용하기
- (`steps/step58.py`)
- (`dezero/models.py`) 정적 메서드인 preprocess 추가
    - preprocess는 정적 메서드이므로 인스턴스가 아닌 클래스에서 호출해야 한다.
    - 학습된 가중치 데이터를 사용하면 새로운 데이터를 추론할 떄 학습했을 때와 똑같은 전처리를 해줘야 한다. 모델에 입력되는 형태가 달라지기 떄문에 올바른 인식을 못한다.
- (`dezero/datasets.py`) ImageNet 클래스 추가
---

</details>

<details open>

<summary>step 59 : RNN을 활용한 시계열 데이터 처리</summary>

---
## 59.1 RNN 계층 구현
- 지금까지는 feed forward 구조의 신경망을 살펴봤다.
- 신호가 한 방향으로만 흘러가기 떄문에 입력 신호만으로 출력을 결정한다.
- 순환 신경망(Recurrent Neural Network, RNN)은 순환(Loop) 구조를 갖는다.
- 순환 구조여서 출력은 자신에게 피드백되기 떄문에 '상태'를 가지게 된다.
- RNN 수식을 이해해보자
    - 시계열(time serise) 데이터인 입력 $x_t$ 가 있고, 은닉 상태 $h_t$ 를 출력하는 RNN을 생각해보자
    $$h_t = \tanh(h_{t-1}W_h + x_tW_x + b)$$
    - 2개의 가중치 
        - $W_x$ : 입력 x를 은닉 상태 h로 변환하기 위한 가중치
        - $W_h$ : 출력을 다음 시각의 출력으로 변환하기 위한 가중치 
- (`dezero/layers.py`)
- 두번째 입력 데이터가 처리된 후 그래프가 '성장'하여 더 큰 계산 그래프가 만들어진다.
- '성장'을 가능하게 하는 매개체가 RNN의 은닉상태다.

## 59.2 RNN 모델 구현
- (`steps/step59.py`)
- 중요한 역전파 구현. (책의 이미지를 같이 봐야함. 그림 59-4)두번째 입력 데이터가 들어왔을 때 역전파를 수행함
- 입력 데이터로 구성된 계산 그래프에서 역전파를 '시간을 거슬러 역전파한다'는 의미로 BPTT(Backpropagataion Through Time)
    - RNN은 입력 데이터가 나열되는 패턴을 학습할 수 있다. 이떄의 나열되는 '순서'는 시계열의 '시간'에 해당한다. BPTT에 time이 들어가는 이유다.
- 입력 데이터가 몇개가 되든 계산 그래프는 계속 길게 뻗어나간다. 하지만 역전파를 잘하려면 적당한 길이에서 '끊어줘야'한다. -> Truncated BPTT(truncate 는 '길이를 줄이다', '절단하다'라는 뜻)
- Truncated BPTT 를 수행할 떄는 은닉상태가 유지된다는 점에 주의
- 이전의 은닉 상태에서 시작해서 그 상태 변수에서 계산의 '연결'을 끊어줘야 한다...?
- 그러면 이전 학습에서 사용한 계산 그래프로까지 기울기가 흐르지 못하게 된다. 이게 Truncated BPTT 이다.

## 59.3 '연결'을 끊어주는 메서드
- (`dezero/core.py`)
    - `unchain_backward`메서드는 호출된 변수에서 시작하여 계산 그래프를 거슬러 올라가며 마주치는 모든 변수의 unchain 메서드를 호출한다.

## 59.4 사인파 예측
- (`dezero/datasets.py`) SinCurve 클래스 추가
- (`dezero/optimizers.py`) Adam 클래스 추가
- (`steps/step59.py`)
---


</details>

<details open>

<summary>step 60 : LSTM과 데이터 로더</summary>

---
## 60.1 시계열 데이터용 데이터 로더
- 시계열 데이터를 미니배치로 처리하려면 데이터를 뽑는 시작 위치를 배치별로 다르게 지정해야 한다.
- (`dezero/dataloaders.py`)
- (`steps/step60.py`)

## 60.2 LSTM 계층 구현
> 계산식 이해가 잘... 안되는.. LSTM은 다시 찾아보자!  
-> 밑바닥부터 시작하는 딥러닝 2 의 6장 참고
- LSTM 에서는 은닉 상태 h 외에도 기억 셀 c를 사용한다.
- (`dezero/layers.py`)

---


</details>