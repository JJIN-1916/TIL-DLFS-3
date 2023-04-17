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