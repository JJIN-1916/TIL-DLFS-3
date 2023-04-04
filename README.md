## Deep-Learning-from-Scratch-3

<details>
<summary>step01</summary>

1.1 변수란 
- 상자와 데이터는 별개
- 상자에 데이터가 들어감(할당)
- 상자 속을 보면 데이터를 알 수 있음(참조)

1.2 Variable 클래스 구현
- Variable 클래스 선언 
- 클래스로 인스턴스를 만듬. 인스턴스는 데이터를 담은 상자가 됨

1.3 [보충] 넘파이의 다차원 배열
- 0차원 배열(0차원 텐서) : 스칼라(scalar)
- 1차원 배열(1차원 텐서) : 벡터(vector) -> 축이 1개
- 2차원 배열(2차원 텐서)  : 행렬(matrix) -> 축이 2개
- 다차원 배열 : 텐서(tensor)
- 3차원 벡터와 3차원 배열은 다른 의

</details>

<details>
<summary>step02</summary>

2.1 힘수란 
- 어떤 변수로부터 다른 변수로의 대응 관계를 정한 것 (x -> *f* -> y)

2.2 Function 클래스 구현
- Variable 인스턴스를 다룰 수 있는 함수를 Function 클래스로 구현  
    - Function 클래스는 Variable 인스턴스를 입력받아 Variable 인스턴스를 출력함
    - Variable 인스턴스의 실제 데이터는 인스턴스 변수인 data에 있음

2.3 Function 클래스의 이용 
- DeZero 함수의 충족 사항   
    - Function 클래스는 기반 클래스로서, 모든 함수에 공통되는 기능을 구현한다.
    - 구체적인 함수는 Function 클래스를 상속한 클래스에서 구현한다.
- Note : Function 클래스의 forward 메서드는 예외를 발생시킨다. 이렇게 해두면 Function 클래스의 forward 메서드를 직접 호출한 사람에게 '이 메서드는  상속하여 구현해야 한다.' 는 사실을 알려줄 수 있다

</details>
