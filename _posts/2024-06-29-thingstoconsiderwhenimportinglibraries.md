---
title: '임포트 방식의 차이'
date: 2024-07-01
permalink: /posts/2024/06/thingstoconsiderwhenimportinglibraries/
tags:
  - cool posts
  - category1
  - category2
---

# 임포트 방식의 차이

# 0. 계기

사실 지금까지 라이브러리를 임포트할 때 크게 의미를 두지 않고 그냥 그런갑다 하면서 임포트를 하다가 이번에 미션을 수행하면서 아래 두 임포트 방식의 차이가 눈에 띄었다.   


```python
from torchvision import transforms
```


```python
from torchvision.transforms import ToTensor
```

**Transforms**는 torchvision 라이브러리 안에 있는 이미지 변환 모듈이고, **ToTensor**는 transforms 모듈 안에 있는 클래스로 이미지 데이터를 텐서 형태로 변환해주는 역할을 한다. 데이터를 가져올 때 아래와 같은 방식으로 사용된다.
```
training_data = datasets.FashionMNIST(
    root = 'data',
    train = True,
    download = True,
    transform = ToTensor()
)
```

왜 어떨 때는 from torchvision import transforms라고 쓰고 어떨 때는 from torchvision.transforms import ToTensor라고 쓰는지 궁금했다. 이번 기회에 짚고 넘어갈 겸 간단히 정리를 해보려고 한다.

## 1. 비교


```python
from torchvision import transforms
```

- torchvision.transforms 모듈 **전체**를 임포트함
- transforms 모듈 내의 모든 클래스와 함수에 접근할 수 있음


```python
from torchvision.transforms import ToTensor
```

- torchvision.transforms 모듈에서 **ToTensor 클래스만** 임포트함
- transforms 모듈의 다른 클래스나 함수는 사용할 수 없음

## 2. 의문점과 그에 대한 대답

- Q: 그냥 항상 다 임포트해서 쓰면 되는거 아닌가?
  - A: 다 임포트해서 쓰면 사용하지 않는 기능까지 임포트가 됨. 이러면 몇 가지 문제가 있음
    - 큰 프로젝트가 될수록 **메모리**가 낭비됨
    - 코드의 **초기 로딩 시간**이 길어짐
    - 두 개의 다른 모듈에 같은 이름의 함수가 있을 경우 **이름 충돌**이 발생할 수 있음
    - 사용하는 것만 임포트하면 코드의 **가독성**이 좋아짐. 코드가 어떤 기능을 사용하고 있는지 명확히 알 수 있음. 그리고 다른 사람들이 코드의 의도를 더 잘 이해할 수 있기 때문에 코드의 **유지보수**도 쉬워짐

## 3. 결론

- 작은 규모의 코드에서는 모든 모듈을 임포트해도 큰 문제는 없지만, **실제 프로젝트나 협업환경**에서는 **필요한 모듈만 임포트**하는 것이 좋음

## 4. 소감

- 오호 그렇군.. 이제 **필요한 모듈만 임포트**하는 습관을 들여봐야 겠다.
