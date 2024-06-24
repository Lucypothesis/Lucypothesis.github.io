# Convolutional Neural Networks를 직접 구성하여 99% 이상의 성능을 내는 MNIST 분류기 만들기

1. Convolution 연산, Flatten 연산, Fully Connected 레이어, Activation 연산만을 가지고 MNIST 분류기 만들기
2. 1에 Max Pooling, Average Pooling 레이어를 추가하여 MNIST 분류기 만들기
3. 2의 Pooling연산을 제거하고 Adaptive Pooling을 적절히 활용하여 MNIST 분류기 만들기

난이도(중)의 첫 번째 미션이다. 텐서플로우와 파이토치 중 어떤거로 구현할까 고민하다가 두 개의 장단점을 찾아본 뒤 결론은 그냥 파이토치로 구현하기로 했다. 결정적인 이유는 스터디 그룹 중 한 분이 텐서플로우로 구현을 하셔서.. 괜히 파이토치로도 해보고 싶었던 게 컸다.

## 0. 라이브러리 임포트 & 전처리


```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
```


```python
# 이미지 전처리

# transforms.Compose: 여러 데이터 변환 작업을 하나로 묶어주는 함수. 두가지 변환을 사용함
# 1) transforms.ToTensor(): [0,255] 범위의 이미지 데이터를 [0,1] 범위의 부동소수점 PyTorch 텐서로 변환함
# 2) transforms.Normalize((0.5,),(0.5,)): 이미지 데이터를 정규화함. MNIST 데이터셋은 평균과 표준편차를 모두 0.5로 정규화
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# 훈련 데이터셋과 테스트 데이터셋 로드

# root='mnist_data': 데이터셋이 저장될 경로를 지정함. 이 디렉토리가 기존에 없다면 자동으로 생성됨
# train=True: 훈련 데이터를 로드할지 여부 지정
# download=True: 데이터셋이 경로에 없으면 인터넷에서 다운로드해서 저장함
# transfer: 앞서 정의한 데이터 변환을 적용함
train_data = datasets.MNIST(root='mnist_data', train=True, download=True, transform=transform)

# 모델 평가에 사용할 데이터기 때문에 train을 False로 함
test_data = datasets.MNIST(root='mnist_data', train=False, download=True, transform=transform)

# 훈련데이터셋과 테스트데이터셋을 배치 단위로 로드

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

# 모델 평가시에는 데이터 순서를 섞을 필요가 없고 모델 성능을 일관되게 평가하기 위해 shuffle을 False로 함
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
```

    Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
    Failed to download (trying next):
    HTTP Error 403: Forbidden
    
    Downloading https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz
    Downloading https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz to mnist_data/MNIST/raw/train-images-idx3-ubyte.gz
    

    100%|██████████| 9912422/9912422 [00:02<00:00, 4557330.09it/s]
    

    Extracting mnist_data/MNIST/raw/train-images-idx3-ubyte.gz to mnist_data/MNIST/raw
    
    Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
    Failed to download (trying next):
    HTTP Error 403: Forbidden
    
    Downloading https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz
    Downloading https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz to mnist_data/MNIST/raw/train-labels-idx1-ubyte.gz
    

    100%|██████████| 28881/28881 [00:00<00:00, 134535.12it/s]
    

    Extracting mnist_data/MNIST/raw/train-labels-idx1-ubyte.gz to mnist_data/MNIST/raw
    
    Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
    Failed to download (trying next):
    HTTP Error 403: Forbidden
    
    Downloading https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz
    Downloading https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz to mnist_data/MNIST/raw/t10k-images-idx3-ubyte.gz
    

    100%|██████████| 1648877/1648877 [00:01<00:00, 1252135.76it/s]
    

    Extracting mnist_data/MNIST/raw/t10k-images-idx3-ubyte.gz to mnist_data/MNIST/raw
    
    Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
    Failed to download (trying next):
    HTTP Error 403: Forbidden
    
    Downloading https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz
    Downloading https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz to mnist_data/MNIST/raw/t10k-labels-idx1-ubyte.gz
    

    100%|██████████| 4542/4542 [00:00<00:00, 11011866.34it/s]

    Extracting mnist_data/MNIST/raw/t10k-labels-idx1-ubyte.gz to mnist_data/MNIST/raw
    
    

    
    

## 1. Convolution 연산, Flatten 연산, Fully Connected 레이어, Activation 연산만을 가지고 MNIST 분류기 만들기

### 1.1 모델 정의


```python
# 모델 정의
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # nn.Conv2d: 2D 컨볼루션 레이어를 정의함
        # 입력채널수(in_channels)가 1이고(∵MNIST데이터셋이 흑백이미지이므로) 출력채널수(out_channels)가 32임. 필터를 32개 쓴다는 뜻
        # kernel_size=3: 3x3크기의 필터를 사용함
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)

        # 첫 번째 Fully Connected Layer
        # 입력크기(in_features)가 32*26*26이고(∵컨볼루션 레이어의 출력을 일렬로 펼친 크기임) 출력크기(out_features)가 128임
        self.fc1 = nn.Linear(32*26*26, 128)

        # 두 번째 Fully Connected Layer
        # 입력크기가 128이고(∵이전 레이어의 출력 크기와 동일) 출력크기가 10임(∵MNIST 데이터셋의 클래스 수(0~9숫자))
        self.fc2 = nn.Linear(128, 10)

    # (필수로 정의해야 함) 데이터가 신경망을 통과할 때의 연산과정 정의
    def forward(self, x):

        # x를 첫번째 컨볼루션 레이어에 통과시키고 ReLU함수 적용
        x = torch.relu(self.conv1(x))

        # 컨볼루션 레이어의 출력을 일렬로 펼침
        # -1은 배치 크기에 맞춰서 자동으로 크기를 조절하라는 뜻임
        x = x.view(-1, 32*26*26)

        # 펼친 출력을 첫 번째 fully connected layer에 통과시키고 ReLu 함수 적용
        x = torch.relu(self.fc1(x))

        # 위 출력을 두 번째 fully connected layer에 통과시키고 softmax 활성화함수 적용
        x = torch.softmax(self.fc2(x), dim=1)
        return x

# CNN 클래스의 인스턴스 생성. 이를 통해 모델의 객체가 만들어지고 이후 객체를 사용하여 모델의 학습과 예측을 수행할 수 있음
model = CNN()

# 손실함수로 CE 사용
criterion = nn.CrossEntropyLoss()

# 최적화 알고리즘으로 Adam 사용
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

### 1.2 모델 학습


```python
# 모델 학습
num_epochs = 10
for epoch in range(num_epochs):

    # 반복할 때마다 총 손실값(running_loss), 정확하게 분류된 예측의 수(correct), 전체 샘플수 초기화(total)
    running_loss = 0.0
    correct = 0
    total = 0
    for i, (images, labels) in enumerate(train_loader, 0):

        # 이전 배치에서 계산된 그래디언트를 초기화. 그래디언트가 누적되는 것을 방지
        optimizer.zero_grad()

        # 모델의 이미지를 입력해서 예측값을 계산함
        output = model(images)

        # 모델의 출력과 실제 레이블을 비교해서 손실값을 계산함
        loss = criterion(output, labels)

        # 손실값을 기준으로 그래디언트를 계산해서 역전파함. 각 파라미터에 대한 그래디언트가 계산되게 됨
        loss.backward()

        # 계산된 그래디언트를 바탕으로 모델의 파라미터를 업데이트
        optimizer.step()


        # 현재 배치의 손실값을 running_loss에 더함
        # item(): 텐서 형식인 loss의 값을 파이썬 숫자 형식으로 변환해서 출력하거나 저장할 수 있도록 함
        running_loss += loss.item()

        # output.data: 모델의 출력 텐서
        # 1: 두 번째 차원(클래스 차원)에서 최대값을 찾도록 지정함
        # torch.max: 두 개의 값(최대값과 그 값의 인덱스)을 반환함
        _, predicted = torch.max(output.data, 1)

        # 에포크 전체 동안의 샘플수를 누적
        # labels.size(0): 현재 배치의 샘플수
        total += labels.size(0)

        # 에포크 전체 동안의 정확한 예측수를 누적
        # (predicted == labels): 모델의 예측값과 실제 레이블을 비교해서 일치하는지 여부를 반환
        correct += (predicted == labels).sum().item()

        if i % 300 == 299:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss / 100:.4f}')
            running_loss = 0.0

    # 1에폭마다 정확도 출력
    print(f'Epoch [{epoch+1}/{num_epochs}] Accuracy: {100 * correct / total:.2f}%')
```

    Epoch [1/10], Step [300/1875], Loss: 5.8965
    Epoch [1/10], Step [600/1875], Loss: 5.0469
    Epoch [1/10], Step [900/1875], Loss: 4.8152
    Epoch [1/10], Step [1200/1875], Loss: 4.7673
    Epoch [1/10], Step [1500/1875], Loss: 4.6430
    Epoch [1/10], Step [1800/1875], Loss: 4.5096
    Epoch [1/10] Accuracy: 81.97%
    Epoch [2/10], Step [300/1875], Loss: 4.4929
    Epoch [2/10], Step [600/1875], Loss: 4.4904
    Epoch [2/10], Step [900/1875], Loss: 4.4745
    Epoch [2/10], Step [1200/1875], Loss: 4.4624
    Epoch [2/10], Step [1500/1875], Loss: 4.4651
    Epoch [2/10], Step [1800/1875], Loss: 4.4658
    Epoch [2/10] Accuracy: 97.07%
    Epoch [3/10], Step [300/1875], Loss: 4.4478
    Epoch [3/10], Step [600/1875], Loss: 4.4401
    Epoch [3/10], Step [900/1875], Loss: 4.4477
    Epoch [3/10], Step [1200/1875], Loss: 4.4501
    Epoch [3/10], Step [1500/1875], Loss: 4.4463
    Epoch [3/10], Step [1800/1875], Loss: 4.4477
    Epoch [3/10] Accuracy: 97.93%
    Epoch [4/10], Step [300/1875], Loss: 4.4259
    Epoch [4/10], Step [600/1875], Loss: 4.4408
    Epoch [4/10], Step [900/1875], Loss: 4.4384
    Epoch [4/10], Step [1200/1875], Loss: 4.4393
    Epoch [4/10], Step [1500/1875], Loss: 4.4422
    Epoch [4/10], Step [1800/1875], Loss: 4.4398
    Epoch [4/10] Accuracy: 98.23%
    Epoch [5/10], Step [300/1875], Loss: 4.4238
    Epoch [5/10], Step [600/1875], Loss: 4.4261
    Epoch [5/10], Step [900/1875], Loss: 4.4226
    Epoch [5/10], Step [1200/1875], Loss: 4.4266
    Epoch [5/10], Step [1500/1875], Loss: 4.4262
    Epoch [5/10], Step [1800/1875], Loss: 4.4310
    Epoch [5/10] Accuracy: 98.60%
    Epoch [6/10], Step [300/1875], Loss: 4.4211
    Epoch [6/10], Step [600/1875], Loss: 4.4193
    Epoch [6/10], Step [900/1875], Loss: 4.4221
    Epoch [6/10], Step [1200/1875], Loss: 4.4219
    Epoch [6/10], Step [1500/1875], Loss: 4.4265
    Epoch [6/10], Step [1800/1875], Loss: 4.4262
    Epoch [6/10] Accuracy: 98.71%
    Epoch [7/10], Step [300/1875], Loss: 4.4146
    Epoch [7/10], Step [600/1875], Loss: 4.4164
    Epoch [7/10], Step [900/1875], Loss: 4.4125
    Epoch [7/10], Step [1200/1875], Loss: 4.4262
    Epoch [7/10], Step [1500/1875], Loss: 4.4145
    Epoch [7/10], Step [1800/1875], Loss: 4.4205
    Epoch [7/10] Accuracy: 98.87%
    Epoch [8/10], Step [300/1875], Loss: 4.4104
    Epoch [8/10], Step [600/1875], Loss: 4.4146
    Epoch [8/10], Step [900/1875], Loss: 4.4119
    Epoch [8/10], Step [1200/1875], Loss: 4.4226
    Epoch [8/10], Step [1500/1875], Loss: 4.4191
    Epoch [8/10], Step [1800/1875], Loss: 4.4232
    Epoch [8/10] Accuracy: 98.89%
    Epoch [9/10], Step [300/1875], Loss: 4.4103
    Epoch [9/10], Step [600/1875], Loss: 4.4078
    Epoch [9/10], Step [900/1875], Loss: 4.4103
    Epoch [9/10], Step [1200/1875], Loss: 4.4073
    Epoch [9/10], Step [1500/1875], Loss: 4.4072
    Epoch [9/10], Step [1800/1875], Loss: 4.4161
    Epoch [9/10] Accuracy: 99.15%
    Epoch [10/10], Step [300/1875], Loss: 4.4054
    Epoch [10/10], Step [600/1875], Loss: 4.4106
    Epoch [10/10], Step [900/1875], Loss: 4.4123
    Epoch [10/10], Step [1200/1875], Loss: 4.4102
    Epoch [10/10], Step [1500/1875], Loss: 4.4059
    Epoch [10/10], Step [1800/1875], Loss: 4.4158
    Epoch [10/10] Accuracy: 99.12%
    

### 1.3 모델 평가


```python
# 모델 평가

# 정확히 예측한 샘플수(correct)와 전체 샘플수(total) 변수 초기화
correct = 0
total = 0

# torch.no_grad(): 그래디언트 계산 비활성화. 평가시에는 모델의 파라미터를 업데이트 안 해도 되기 때문에 메모리 사용량을 줄이고 계산속도를 높일 수 있음
with torch.no_grad():

    # images: 입력 이미지 / labels: 실제 레이블
    for images, labels in test_loader:

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'테스트 정확도: {correct / total}')
```

    테스트 정확도: 0.9798
    

어 정확도가 0.9798밖에 안 나온다. 물론 이것도도 높은 숫자지만 목표는 0.99이므로 조금 아쉽다.    
이를 개선하기 위해서는
1. 데이터 증강
2. 모델 구조 개선
  - 더 깊은 신경망 사용
  - 더 많은 필터 사용
3. Dropout 적용(과적합 방지)
4. 학습률 조정
  - 다른 적절한 학습률 설정하기
  - 학습이 진행될수록 학습률 낮추기
5. 배치 정규화 추가(안정성과 속도)

와 같은 방법들을 사용할 수 있다.

아래에서는 Max Pooling과 Average Pooling 레이어를 추가하는 방법을 사용했다.(2. 모델 구조 개선)   
모델 구조 개선을 위해 Pooling을 사용하는 이유에는 3가지가 있다.   
먼저 Pooling은 주요 특징을 추출하고, **공간 정보를 축소**해서 연산 효율성을 높인다. 그리고 위치 변화에 대해 모델의 **불변성을 증가**시켜 더 일반화된 특성을 학습할 수 있게 하고, 모델의 파라미터 수를 줄여 **과적합을 방지**하는데 도움을 준다. 아래는 Max Pooling과 Average Pooling을 비교한 표이다.

|유형    | 역할 및 설명                                                  
|-----------------|---------------------------------------------------------------|
| **Average Pooling** | - 주어진 영역의 평균 값을 계산, 특징 맵의 정보를 더 부드럽게 반영                     |
| **Max Pooling** | - 주어진 영역의 최대 값을 선택, 가장 두드러진 특징을 강조                            |

## 2. 1에 Max Pooling, Average Pooling 레이어를 추가하여 MNIST 분류기 만들기

순서를 'Convolutional -> Max Pooling -> Convolutional -> Average Pooling -> FC -> FC' 로 배치한 이유가 궁금했다. 챗지피티에게 물어본 결과..    
**Pooling 레이어는 Convolutional 레이어의 출력을 다운샘플링**하는 역할을 하므로 Convolutional 레이어 이후에 위치해야 한다고 한다. 그리고 **FC 레이어는 Convolutional 레이어의 출력을 Flatten해서 1D 벡터로 변환**하는 역할을 하기 때문에 보통 맨 뒤에 배치되지만, 특정 패턴을 배치하거나 복잡한 모델 구조를 구현할 때는 중간에 배치될 수도 있다고 한다.    
오호..

### 2.1 모델 정의


```python
class CNNWithPooling(nn.Module):
    def __init__(self):
        super(CNNWithPooling, self).__init__()

        # 첫 번째 Convolutional 레이어. 입력채널1, 출력채널32, 커널크기 3x3 임
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)

        # 첫 번째 Max Pooling 레이어
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 두 번째 Convolutional 레이어
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)

        # 두 번째 Average Pooling 레이어
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        # 첫 번째 Fully Connected 레이어. 입력크기가 64*5*5이고 출력크기가 128임
        self.fc1 = nn.Linear(64*5*5, 128)

        # 두 번째 Fully Connected 레이어. 입력크기가 128이고 출력크기는 10임(MNIST가 0~9까지의 숫자이기 때문)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):

        # 입력데이터 x를 첫 번째 Convolutional 레이어에 통과시키고 ReLU함수 적용함
        x = torch.relu(self.conv1(x))

        # 첫 번째 Max Pooling 레이어를 적용해서 특징 맵의 크기를 줄임
        x = self.pool1(x)

        # 입력데이터 x를 두 번째 Convolutional 레이어에 통과시키고 ReLU함수 적용함
        x = torch.relu(self.conv2(x))

        # 두 번째 Average Pooling 레이어를 적용해서 특징 맵의 크기를 줄임
        x = self.pool2(x)

        # 2D 텐서를 1D 텐서로 펼쳐서 Fully Connected 레이어에 입력할 준비를 함
        x = x.view(-1, 64*5*5)

        # 첫 번째 Fully Connected 레이어에 통과시키고 ReLU함수 적용함
        x = torch.relu(self.fc1(x))

        # 두 번째 Fully Connected 레이어에 통과시키고  각 샘플(dim=1, 배치 내의 각 행)마다 Softmax함수를 적용해서 각 클래스에 대한 확률 값을 계산함
        x = torch.softmax(self.fc2(x), dim=1)

        # 최종 출력을 반환함
        return x

model = CNNWithPooling()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

### 2.2 모델 학습


```python
num_epochs = 10
for epoch in range(num_epochs):
    running_loss = 0.0
    correct = 0
    total = 0
    for i, (images, labels) in enumerate(train_loader, 0):
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if i % 300 == 299:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss / 100:.4f}')
            running_loss = 0.0

    print(f'Epoch [{epoch+1}/{num_epochs}] Accuracy: {100 * correct / total:.2f}%')
```

    Epoch [1/10], Step [300/1875], Loss: 5.3979
    Epoch [1/10], Step [600/1875], Loss: 5.0624
    Epoch [1/10], Step [900/1875], Loss: 5.0673
    Epoch [1/10], Step [1200/1875], Loss: 4.9433
    Epoch [1/10], Step [1500/1875], Loss: 4.7628
    Epoch [1/10], Step [1800/1875], Loss: 4.7449
    Epoch [1/10] Accuracy: 80.14%
    Epoch [2/10], Step [300/1875], Loss: 4.7165
    Epoch [2/10], Step [600/1875], Loss: 4.7142
    Epoch [2/10], Step [900/1875], Loss: 4.7032
    Epoch [2/10], Step [1200/1875], Loss: 4.4723
    Epoch [2/10], Step [1500/1875], Loss: 4.4639
    Epoch [2/10], Step [1800/1875], Loss: 4.4601
    Epoch [2/10] Accuracy: 93.40%
    Epoch [3/10], Step [300/1875], Loss: 4.4453
    Epoch [3/10], Step [600/1875], Loss: 4.4462
    Epoch [3/10], Step [900/1875], Loss: 4.4374
    Epoch [3/10], Step [1200/1875], Loss: 4.4394
    Epoch [3/10], Step [1500/1875], Loss: 4.4371
    Epoch [3/10], Step [1800/1875], Loss: 4.4363
    Epoch [3/10] Accuracy: 98.11%
    Epoch [4/10], Step [300/1875], Loss: 4.4268
    Epoch [4/10], Step [600/1875], Loss: 4.4241
    Epoch [4/10], Step [900/1875], Loss: 4.4311
    Epoch [4/10], Step [1200/1875], Loss: 4.4328
    Epoch [4/10], Step [1500/1875], Loss: 4.4319
    Epoch [4/10], Step [1800/1875], Loss: 4.4289
    Epoch [4/10] Accuracy: 98.49%
    Epoch [5/10], Step [300/1875], Loss: 4.4254
    Epoch [5/10], Step [600/1875], Loss: 4.4277
    Epoch [5/10], Step [900/1875], Loss: 4.4311
    Epoch [5/10], Step [1200/1875], Loss: 4.4258
    Epoch [5/10], Step [1500/1875], Loss: 4.4207
    Epoch [5/10], Step [1800/1875], Loss: 4.4216
    Epoch [5/10] Accuracy: 98.63%
    Epoch [6/10], Step [300/1875], Loss: 4.4192
    Epoch [6/10], Step [600/1875], Loss: 4.4160
    Epoch [6/10], Step [900/1875], Loss: 4.4190
    Epoch [6/10], Step [1200/1875], Loss: 4.4180
    Epoch [6/10], Step [1500/1875], Loss: 4.4181
    Epoch [6/10], Step [1800/1875], Loss: 4.4166
    Epoch [6/10] Accuracy: 98.88%
    Epoch [7/10], Step [300/1875], Loss: 4.4061
    Epoch [7/10], Step [600/1875], Loss: 4.4125
    Epoch [7/10], Step [900/1875], Loss: 4.4119
    Epoch [7/10], Step [1200/1875], Loss: 4.4175
    Epoch [7/10], Step [1500/1875], Loss: 4.4184
    Epoch [7/10], Step [1800/1875], Loss: 4.4087
    Epoch [7/10] Accuracy: 99.04%
    Epoch [8/10], Step [300/1875], Loss: 4.4149
    Epoch [8/10], Step [600/1875], Loss: 4.4233
    Epoch [8/10], Step [900/1875], Loss: 4.4113
    Epoch [8/10], Step [1200/1875], Loss: 4.4147
    Epoch [8/10], Step [1500/1875], Loss: 4.4135
    Epoch [8/10], Step [1800/1875], Loss: 4.4225
    Epoch [8/10] Accuracy: 98.91%
    Epoch [9/10], Step [300/1875], Loss: 4.4121
    Epoch [9/10], Step [600/1875], Loss: 4.4139
    Epoch [9/10], Step [900/1875], Loss: 4.4115
    Epoch [9/10], Step [1200/1875], Loss: 4.4096
    Epoch [9/10], Step [1500/1875], Loss: 4.4211
    Epoch [9/10], Step [1800/1875], Loss: 4.4082
    Epoch [9/10] Accuracy: 99.02%
    Epoch [10/10], Step [300/1875], Loss: 4.4069
    Epoch [10/10], Step [600/1875], Loss: 4.4059
    Epoch [10/10], Step [900/1875], Loss: 4.4092
    Epoch [10/10], Step [1200/1875], Loss: 4.4086
    Epoch [10/10], Step [1500/1875], Loss: 4.4107
    Epoch [10/10], Step [1800/1875], Loss: 4.4106
    Epoch [10/10] Accuracy: 99.17%
    

### 2.3 모델 평가


```python
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'테스트 정확도: {correct / total}')
```

    테스트 정확도: 0.9873
    

정확도가 0.9873으로, 전 정확도인 0.9828보다 한층 오른 것을 확인할 수 있다. 확실히 Max Pooling 레이어와 Average Pooling 레이어를 추가하니 성능이 좋아졌다. 어떻게 하면 0.99를 뚫을 수 있을까? 과제는 Pooling 연산을 제거하고 Adaptive Pooling을 적절히 활용해보라고 한다.   
Adaptive Pooling을 사용하면 입력 이미지의 크기에 상관 없이 항상 **일정한 크기의 출력이 생성**되어 FC레이어에 전달하기 전에 항상 동일한 크기의 특징맵을 사용할 수 있게 한다. 즉, 모델이 다양한 크기의 입력 데이터에 대해서 더 잘 **일반화**할 수 있도록 도와주고 출력의 크기가 일정하기 때문에 **계산량을 줄이**고 **모델의 파라미터 수를 일정하게 유지**할 수 있다.   

한번 Adaptive Pooling을 사용해서 코드를 만들어보자

## 3. 2의 Pooling연산을 제거하고 Adaptive Pooling을 적절히 활용하여 MNIST 분류기 만들기

### 3.1 모델 정의


```python
class CNNWithAdaptivePooling(nn.Module):
    def __init__(self):
        super(CNNWithAdaptivePooling, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((5, 5))  # 출력 크기를 5x5로 고정
      # self.fc1 = nn.Linear(64, 128)
        self.fc1 = nn.Linear(64*5*5, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.adaptive_pool(x)  # Adaptive Pooling 레이어
        x = x.view(-1, 64*5*5)
        x = torch.relu(self.fc1(x))
        x = torch.softmax(self.fc2(x), dim=1)
        return x

model = CNNWithAdaptivePooling()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

### 3.2 모델 학습


```python
num_epochs = 10
for epoch in range(num_epochs):
    running_loss = 0.0
    correct = 0
    total = 0
    for i, (images, labels) in enumerate(train_loader, 0):
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if i % 300 == 299:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss / 100:.4f}')
            running_loss = 0.0

    print(f'Epoch [{epoch+1}/{num_epochs}] Training Accuracy: {100 * correct / total:.2f}%')
```

    Epoch [1/10], Step [300/1875], Loss: 5.1731
    Epoch [1/10], Step [600/1875], Loss: 4.6241
    Epoch [1/10], Step [900/1875], Loss: 4.5407
    Epoch [1/10], Step [1200/1875], Loss: 4.5050
    Epoch [1/10], Step [1500/1875], Loss: 4.5020
    Epoch [1/10], Step [1800/1875], Loss: 4.4861
    Epoch [1/10] Training Accuracy: 92.27%
    Epoch [2/10], Step [300/1875], Loss: 4.4844
    Epoch [2/10], Step [600/1875], Loss: 4.4762
    Epoch [2/10], Step [900/1875], Loss: 4.4701
    Epoch [2/10], Step [1200/1875], Loss: 4.4614
    Epoch [2/10], Step [1500/1875], Loss: 4.4623
    Epoch [2/10], Step [1800/1875], Loss: 4.4469
    Epoch [2/10] Training Accuracy: 97.35%
    Epoch [3/10], Step [300/1875], Loss: 4.4462
    Epoch [3/10], Step [600/1875], Loss: 4.4502
    Epoch [3/10], Step [900/1875], Loss: 4.4355
    Epoch [3/10], Step [1200/1875], Loss: 4.4449
    Epoch [3/10], Step [1500/1875], Loss: 4.4451
    Epoch [3/10], Step [1800/1875], Loss: 4.4467
    Epoch [3/10] Training Accuracy: 98.02%
    Epoch [4/10], Step [300/1875], Loss: 4.4407
    Epoch [4/10], Step [600/1875], Loss: 4.4393
    Epoch [4/10], Step [900/1875], Loss: 4.4340
    Epoch [4/10], Step [1200/1875], Loss: 4.4407
    Epoch [4/10], Step [1500/1875], Loss: 4.4281
    Epoch [4/10], Step [1800/1875], Loss: 4.4314
    Epoch [4/10] Training Accuracy: 98.28%
    Epoch [5/10], Step [300/1875], Loss: 4.4310
    Epoch [5/10], Step [600/1875], Loss: 4.4286
    Epoch [5/10], Step [900/1875], Loss: 4.4291
    Epoch [5/10], Step [1200/1875], Loss: 4.4277
    Epoch [5/10], Step [1500/1875], Loss: 4.4316
    Epoch [5/10], Step [1800/1875], Loss: 4.4259
    Epoch [5/10] Training Accuracy: 98.54%
    Epoch [6/10], Step [300/1875], Loss: 4.4283
    Epoch [6/10], Step [600/1875], Loss: 4.4205
    Epoch [6/10], Step [900/1875], Loss: 4.4179
    Epoch [6/10], Step [1200/1875], Loss: 4.4306
    Epoch [6/10], Step [1500/1875], Loss: 4.4224
    Epoch [6/10], Step [1800/1875], Loss: 4.4230
    Epoch [6/10] Training Accuracy: 98.68%
    Epoch [7/10], Step [300/1875], Loss: 4.4195
    Epoch [7/10], Step [600/1875], Loss: 4.4197
    Epoch [7/10], Step [900/1875], Loss: 4.4284
    Epoch [7/10], Step [1200/1875], Loss: 4.4166
    Epoch [7/10], Step [1500/1875], Loss: 4.4210
    Epoch [7/10], Step [1800/1875], Loss: 4.4211
    Epoch [7/10] Training Accuracy: 98.78%
    Epoch [8/10], Step [300/1875], Loss: 4.4212
    Epoch [8/10], Step [600/1875], Loss: 4.4153
    Epoch [8/10], Step [900/1875], Loss: 4.4163
    Epoch [8/10], Step [1200/1875], Loss: 4.4156
    Epoch [8/10], Step [1500/1875], Loss: 4.4154
    Epoch [8/10], Step [1800/1875], Loss: 4.4175
    Epoch [8/10] Training Accuracy: 98.89%
    Epoch [9/10], Step [300/1875], Loss: 4.4151
    Epoch [9/10], Step [600/1875], Loss: 4.4188
    Epoch [9/10], Step [900/1875], Loss: 4.4197
    Epoch [9/10], Step [1200/1875], Loss: 4.4112
    Epoch [9/10], Step [1500/1875], Loss: 4.4173
    Epoch [9/10], Step [1800/1875], Loss: 4.4148
    Epoch [9/10] Training Accuracy: 98.92%
    Epoch [10/10], Step [300/1875], Loss: 4.4110
    Epoch [10/10], Step [600/1875], Loss: 4.4158
    Epoch [10/10], Step [900/1875], Loss: 4.4119
    Epoch [10/10], Step [1200/1875], Loss: 4.4153
    Epoch [10/10], Step [1500/1875], Loss: 4.4084
    Epoch [10/10], Step [1800/1875], Loss: 4.4118
    Epoch [10/10] Training Accuracy: 99.03%
    

### 3.3 모델 평가


```python
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'테스트 정확도: {correct / total}')
```

    테스트 정확도: 0.9879
    

완전 처음에 돌렸을 때는 정확도가 0.9219가 나왔다. 이상하리만큼 뚝 떨어진 정확도였다. 문제가 뭘까..하고 고뇌를 하면서 뚫어져라 쳐다보니 내가 첫 번째 FC레이어의 입력크기를 64x5x5로 설정했어야 하는데 그냥 64로 설정해서 생긴 문제였다. 다시 잘 설정하니 정확도가 0.9879로 아주 조금 올랐다.   

하지만 아직도 부족하다. 우리의 목표는 0.99! 무엇을 어떻게 개선해야 할 것인가. 일단 Adaptive Pooling의 출력 크기를 먼저 조정해보려고 한다. 출력 크기를 더 크게하면 더 많은 특징을 유지하게 되고, 작게하면 더 압축된 특징을 제공할 수 있다. 두 가지 방법 모두 해보려고 한다. 과연 어떻게 했을 때 정확도가 올라갈 것인가!   
3.4 ~ 3.6 CNNWithAdaptivePooling7
  - (5,5) -> (7,7)

3.7 ~ 3.9 CNNWithAdaptivePooling3
  - (5,5) -> (3,3)

### 3.4 모델 정의 (CNNWithAdaptivePooling7)


```python
class CNNWithAdaptivePooling7(nn.Module):
    def __init__(self):
        super(CNNWithAdaptivePooling7, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))  # 출력 크기를 7x7로 변경
        self.fc1 = nn.Linear(64*7*7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.adaptive_pool(x)
        x = x.view(-1, 64*7*7)
        x = torch.relu(self.fc1(x))
        x = torch.softmax(self.fc2(x), dim=1)
        return x

model = CNNWithAdaptivePooling7()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

### 3.5 모델 학습 (CNNWithAdaptivePooling7)


```python
num_epochs = 10
for epoch in range(num_epochs):
    running_loss = 0.0
    correct = 0
    total = 0
    for i, (images, labels) in enumerate(train_loader, 0):
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if i % 300 == 299:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss / 100:.4f}')
            running_loss = 0.0

    print(f'Epoch [{epoch+1}/{num_epochs}] Training Accuracy: {100 * correct / total:.2f}%')
```

    Epoch [1/10], Step [300/1875], Loss: 5.1906
    Epoch [1/10], Step [600/1875], Loss: 4.6367
    Epoch [1/10], Step [900/1875], Loss: 4.5464
    Epoch [1/10], Step [1200/1875], Loss: 4.5051
    Epoch [1/10], Step [1500/1875], Loss: 4.4892
    Epoch [1/10], Step [1800/1875], Loss: 4.4724
    Epoch [1/10] Training Accuracy: 91.98%
    Epoch [2/10], Step [300/1875], Loss: 4.4631
    Epoch [2/10], Step [600/1875], Loss: 4.4586
    Epoch [2/10], Step [900/1875], Loss: 4.4527
    Epoch [2/10], Step [1200/1875], Loss: 4.4604
    Epoch [2/10], Step [1500/1875], Loss: 4.4511
    Epoch [2/10], Step [1800/1875], Loss: 4.4469
    Epoch [2/10] Training Accuracy: 97.68%
    Epoch [3/10], Step [300/1875], Loss: 4.4381
    Epoch [3/10], Step [600/1875], Loss: 4.4538
    Epoch [3/10], Step [900/1875], Loss: 4.4419
    Epoch [3/10], Step [1200/1875], Loss: 4.4420
    Epoch [3/10], Step [1500/1875], Loss: 4.4445
    Epoch [3/10], Step [1800/1875], Loss: 4.4337
    Epoch [3/10] Training Accuracy: 98.08%
    Epoch [4/10], Step [300/1875], Loss: 4.4331
    Epoch [4/10], Step [600/1875], Loss: 4.4304
    Epoch [4/10], Step [900/1875], Loss: 4.4377
    Epoch [4/10], Step [1200/1875], Loss: 4.4261
    Epoch [4/10], Step [1500/1875], Loss: 4.4360
    Epoch [4/10], Step [1800/1875], Loss: 4.4235
    Epoch [4/10] Training Accuracy: 98.42%
    Epoch [5/10], Step [300/1875], Loss: 4.4272
    Epoch [5/10], Step [600/1875], Loss: 4.4366
    Epoch [5/10], Step [900/1875], Loss: 4.4256
    Epoch [5/10], Step [1200/1875], Loss: 4.4256
    Epoch [5/10], Step [1500/1875], Loss: 4.4258
    Epoch [5/10], Step [1800/1875], Loss: 4.4275
    Epoch [5/10] Training Accuracy: 98.54%
    Epoch [6/10], Step [300/1875], Loss: 4.4202
    Epoch [6/10], Step [600/1875], Loss: 4.4215
    Epoch [6/10], Step [900/1875], Loss: 4.4241
    Epoch [6/10], Step [1200/1875], Loss: 4.4204
    Epoch [6/10], Step [1500/1875], Loss: 4.4222
    Epoch [6/10], Step [1800/1875], Loss: 4.4225
    Epoch [6/10] Training Accuracy: 98.73%
    Epoch [7/10], Step [300/1875], Loss: 4.4238
    Epoch [7/10], Step [600/1875], Loss: 4.4179
    Epoch [7/10], Step [900/1875], Loss: 4.4161
    Epoch [7/10], Step [1200/1875], Loss: 4.4210
    Epoch [7/10], Step [1500/1875], Loss: 4.4192
    Epoch [7/10], Step [1800/1875], Loss: 4.4177
    Epoch [7/10] Training Accuracy: 98.81%
    Epoch [8/10], Step [300/1875], Loss: 4.4130
    Epoch [8/10], Step [600/1875], Loss: 4.4155
    Epoch [8/10], Step [900/1875], Loss: 4.4133
    Epoch [8/10], Step [1200/1875], Loss: 4.4202
    Epoch [8/10], Step [1500/1875], Loss: 4.4216
    Epoch [8/10], Step [1800/1875], Loss: 4.4116
    Epoch [8/10] Training Accuracy: 98.90%
    Epoch [9/10], Step [300/1875], Loss: 4.4133
    Epoch [9/10], Step [600/1875], Loss: 4.4230
    Epoch [9/10], Step [900/1875], Loss: 4.4219
    Epoch [9/10], Step [1200/1875], Loss: 4.4189
    Epoch [9/10], Step [1500/1875], Loss: 4.4071
    Epoch [9/10], Step [1800/1875], Loss: 4.4186
    Epoch [9/10] Training Accuracy: 98.90%
    Epoch [10/10], Step [300/1875], Loss: 4.4039
    Epoch [10/10], Step [600/1875], Loss: 4.4068
    Epoch [10/10], Step [900/1875], Loss: 4.4132
    Epoch [10/10], Step [1200/1875], Loss: 4.4239
    Epoch [10/10], Step [1500/1875], Loss: 4.4158
    Epoch [10/10], Step [1800/1875], Loss: 4.4125
    Epoch [10/10] Training Accuracy: 99.04%
    

### 3.6 모델 평가 (CNNWithAdaptivePooling7)


```python
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'테스트 정확도: {correct / total}')
```

    테스트 정확도: 0.9869
    

### 3.7 모델 정의 (CNNWithAdaptivePooling3)


```python
class CNNWithAdaptivePooling3(nn.Module):
    def __init__(self):
        super(CNNWithAdaptivePooling3, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((3, 3))  # 출력 크기를 3x3로 변경
        self.fc1 = nn.Linear(64*3*3, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.adaptive_pool(x)
        x = x.view(-1, 64*3*3)
        x = torch.relu(self.fc1(x))
        x = torch.softmax(self.fc2(x), dim=1)
        return x

model = CNNWithAdaptivePooling3()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

### 3.8 모델 학습 (CNNWithAdaptivePooling3)


```python
num_epochs = 10
for epoch in range(num_epochs):
    running_loss = 0.0
    correct = 0
    total = 0
    for i, (images, labels) in enumerate(train_loader, 0):
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if i % 300 == 299:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss / 100:.4f}')
            running_loss = 0.0

    print(f'Epoch [{epoch+1}/{num_epochs}] Training Accuracy: {100 * correct / total:.2f}%')
```

    Epoch [1/10], Step [300/1875], Loss: 5.7708
    Epoch [1/10], Step [600/1875], Loss: 5.1081
    Epoch [1/10], Step [900/1875], Loss: 4.8961
    Epoch [1/10], Step [1200/1875], Loss: 4.8492
    Epoch [1/10], Step [1500/1875], Loss: 4.8234
    Epoch [1/10], Step [1800/1875], Loss: 4.8131
    Epoch [1/10] Training Accuracy: 78.78%
    Epoch [2/10], Step [300/1875], Loss: 4.7863
    Epoch [2/10], Step [600/1875], Loss: 4.7833
    Epoch [2/10], Step [900/1875], Loss: 4.7993
    Epoch [2/10], Step [1200/1875], Loss: 4.7788
    Epoch [2/10], Step [1500/1875], Loss: 4.7610
    Epoch [2/10], Step [1800/1875], Loss: 4.7511
    Epoch [2/10] Training Accuracy: 86.97%
    Epoch [3/10], Step [300/1875], Loss: 4.7093
    Epoch [3/10], Step [600/1875], Loss: 4.4942
    Epoch [3/10], Step [900/1875], Loss: 4.4843
    Epoch [3/10], Step [1200/1875], Loss: 4.4804
    Epoch [3/10], Step [1500/1875], Loss: 4.4798
    Epoch [3/10], Step [1800/1875], Loss: 4.4798
    Epoch [3/10] Training Accuracy: 95.60%
    Epoch [4/10], Step [300/1875], Loss: 4.4727
    Epoch [4/10], Step [600/1875], Loss: 4.4756
    Epoch [4/10], Step [900/1875], Loss: 4.4566
    Epoch [4/10], Step [1200/1875], Loss: 4.4708
    Epoch [4/10], Step [1500/1875], Loss: 4.4632
    Epoch [4/10], Step [1800/1875], Loss: 4.4641
    Epoch [4/10] Training Accuracy: 97.34%
    Epoch [5/10], Step [300/1875], Loss: 4.4618
    Epoch [5/10], Step [600/1875], Loss: 4.4573
    Epoch [5/10], Step [900/1875], Loss: 4.4580
    Epoch [5/10], Step [1200/1875], Loss: 4.4488
    Epoch [5/10], Step [1500/1875], Loss: 4.4469
    Epoch [5/10], Step [1800/1875], Loss: 4.4434
    Epoch [5/10] Training Accuracy: 97.73%
    Epoch [6/10], Step [300/1875], Loss: 4.4479
    Epoch [6/10], Step [600/1875], Loss: 4.4470
    Epoch [6/10], Step [900/1875], Loss: 4.4483
    Epoch [6/10], Step [1200/1875], Loss: 4.4427
    Epoch [6/10], Step [1500/1875], Loss: 4.4515
    Epoch [6/10], Step [1800/1875], Loss: 4.4364
    Epoch [6/10] Training Accuracy: 97.98%
    Epoch [7/10], Step [300/1875], Loss: 4.4334
    Epoch [7/10], Step [600/1875], Loss: 4.4359
    Epoch [7/10], Step [900/1875], Loss: 4.4402
    Epoch [7/10], Step [1200/1875], Loss: 4.4434
    Epoch [7/10], Step [1500/1875], Loss: 4.4337
    Epoch [7/10], Step [1800/1875], Loss: 4.4420
    Epoch [7/10] Training Accuracy: 98.23%
    Epoch [8/10], Step [300/1875], Loss: 4.4343
    Epoch [8/10], Step [600/1875], Loss: 4.4373
    Epoch [8/10], Step [900/1875], Loss: 4.4413
    Epoch [8/10], Step [1200/1875], Loss: 4.4293
    Epoch [8/10], Step [1500/1875], Loss: 4.4396
    Epoch [8/10], Step [1800/1875], Loss: 4.4322
    Epoch [8/10] Training Accuracy: 98.30%
    Epoch [9/10], Step [300/1875], Loss: 4.4249
    Epoch [9/10], Step [600/1875], Loss: 4.4357
    Epoch [9/10], Step [900/1875], Loss: 4.4309
    Epoch [9/10], Step [1200/1875], Loss: 4.4299
    Epoch [9/10], Step [1500/1875], Loss: 4.4265
    Epoch [9/10], Step [1800/1875], Loss: 4.4322
    Epoch [9/10] Training Accuracy: 98.48%
    Epoch [10/10], Step [300/1875], Loss: 4.4289
    Epoch [10/10], Step [600/1875], Loss: 4.4235
    Epoch [10/10], Step [900/1875], Loss: 4.4214
    Epoch [10/10], Step [1200/1875], Loss: 4.4244
    Epoch [10/10], Step [1500/1875], Loss: 4.4314
    Epoch [10/10], Step [1800/1875], Loss: 4.4284
    Epoch [10/10] Training Accuracy: 98.60%
    

### 3.9 모델 평가 (CNNWithAdaptivePooling3)


```python
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'테스트 정확도: {correct / total}')
```

    테스트 정확도: 0.9859
    

출력크기를 (7,7)로 했을 때 0.9869가 나왔고 (3,3)로 했을 때는 0.9859가 나왔다. 이게 무슨 일인가. (5,5)로 했을 때 나온 0.9879가 일단 가장 높은 값인 것 같다. 정리해보자면,   
1. 그냥 Convolution, flatten, FC, Activation 연산만 가지고 만들었을 때 -> 정확도 0.9798   
2. Average Pooling, Max Pooling 적용 -> 정확도 0.9873   
3. 두 개 풀링 빼고 Adaptive Pooling 적용    
  i) 출력크기 (5,5) -> 정확도 0.9879(max)   
 ii) 출력크기 (7,7) -> 정확도 0.9869   
iii) 출력크기 (3,3) -> 정확도 0.9859   

Adaptive Pooling에서 출력크기를 조정해봤는데도 0.99를 넘지 못했다. 이럴 땐 드롭아웃을 추가하거나 레이어를 더 쌓아야 되는걸까 아니면 Adaptive Pooling만 가지고 정확도를 높일 수 있는 걸까? 잘 모르겠다.. 일단 할 과제가 몇 개 더 남았으니 이건 이쯤에서 마무리 해보려고 한다.    
끝~
