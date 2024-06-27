---
layout: post
title: "240626(수) [온라인강의] Pytorch7_DNN구현1"
subtitle: "[Tips]"
date: 2024-06-26 22:10
background: 
tag: [Tips, Github io, Notion]
---

# [온라인강의]Pytorch7_DNN구현1

### **~ 목차 ~**
1. 데이터   
  1.1 torchvision 라이브러리를 사용해서 MNIST Dataset 불러오기   
  1.2 불러온 Dataset를 사용해서 DataLoader를 정의하고 DataLoader의 인자에 대한 이해
2. 모델   
  2.1 nn.Module을 사용해서 Custom model 정의   
  2.2 모델의 파라미터 초기화    
3. 전체 코드

# 0. 패키지 설치 및 임포트, 시드고정


```python
!pip install torch==2.0.1 -q
!pip install torchvision==0.15.2 -q
```

    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m619.9/619.9 MB[0m [31m2.1 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m21.0/21.0 MB[0m [31m56.8 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m849.3/849.3 kB[0m [31m48.4 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m11.8/11.8 MB[0m [31m73.8 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m557.1/557.1 MB[0m [31m2.2 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m317.1/317.1 MB[0m [31m2.0 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m168.4/168.4 MB[0m [31m7.5 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m54.6/54.6 MB[0m [31m10.9 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m102.6/102.6 MB[0m [31m9.0 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m173.2/173.2 MB[0m [31m3.4 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m177.1/177.1 MB[0m [31m6.7 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m98.6/98.6 kB[0m [31m11.0 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m63.3/63.3 MB[0m [31m9.9 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m96.4/96.4 kB[0m [31m11.3 MB/s[0m eta [36m0:00:00[0m
    [?25h[31mERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
    torchaudio 2.3.0+cu121 requires torch==2.3.0, but you have torch 2.0.1 which is incompatible.
    torchtext 0.18.0 requires torch>=2.3.0, but you have torch 2.0.1 which is incompatible.
    torchvision 0.18.0+cu121 requires torch==2.3.0, but you have torch 2.0.1 which is incompatible.[0m[31m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m6.0/6.0 MB[0m [31m12.0 MB/s[0m eta [36m0:00:00[0m
    [?25h


```python
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader # Optimizer 설정을 위한 라이브러리

import torchvision # PyTorch의 컴퓨터 비전 라이브러리
import torchvision.transforms as T # 이미지 변환을 위한 모듈
import torchvision.utils as vutils # 이미지를 쉽게 처리하기 위한 유틸리티 모듈
```

항상 일관된 실험결과를 얻기 위해 시드고정을 해줌


```python
# seed 고정
import random # 파이썬의 기본 난수생성기
import torch.backends.cudnn as cudnn # PyTorch의 CUDA 버전에서 딥러닝 모델의 성능을 향상키기위 위해 사용되는 CuDNN 라이브러리

def random_seed(seed_num):

  # PyTorch의 난수 생성기 고정
  torch.manual_seed(seed_num)

  # PyTorch의 CUDA연산을 위한 난수 생성기 고정
  torch.cuda.manual_seed(seed_num)

  # 모든 GPU에서 동일한 seed 값을 사용하도록 설정
  torch.cuda.manual_seed_all(seed_num)

  # CuDNN이 가장 빠른 알고리즘을 선택하기 위해 벤치마킹하는 것을 비활성화함. 이는 일관된 결과를 얻기 위해 필요함
  cudnn.benchmark = False

  # CuDNN의 결정론적 동작(동일한 작업을 반복할 때 항상 동일한 결과를 산출하는 것)을 활성화하여 동일한 입력에 대해 항상 동일한 결과를 출력하게 함
  cudnn.deterministic = True

  random.seed(seed_num)

random_seed(42)
```

## 1. 데이터

### 1.1 torchvision 라이브러리를 사용해서 MNIST Dataset 불러오기   



```python
# 데이터를 불러올 때 필요한 변환(transform)을 정의함
mnist_transform = T.Compose([
    T.ToTensor() # 텐서 형식으로 변환
])
```


```python
# torchvision 라이브러리를 사용해서 MNIST Dataset을 불러옴
download_root = './MNIST_DATASET'

train_dataset = torchvision.datasets.MNIST(download_root, transform=mnist_transform, train=True, download=True)
test_dataset = torchvision.datasets.MNIST(download_root, transform=mnist_transform, train=False, download=True)
```

    Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
    Failed to download (trying next):
    HTTP Error 403: Forbidden
    
    Downloading https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz
    Downloading https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz to ./MNIST_DATASET/MNIST/raw/train-images-idx3-ubyte.gz
    

    100%|██████████| 9912422/9912422 [00:01<00:00, 6042399.00it/s]
    

    Extracting ./MNIST_DATASET/MNIST/raw/train-images-idx3-ubyte.gz to ./MNIST_DATASET/MNIST/raw
    
    Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
    Failed to download (trying next):
    HTTP Error 403: Forbidden
    
    Downloading https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz
    Downloading https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz to ./MNIST_DATASET/MNIST/raw/train-labels-idx1-ubyte.gz
    

    100%|██████████| 28881/28881 [00:00<00:00, 157966.54it/s]
    

    Extracting ./MNIST_DATASET/MNIST/raw/train-labels-idx1-ubyte.gz to ./MNIST_DATASET/MNIST/raw
    
    Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
    Failed to download (trying next):
    HTTP Error 403: Forbidden
    
    Downloading https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz
    Downloading https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz to ./MNIST_DATASET/MNIST/raw/t10k-images-idx3-ubyte.gz
    

    100%|██████████| 1648877/1648877 [00:01<00:00, 1287927.30it/s]
    

    Extracting ./MNIST_DATASET/MNIST/raw/t10k-images-idx3-ubyte.gz to ./MNIST_DATASET/MNIST/raw
    
    Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
    Failed to download (trying next):
    HTTP Error 403: Forbidden
    
    Downloading https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz
    Downloading https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz to ./MNIST_DATASET/MNIST/raw/t10k-labels-idx1-ubyte.gz
    

    100%|██████████| 4542/4542 [00:00<00:00, 3148327.35it/s]

    Extracting ./MNIST_DATASET/MNIST/raw/t10k-labels-idx1-ubyte.gz to ./MNIST_DATASET/MNIST/raw
    
    

    
    


```python
for image, label in train_dataset:
  print(image.shape, label)
  break
```

    torch.Size([1, 28, 28]) 5
    


```python
# 데이터셋을 하습 데이터셋과 검증 데이터셋으로 분리함
total_size = len(train_dataset)
train_num, valid_num = int(total_size * 0.8), int(total_size * 0.2)
print('Train dataset 개수: ', train_num)
print('Test dataset 개수: ', valid_num)
train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [train_num, valid_num])
```

    Train dataset 개수:  48000
    Test dataset 개수:  12000
    

###1.2 불러온 Dataset를 사용해서 DataLoader를 정의하고 DataLoader의 인자에 대한 이해

- DataLoader는 인자로 주어진 Dataset을 이용해서 단일 데이터들을 정해진 개수만큼 모아 미니배치(mini-batch)를 구성하는 역할을 함. `torch.utils.data`라이브러리를 씀


```python
batch_size = 32

# 앞서 선언한 Dataset을 인자로 주어 DataLoader를 선언함
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # shuffle=True: 모델이 데이터를 학습할 때 에포크마다 데이터가 섞여야 일반화된 학습이 가능함
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False) # shuffle=False: 일관된 순서로 제공되어야 메 에포크마다 동일한 조건에서 성능을 평가할 수 있음
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) # shuffle=False: 순서가 일정해야 비교 가능한 결과를 얻을 수 있음
```

- 앞에서는 image.shape가 [1,28,28]였는데 지금은 [32,1,28,28]임. 이는 32개의 손글씨 데이터가 하나의 배치로 묶여있다는 뜻임


```python
# 학습 데이터로더에서 일부의 미니배치데이터를 가져옴
for images, labels in train_dataloader:
  print(images.shape, labels.shape)
  break
```

    torch.Size([32, 1, 28, 28]) torch.Size([32])
    


```python
grid = vutils.make_grid(images, nrow=8) # 각 행마다 8개의 미지를 배치해서 격자로 구성함

# 학습 데이터로더로부터 불러온 이미지를 시각화함
plt.figure(figsize=(12,12))
plt.imshow(grid.numpy().transpose(1,2,0))
plt.title('mini batch visualization')
plt.axis('off')
plt.show()
```


    
![png](240626_Pytorch7_DNN%EA%B5%AC%ED%98%841_files/240626_Pytorch7_DNN%EA%B5%AC%ED%98%841_18_0.png)
    


## 2. 모델

###2.1 nn.Module을 사용해서 Custom model 정의   

- CNN을 사용하기 전에 이번에는 FC레이어로만 구성된 DNN(Depp Neural Network)을 먼저 구현해볼거임

```
class DNN(nn.Module):
  def __init__(self, hidden_dims, num_classes, dropout_ratio, apply_batchnorm, apply_dropout, apply_Activation, set_super):
    if set_super:
      super().__init__()
    # FC 레이어를 선언함
    self.fc1 = nn.Linear(28*28, hidden_dim*4)
    self.fc2 = nn.Linear(hidden_dim*4, hidden_dim*2)
    self.fc3 = nn.Linear(hidden_dim*2, hidden_dim)
    self.classifier = nn.Linear(hidden_dim, num_classes)

    # Batch normalization을 선언함. apply_batchnorm 인자가 False일 경우, batch normalization은 적용되지 않음
    # nn.Identity(): 아무런 변환도 수행하지 않는 레이어임. 입력을 그대로 출력
    self.batchnorm1 = nn.BatchNorm1d(hidden_dim*4) if apply_batchnorm else nn.Identity()
    self.batchnorm2 = nn.BatchNorm1d(hidden_dim*2) if apply_batchnorm else nn.Identity()
    self.batchnorm3 = nn.BatchNorm1d(hidden_dim) if apply_batchnorm else nn.Identity()

    # Dropout은 레이어를 통과한 중간 연산 결과를 dropout_ratio만큼의 비율로 element를 0으로 변경함
    # Dropout을 선언함. apply_dropout 인자가 False일 경우 dropout은 적용되지 않음
    self.dropout1 = nn.Dropout(dropout_ratio) if apply_dropout else nn.Identity()
    self.dropout2 = nn.Dropout(dropout_ratio) if apply_dropout else nn.Identity()
    self.dropout3 = nn.Dropout(dropout_ratio) if apply_dropout else nn.Identity()

    # Activation function을 선언함. apply_activation 인자가 False일 경우 activation function은 적용되지 않음
    self.activation1 = nn.ReLU() if apply_activation else nn.Identity()
    self.activation2 = nn.ReLU() if apply_activation else nn.Identity()
    self.activation3 = nn.ReLU() if apply_activation else nn.Identity()

    self.softmax = nn.LogSoftmax(dim=1) # LogSoftmax: 그냥 softmax 함수보다 수치적으로 안정적 결과를 얻을 수 있음

  def forward(self,x):
    '''
    Input:
      x : [batch_size, 1, 28, 28]
    Output:
      output: [batch_size, num_classes]
    '''

    x = x.view(x.shape[0],-1)

    x = self.fc1(x) # [batch_size, dim*4]
    x = self.batchnorm1(x)
    x = self.activation1(x)
    x = self.dropout1(x)

    x = self.fc2(x) # [batch_size, dim*2]
    x = self.batchnorm2(x)
    x = self.activation2(x)
    x = self.dropout2(x)

    x = self.fc3(x) # [batch_size, dim]
    x = self.batchnorm3(x)
    x = self.activation3(x)
    x = self.dropout3(x)

    x = self.classifier(x) # [batch_size, 10]
    output = self.softmax(x)
    return output
```

아래는 위의 코드를 조금 더 효율적으로 적은 버전임


```python
class DNN(nn.Module):
  def __init__(self, hidden_dims, num_classes, dropout_ratio, apply_batchnorm, apply_dropout, apply_activation, set_super):
    if set_super:
      super().__init__()

    self.hidden_dims = hidden_dims
    self.layers = nn.ModuleList()

    for i in range(len(self.hidden_dims)-1):
      self.layers.append(nn.Linear(self.hidden_dims[i], self.hidden_dims[i+1]))

      if apply_batchnorm:
        self.layers.append(nn.BatchNorm1d(self.hidden_dims[i+1]))

      if apply_activation:
        self.layers.append(nn.ReLU())

      if apply_dropout:
        self.layers.append(nn.Dropout(dropout_ratio))

    self.classifier = nn.Linear(self.hidden_dims[-1], num_classes)
    self.softmax = nn.LogSoftmax(dim=1)

  def forward(self,x):
    '''
    Input:
      x : [batch_size, 1, 28, 28]
    Output:
      output: [batch_size, num_classes]
    '''

    x = x.view(x.shape[0],-1)

    for layer in self.layers:
      x = layer(x)

    x = self.classifier(x)
    output = self.softmax(x)
    return output
```


```python
# 모델을 선언한 후 MNIST 숫자 이미지 데이터오 동일한 크기의 random Tensor를 입력으로넣어 연산상 문제가 없는지 확인함
hidden_dim = 128
hidden_dims = [784, hidden_dim * 4, hidden_dim * 2, hidden_dim]
model = DNN(hidden_dims, num_classes=10, dropout_ratio=0.2, apply_batchnorm=True, apply_dropout=True, apply_activation=True, set_super=True)
output = model(torch.randn((32,1,28,28)))
```

- 만약에 nn.Module을 먼저 초기화하지 않는다면 오류가 발생함


```python
model = DNN(hidden_dims, num_classes=10, dropout_ratio=0.2, apply_batchnorm=True, apply_dropout=True, apply_activation=True, set_super=False)
```


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    <ipython-input-20-ac88750ee4ad> in <cell line: 1>()
    ----> 1 model = DNN(hidden_dims, num_classes=10, dropout_ratio=0.2, apply_batchnorm=True, apply_dropout=True, apply_activation=True, set_super=False)
    

    <ipython-input-17-c90092a68599> in __init__(self, hidden_dim, num_classes, dropout_ratio, apply_batchnorm, apply_dropout, apply_activation, set_super)
          5 
          6     self.hidden_dims = hidden_dims
    ----> 7     self.layers = nn.ModuleList()
          8 
          9     for i in range(len(self.hidden_dims)-1):
    

    /usr/local/lib/python3.10/dist-packages/torch/nn/modules/module.py in __setattr__(self, name, value)
       1641             if isinstance(value, Module):
       1642                 if modules is None:
    -> 1643                     raise AttributeError(
       1644                         "cannot assign module before Module.__init__() call")
       1645                 remove_from(self.__dict__, self._parameters, self._buffers, self._non_persistent_buffers_set)
    

    AttributeError: cannot assign module before Module.__init__() call


###2.2 모델의 파라미터 초기화


```python
# 모델의 가중치를 초기화하는함수
def weight_initialization(model, weight_init_method):
  for m in model.modules():
    if isinstance(m, nn.Linear):
      if weight_init_method == 'gaussian':
        nn.init.normal_(m.weight)
      elif weight_init_method == 'xavier':
        nn.init.xavier_normal_(m.weight)
      elif weight_init_method == 'kaiming':
        nn.init.kaiming_normal_(m.weight)
      elif weight_init_method == 'zeros':
        nn.init.zeros_(m.weight)

      nn.init.zeros_(m.bias)

  return model
```


```python
init_method = 'zeros' # gaussian, xavier, kaiming, zeros
model = weight_initialization(model, init_method)

for m in model.modules():
  if isinstance(m, nn.Linear):
    print(m.weight.data)
    break
```

    tensor([[0., 0., 0.,  ..., 0., 0., 0.],
            [0., 0., 0.,  ..., 0., 0., 0.],
            [0., 0., 0.,  ..., 0., 0., 0.],
            ...,
            [0., 0., 0.,  ..., 0., 0., 0.],
            [0., 0., 0.,  ..., 0., 0., 0.],
            [0., 0., 0.,  ..., 0., 0., 0.]])
    

## 3. 전체 코드

- 결론적으로 전체 코드는 이렇게 됨


```python
class DNN(nn.Module):
  def __init__(self, hidden_dims, num_classes, dropout_ratio, apply_batchnorm, apply_dropout, apply_activation, set_super):
    if set_super:
      super().__init__()

    self.hidden_dims = hidden_dims
    self.layers = nn.ModuleList()

    for i in range(len(self.hidden_dims)-1):
      self.layers.append(nn.Linear(self.hidden_dims[i], self.hidden_dims[i+1]))

      if apply_batchnorm:
        self.layers.append(nn.BatchNorm1d(self.hidden_dims[i+1]))

      if apply_activation:
        self.layers.append(nn.ReLU())

      if apply_dropout:
        self.layers.append(nn.Dropout(dropout_ratio))

    self.classifier = nn.Linear(self.hidden_dims[-1], num_classes)
    self.softmax = nn.LogSoftmax(dim=1)

  def forward(self,x):
    '''
    Input:
      x : [batch_size, 1, 28, 28]
    Output:
      output: [batch_size, num_classes]
    '''

    x = x.view(x.shape[0],-1)

    for layer in self.layers:
      x = layer(x)

    x = self.classifier(x)
    output = self.softmax(x)
    return output

  def weight_initialization(self, weight_init_method):
    for m in model.modules():
      if isinstance(m, nn.Linear):
        if weight_init_method == 'gaussian':
          nn.init.normal_(m.weight)
        elif weight_init_method == 'xavier':
          nn.init.xavier_normal_(m.weight)
        elif weight_init_method == 'kaiming':
          nn.init.kaiming_normal_(m.weight)
        elif weight_init_method == 'zeros':
          nn.init.zeros_(m.weight)

        nn.init.zeros_(m.bias)

  def count_parameters(self):
    return sum(p.numel() for p in self.parameters() if p.requires_grad) # numel(): 텐서의 원소 개수를 반환하는 함수
```


```python
model = DNN(hidden_dims, num_classes=10, dropout_ratio=0.2, apply_batchnorm=True, apply_dropout=True, apply_activation=True, set_super=True)
init_method = 'gaussian'
model.weight_initialization(init_method)
```


```python
print(f'The model has {model.count_parameters():,} trainable parameters')
```

    The model has 569,226 trainable parameters
    
