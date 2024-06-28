---
layout: post
title: "240626(수) [온라인강의] Pytorch8_DNN구현1"
subtitle: "[Tips]"
date: 2024-06-26 22:13
background: 
tag: [Tips, Github io, Notion]
---

# [온라인강의]Pytorch8_DNN구현2

### **~ 목차 ~**
1. 학습(training)   
  1.1 손실함수와 최적화 알고리즘   
  1.2 학습 과정   
  1.3 활성화 함수와 가중치 초기화의 중요성
2. 추론과 평가(inference & evaluation)   
  2.1 학습한 딥러닝 모델로 테스트 데이터 추론   
  2.2 학습한 딥러닝 모델의 평가

## 0. 패키지 설치 및 임포트


```python
!pip install scikit-learn==1.3.0 -q
!pip install torch==2.0.1 -q
!pip install torchvision==0.15.2 -q
```

    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m10.8/10.8 MB[0m [31m12.0 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m619.9/619.9 MB[0m [31m1.6 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m21.0/21.0 MB[0m [31m32.6 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m849.3/849.3 kB[0m [31m14.8 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m11.8/11.8 MB[0m [31m46.5 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m557.1/557.1 MB[0m [31m1.5 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m317.1/317.1 MB[0m [31m3.2 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m168.4/168.4 MB[0m [31m6.6 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m54.6/54.6 MB[0m [31m10.5 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m102.6/102.6 MB[0m [31m7.7 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m173.2/173.2 MB[0m [31m6.7 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m177.1/177.1 MB[0m [31m6.5 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m98.6/98.6 kB[0m [31m8.8 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m63.3/63.3 MB[0m [31m9.8 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m96.4/96.4 kB[0m [31m7.9 MB/s[0m eta [36m0:00:00[0m
    [?25h[31mERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
    torchaudio 2.3.0+cu121 requires torch==2.3.0, but you have torch 2.0.1 which is incompatible.
    torchtext 0.18.0 requires torch>=2.3.0, but you have torch 2.0.1 which is incompatible.
    torchvision 0.18.0+cu121 requires torch==2.3.0, but you have torch 2.0.1 which is incompatible.[0m[31m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m6.0/6.0 MB[0m [31m32.1 MB/s[0m eta [36m0:00:00[0m
    [?25h


```python
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.transforms as T

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score # 성능지표 측정
```


```python
# seed 고정
import random
import torch.backends.cudnn as cudnn

def random_seed(seed_num):
  torch.manual_seed(seed_num)
  torch.cuda.manual_seed(seed_num)
  torch.cuda.manual_seed_all(seed_num)
  np.random.seed(seed_num)
  cudnn.benchmark = False
  cudnn.deterministic = True
  random.seed(seed_num)

random_seed(42)
```


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
    


```python
batch_size = 32

# 앞서 선언한 Dataset을 인자로 주어 DataLoader를 선언함
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
```


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

##1. 학습(training)   

###1.1 손실함수와 최적화 알고리즘   

- `torch.nn` 모듈은 다양한 손실함수들을 제공함
  - `torch.nn.NLLLoss`
  - `torch.nn.MSELoss`
  - `torch.nn.L1Loss`
  - `torch.nn.BCELoss`
  - `torch.nn.CrossEntropyLoss`


```python
# log softmax와 NLLLoss가 합쳐진 것이 CE loss이므로 여기서는 NLLLoss를 사용함
criterion = nn.NLLLoss()
```

- `torch.optim`에는 다양한 최적화 알고리즘이 구현되어 있음
  - `torch.optim.SGD`
  - `torch.optim.Adam`
  - `torch.optim.Adagrad`
  - `torch.optim.FMSoroo`


```python
# 여기서는 Adam optimizer를 사용할거임
lr = 0.001
hidden_dim = 128
hidden_dims = [784, hidden_dim*4, hidden_dim*2, hidden_dim]
model = DNN(hidden_dims=hidden_dims, num_classes=10, dropout_ratio=0.2, apply_batchnorm=True, apply_dropout=True, apply_activation=True, set_super=True)
optimizer = optim.Adam(model.parameters(), lr=lr)
```

###1.2 학습 과정   

- 학습을 진행하면서 검증 데이터셋에 대해 loss가 감소하지 않고 `patience`만큼 계속 증가한다면 학습을 중단함


```python
def training(model, dataloader, train_dataset, criterion, optimizer, device, epoch, num_epochs):
  model.train() # 모델을 학습모드로 설정
  train_loss = 0.0
  train_accuracy =0

  tbar = tqdm(dataloader)
  for images, labels in tbar:
    images = images.to(device)
    labels = labels.to(device)

    # 순전파
    outputs = model(images)
    loss = criterion(outputs, labels)

    # 역전파 및 weights 업데이트
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 손실과 정확도 계산
    train_loss += loss.item()
    # torch.max에서 dim 인자에 값을 추가할 경우 해당 dimension에서 최댓값과 최댓값에 해당하는 인덱스 반환
    _, predicted = torch.max(outputs,1)
    train_accuracy += (predicted == labels).sum().item()

    # tbar의 진행바에 표시될 설명 텍스트 설정
    tbar.set_description(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item(): .4f}')

  # 에폭별 학습 결과 출력
  train_loss = train_loss / len(dataloader)
  train_accuracy = train_accuracy / len(train_dataset)

  return model, train_loss, train_accuracy
```


```python
def evaluation(model, dataloader, valid_dataset, criterion, device, epoch, num_epochs):
  model.eval() # 모델을 평가모드로 설정
  valid_loss = 0.0
  valid_accuracy = 0

  with torch.no_grad(): # 모델의 업데이트 막기
    tbar = tqdm(dataloader)
    for images, labels in tbar:
      images = images.to(device)
      labels = labels.to(device)

      # 순전파
      outputs = model(images)
      loss = criterion(outputs, labels)

      # 손실과 정확도 계산
      valid_loss += loss.item()
      # torch.max에서 dim인자에 값을 추가할 경우 해당 dimension에서 최댓값과 최댓값에 해당하는 인덱스를 반환
      _, predicted = torch.max(outputs,1)
      valid_accuracy += (predicted == labels).sum().item()

      # tqdm의 진행바에 표시될 설명 텍스트를 설정
      tbar.set_description(f'Epoch [{epoch+1}/{num_epochs}], Valid Loss: {loss.item():.4f}')

  valid_loss = valid_loss / len(dataloader)
  valid_accuracy = valid_accuracy / len(valid_dataset)

  return model, valid_loss, valid_accuracy
```


```python
def training_loop(model, train_dataloader, valid_dataloader, criterion, optimizer, device, num_epochs, patience, model_name):
  best_valid_loss = float('inf') # 가장 좋은 validation loss를 저장
  early_stop_counter = 0
  valid_max_accuracy = -1

  for epoch in range(num_epochs):
    model, train_loss, train_accuracy = training(model, train_dataloader, train_dataset, criterion, optimizer, device, epoch, num_epochs)
    model, valid_loss, valid_accuracy = evaluation(model, valid_dataloader, valid_dataset, criterion, device, epoch, num_epochs)

    if valid_accuracy > valid_max_accuracy:
      valid_max_accuracy = valid_accuracy

    # validation loss가 감소하면 모델 저장 및 카운터리셋
    if valid_loss < best_valid_loss:
      best_valid_loss = valid_loss
      torch.save(model.state_dict(), f'./model_{model_name}.pt')
      early_stop_counter = 0

    # validation loss가 증가하거나 같으면 카운터 증가
    else:
      early_stop_counter += 1

    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Valid Loss: {valid_loss:.4f}, Valid Accuracy: {valid_accuracy:.4f}')

    # 조기 종료 카운터가 설정한 patience를 초과하면 학습 종료
    if early_stop_counter >= patience:
      print('Early stopping')
      break

  return model, valid_max_accuracy
```


```python
num_epochs = 100
patience = 3
scores = dict()
device = 'cpu' # gpu나 cpu 설정
model_name = 'exp1'
init_method = 'kaiming'

model = DNN(hidden_dims=hidden_dims, num_classes=10, dropout_ratio=0.2, apply_batchnorm=True, apply_dropout=True, apply_activation=True, set_super=True)
model.weight_initialization(init_method)
model = model.to(device)

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

model, valid_max_accuracy = training_loop(model, train_dataloader, valid_dataloader, criterion, optimizer, device, num_epochs, patience, model_name)
scores[model_name] = valid_max_accuracy
```


      0%|          | 0/1500 [00:00<?, ?it/s]



      0%|          | 0/375 [00:00<?, ?it/s]


    Epoch [1/100], Train Loss: 0.3170, Train Accuracy: 0.9061, Valid Loss: 0.1206, Valid Accuracy: 0.9623
    


      0%|          | 0/1500 [00:00<?, ?it/s]



      0%|          | 0/375 [00:00<?, ?it/s]


    Epoch [2/100], Train Loss: 0.1676, Train Accuracy: 0.9479, Valid Loss: 0.0994, Valid Accuracy: 0.9686
    


      0%|          | 0/1500 [00:00<?, ?it/s]



      0%|          | 0/375 [00:00<?, ?it/s]


    Epoch [3/100], Train Loss: 0.1288, Train Accuracy: 0.9599, Valid Loss: 0.0837, Valid Accuracy: 0.9732
    


      0%|          | 0/1500 [00:00<?, ?it/s]



      0%|          | 0/375 [00:00<?, ?it/s]


    Epoch [4/100], Train Loss: 0.1116, Train Accuracy: 0.9659, Valid Loss: 0.0820, Valid Accuracy: 0.9749
    


      0%|          | 0/1500 [00:00<?, ?it/s]



      0%|          | 0/375 [00:00<?, ?it/s]


    Epoch [5/100], Train Loss: 0.0948, Train Accuracy: 0.9701, Valid Loss: 0.0780, Valid Accuracy: 0.9765
    


      0%|          | 0/1500 [00:00<?, ?it/s]



      0%|          | 0/375 [00:00<?, ?it/s]


    Epoch [6/100], Train Loss: 0.0828, Train Accuracy: 0.9734, Valid Loss: 0.0775, Valid Accuracy: 0.9759
    


      0%|          | 0/1500 [00:00<?, ?it/s]



      0%|          | 0/375 [00:00<?, ?it/s]


    Epoch [7/100], Train Loss: 0.0749, Train Accuracy: 0.9764, Valid Loss: 0.0669, Valid Accuracy: 0.9795
    


      0%|          | 0/1500 [00:00<?, ?it/s]



      0%|          | 0/375 [00:00<?, ?it/s]


    Epoch [8/100], Train Loss: 0.0660, Train Accuracy: 0.9791, Valid Loss: 0.0681, Valid Accuracy: 0.9802
    


      0%|          | 0/1500 [00:00<?, ?it/s]



      0%|          | 0/375 [00:00<?, ?it/s]


    Epoch [9/100], Train Loss: 0.0621, Train Accuracy: 0.9806, Valid Loss: 0.0745, Valid Accuracy: 0.9786
    


      0%|          | 0/1500 [00:00<?, ?it/s]



      0%|          | 0/375 [00:00<?, ?it/s]


    Epoch [10/100], Train Loss: 0.0603, Train Accuracy: 0.9801, Valid Loss: 0.0685, Valid Accuracy: 0.9800
    Early stopping
    

- Batch normalization을 제외하고 학습을 진행해보자


```python
model_name = 'exp2'
init_method = 'kaiming'

model = DNN(hidden_dims=hidden_dims, num_classes=10, dropout_ratio=0.2, apply_batchnorm=False, apply_dropout=True, apply_activation=True, set_super=True)
model.weight_initialization(init_method)
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=lr)
model, valid_max_accuracy = training_loop(model, train_dataloader, valid_dataloader, criterion, optimizer, device, num_epochs, patience, model_name)
scores[model_name] = valid_max_accuracy
```


      0%|          | 0/1500 [00:00<?, ?it/s]



      0%|          | 0/375 [00:00<?, ?it/s]


    Epoch [1/100], Train Loss: 0.2857, Train Accuracy: 0.9133, Valid Loss: 0.1311, Valid Accuracy: 0.9597
    


      0%|          | 0/1500 [00:00<?, ?it/s]



      0%|          | 0/375 [00:00<?, ?it/s]


    Epoch [2/100], Train Loss: 0.1371, Train Accuracy: 0.9597, Valid Loss: 0.1153, Valid Accuracy: 0.9653
    


      0%|          | 0/1500 [00:00<?, ?it/s]



      0%|          | 0/375 [00:00<?, ?it/s]


    Epoch [3/100], Train Loss: 0.1017, Train Accuracy: 0.9699, Valid Loss: 0.1008, Valid Accuracy: 0.9713
    


      0%|          | 0/1500 [00:00<?, ?it/s]



      0%|          | 0/375 [00:00<?, ?it/s]


    Epoch [4/100], Train Loss: 0.0892, Train Accuracy: 0.9736, Valid Loss: 0.0843, Valid Accuracy: 0.9768
    


      0%|          | 0/1500 [00:00<?, ?it/s]



      0%|          | 0/375 [00:00<?, ?it/s]


    Epoch [5/100], Train Loss: 0.0735, Train Accuracy: 0.9776, Valid Loss: 0.0929, Valid Accuracy: 0.9741
    


      0%|          | 0/1500 [00:00<?, ?it/s]



      0%|          | 0/375 [00:00<?, ?it/s]


    Epoch [6/100], Train Loss: 0.0640, Train Accuracy: 0.9809, Valid Loss: 0.0904, Valid Accuracy: 0.9764
    


      0%|          | 0/1500 [00:00<?, ?it/s]



      0%|          | 0/375 [00:00<?, ?it/s]


    Epoch [7/100], Train Loss: 0.0601, Train Accuracy: 0.9820, Valid Loss: 0.0947, Valid Accuracy: 0.9753
    Early stopping
    

- Batch normaliztion은 사용하고 dropout을 제외하고 학습을 진행시켜보자


```python
model_name = 'exp3'
init_method = 'kaiming'

model = DNN(hidden_dims=hidden_dims, num_classes=10, dropout_ratio=0.2, apply_batchnorm=True, apply_dropout=False, apply_activation=True, set_super=True)
model.weight_initialization(init_method)
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=lr)
model, valid_max_accuracy = training_loop(model, train_dataloader, valid_dataloader, criterion, optimizer, device, num_epochs, patience, model_name)
scores[model_name] = valid_max_accuracy
```


      0%|          | 0/1500 [00:00<?, ?it/s]



      0%|          | 0/375 [00:00<?, ?it/s]


    Epoch [1/100], Train Loss: 0.2190, Train Accuracy: 0.9344, Valid Loss: 0.1083, Valid Accuracy: 0.9681
    


      0%|          | 0/1500 [00:00<?, ?it/s]



      0%|          | 0/375 [00:00<?, ?it/s]


    Epoch [2/100], Train Loss: 0.1088, Train Accuracy: 0.9651, Valid Loss: 0.0905, Valid Accuracy: 0.9719
    


      0%|          | 0/1500 [00:00<?, ?it/s]



      0%|          | 0/375 [00:00<?, ?it/s]


    Epoch [3/100], Train Loss: 0.0795, Train Accuracy: 0.9748, Valid Loss: 0.0865, Valid Accuracy: 0.9730
    


      0%|          | 0/1500 [00:00<?, ?it/s]



      0%|          | 0/375 [00:00<?, ?it/s]


    Epoch [4/100], Train Loss: 0.0645, Train Accuracy: 0.9794, Valid Loss: 0.0763, Valid Accuracy: 0.9771
    


      0%|          | 0/1500 [00:00<?, ?it/s]



      0%|          | 0/375 [00:00<?, ?it/s]


    Epoch [5/100], Train Loss: 0.0532, Train Accuracy: 0.9823, Valid Loss: 0.0695, Valid Accuracy: 0.9798
    


      0%|          | 0/1500 [00:00<?, ?it/s]



      0%|          | 0/375 [00:00<?, ?it/s]


    Epoch [6/100], Train Loss: 0.0435, Train Accuracy: 0.9857, Valid Loss: 0.0648, Valid Accuracy: 0.9808
    


      0%|          | 0/1500 [00:00<?, ?it/s]



      0%|          | 0/375 [00:00<?, ?it/s]


    Epoch [7/100], Train Loss: 0.0421, Train Accuracy: 0.9855, Valid Loss: 0.0719, Valid Accuracy: 0.9784
    


      0%|          | 0/1500 [00:00<?, ?it/s]



      0%|          | 0/375 [00:00<?, ?it/s]


    Epoch [8/100], Train Loss: 0.0325, Train Accuracy: 0.9891, Valid Loss: 0.0689, Valid Accuracy: 0.9810
    


      0%|          | 0/1500 [00:00<?, ?it/s]



      0%|          | 0/375 [00:00<?, ?it/s]


    Epoch [9/100], Train Loss: 0.0304, Train Accuracy: 0.9897, Valid Loss: 0.0696, Valid Accuracy: 0.9802
    Early stopping
    

- 활성화함수(activation fuction)을 제외하고 학습을 진행시켜보자


```python
model_name = 'exp4'
init_method = 'kaiming'

model = DNN(hidden_dims=hidden_dims, num_classes=10, dropout_ratio=0.2, apply_batchnorm=True, apply_dropout=True, apply_activation=False, set_super=True)
model.weight_initialization(init_method)
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=lr)
model, valid_max_accuracy = training_loop(model, train_dataloader, valid_dataloader, criterion, optimizer, device, num_epochs, patience, model_name)
scores[model_name] = valid_max_accuracy
```


      0%|          | 0/1500 [00:00<?, ?it/s]



      0%|          | 0/375 [00:00<?, ?it/s]


    Epoch [1/100], Train Loss: 0.4571, Train Accuracy: 0.8630, Valid Loss: 0.3553, Valid Accuracy: 0.9039
    


      0%|          | 0/1500 [00:00<?, ?it/s]



      0%|          | 0/375 [00:00<?, ?it/s]


    Epoch [2/100], Train Loss: 0.3745, Train Accuracy: 0.8904, Valid Loss: 0.3428, Valid Accuracy: 0.9028
    


      0%|          | 0/1500 [00:00<?, ?it/s]



      0%|          | 0/375 [00:00<?, ?it/s]


    Epoch [3/100], Train Loss: 0.3631, Train Accuracy: 0.8935, Valid Loss: 0.3298, Valid Accuracy: 0.9054
    


      0%|          | 0/1500 [00:00<?, ?it/s]



      0%|          | 0/375 [00:00<?, ?it/s]


    Epoch [4/100], Train Loss: 0.3488, Train Accuracy: 0.8967, Valid Loss: 0.3271, Valid Accuracy: 0.9075
    


      0%|          | 0/1500 [00:00<?, ?it/s]



      0%|          | 0/375 [00:00<?, ?it/s]


    Epoch [5/100], Train Loss: 0.3412, Train Accuracy: 0.9008, Valid Loss: 0.3162, Valid Accuracy: 0.9123
    


      0%|          | 0/1500 [00:00<?, ?it/s]



      0%|          | 0/375 [00:00<?, ?it/s]


    Epoch [6/100], Train Loss: 0.3359, Train Accuracy: 0.9033, Valid Loss: 0.3133, Valid Accuracy: 0.9131
    


      0%|          | 0/1500 [00:00<?, ?it/s]



      0%|          | 0/375 [00:00<?, ?it/s]


    Epoch [7/100], Train Loss: 0.3274, Train Accuracy: 0.9044, Valid Loss: 0.3185, Valid Accuracy: 0.9101
    


      0%|          | 0/1500 [00:00<?, ?it/s]



      0%|          | 0/375 [00:00<?, ?it/s]


    Epoch [8/100], Train Loss: 0.3283, Train Accuracy: 0.9053, Valid Loss: 0.3145, Valid Accuracy: 0.9118
    


      0%|          | 0/1500 [00:00<?, ?it/s]



      0%|          | 0/375 [00:00<?, ?it/s]


    Epoch [9/100], Train Loss: 0.3264, Train Accuracy: 0.9059, Valid Loss: 0.3190, Valid Accuracy: 0.9102
    Early stopping
    

- 이번에는 다 True로 두되 init_method를 zero로 두고 학습을 진행시켜보자


```python
model_name = 'exp5'
init_method = 'zeros'

model = DNN(hidden_dims=hidden_dims, num_classes=10, dropout_ratio=0.2, apply_batchnorm=True, apply_dropout=True, apply_activation=True, set_super=True)
model.weight_initialization(init_method)
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=lr)
model, valid_max_accuracy = training_loop(model, train_dataloader, valid_dataloader, criterion, optimizer, device, num_epochs, patience, model_name)
scores[model_name] = valid_max_accuracy
```


      0%|          | 0/1500 [00:00<?, ?it/s]



      0%|          | 0/375 [00:00<?, ?it/s]


    Epoch [1/100], Train Loss: 2.3017, Train Accuracy: 0.1114, Valid Loss: 2.3008, Valid Accuracy: 0.1133
    


      0%|          | 0/1500 [00:00<?, ?it/s]



      0%|          | 0/375 [00:00<?, ?it/s]


    Epoch [2/100], Train Loss: 2.3015, Train Accuracy: 0.1121, Valid Loss: 2.3007, Valid Accuracy: 0.1133
    


      0%|          | 0/1500 [00:00<?, ?it/s]



      0%|          | 0/375 [00:00<?, ?it/s]


    Epoch [3/100], Train Loss: 2.3015, Train Accuracy: 0.1121, Valid Loss: 2.3008, Valid Accuracy: 0.1133
    


      0%|          | 0/1500 [00:00<?, ?it/s]



      0%|          | 0/375 [00:00<?, ?it/s]


    Epoch [4/100], Train Loss: 2.3015, Train Accuracy: 0.1121, Valid Loss: 2.3008, Valid Accuracy: 0.1133
    


      0%|          | 0/1500 [00:00<?, ?it/s]



      0%|          | 0/375 [00:00<?, ?it/s]


    Epoch [5/100], Train Loss: 2.3015, Train Accuracy: 0.1121, Valid Loss: 2.3008, Valid Accuracy: 0.1133
    Early stopping
    

###1.3 활성화 함수와 가중치 초기화의 중요성

![image.png](data:image/png;base64,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)

1.3.1 Activation function을 False로 설정했을 때 정확도가 낮게 나오는 이유
- 활성화함수가 없는 딥러닝 모데른 기본적으로 선형분류기와 다르지 않으며 복잡한 패턴을 학습하는데에 제한적일 수 있음. 활성화함수로 인해 딥러닝 모델은 선형 변환 이상의 복잡한 함수를 표현하고 학습할 수 있게 됨

1.3.2 Zero weight initialization을 사용했을 때 정확도가 매우 낮게 나오는 이유
- 딥러닝 모델의 가중치를 0으로 초기화하면 모든 뉴런들의출력값은 동일하게 0으로 계산되고, 이 경우 역전파 단계에서 chain rule이 계산되는 값에 0이 곱해지므로 gradeint가 0으로 계산되어 학습이 일어나지 않게 됨 (bias만큼 아주 조금만 학습됨)

##2. 추론과 평가(inference & evaluation)   


```python
# BatchNorm, Dropout을 사용한 모델을 로드함
model = DNN(hidden_dims=hidden_dims, num_classes=10, dropout_ratio=0.2, apply_batchnorm=True, apply_dropout=True, apply_activation=True, set_super=True)
model.load_state_dict(torch.load('./model_exp1.pt'))
model = model.to(device)
```

###2.1 학습한 딥러닝 모델로 테스트 데이터 추론   


```python
model.eval()
total_labels = []
total_preds = []
total_probs = []
with torch.no_grad():
  for images, labels in test_dataloader:
    images = images.to(device)
    labels = labels

    outputs = model(images)
    # torch.max에서 dim인자에 값을 추가할 경우 해당 dimension에서 최댓값과 최댓값에 해당하는 인덱스를 반환
    _, predicted = torch.max(outputs.data,1)

    total_preds.extend(predicted.detach().cpu().tolist())
    total_labels.extend(labels.tolist())
    total_probs.append(outputs.detach().cpu().numpy())

total_preds = np.array(total_preds)
total_labels = np.array(total_labels)
total_probs = np.concatenate(total_probs, axis=0)
```

###2.2 학습한 딥러닝 모델의 평가


```python
# precision, recall, f1을 계산함
precision = precision_score(total_labels, total_preds, average='macro')
recall = recall_score(total_labels, total_preds, average='macro')
f1 = f1_score(total_labels, total_preds, average='macro')

# AUC를 계산함
# 모델의 출력으로 nn.LogSoftmax 함수가 적용되어 결과물이 출력됨. sklearn의 roc_auc_score 메서드를 사용하기 위해서는 단일 데이터의 클래스들의 확률합은 1이 되어야 하므로 이를 위해 확률 matrix에 지수함수를 적용함
total_probs = np.exp(total_probs)
auc = roc_auc_score(total_labels, total_probs, average='macro', multi_class='ovr')

print(f'Precision: {precision}, Recall: {recall}, F1 Score: {f1}, AUC: {auc}')
```

    Precision: 0.9795304454159091, Recall: 0.9792397602498941, F1 Score: 0.9793610126010959, AUC: 0.9995917139930912
    
