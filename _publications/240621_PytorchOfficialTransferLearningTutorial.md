---
layout: post
title: "240621(금) [과제] 난이도(하)-2 PytorchOfficialTransferLearningTutorial"
subtitle: "[Tips]"
date: 2024-06-21 17:37
background: 
tag: [Tips, Github io, Notion]
---

# PytorchOfficialTransferLearningTutorial

[참고한 링크](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)

## 0. 라이브러리 임포트


```python
# License: BSD
# Author: Sasank Chilamkurthy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
from PIL import Image
from tempfile import TemporaryDirectory

cudnn.benchmark = True
plt.ion()   # interactive mode
```




    <contextlib.ExitStack at 0x7a73de727880>



## 1. 데이터 열기


```python
# 데이터 증강 및 정규화 설정
# 학습데이터는 데이터 증강 및 정규화 / 검증데이터는 정규화만

data_transforms = {
    'train': transforms.Compose([

        # 이미지를 무작위로 크롭해서 224x224 크기로 변환함
        transforms.RandomResizedCrop(224),

        # 이미지를 무작위로 좌우반전함
        transforms.RandomHorizontalFlip(),

        # 이미지를 텐서로 변환함. PyTorch 모델은 텐서 형태의 데이터를 입력으로 받기 때문
        # 이미지 픽셀값이 [0,255] 범위에서 [0.0,1.0] 범위로 정규화됨
        transforms.ToTensor(),

        # 이미지 데이터를 정규화함. 각 채널(R,G,B)에 대해 평균(0.485, 0.456, 0.406)과 표준편차(0.229, 0.224, 0.225)를 사용하여 정규화함
        # 정규화 공식: normalized_image = (image - mean) / std
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([

        # 이미지를 256x256 크기로 조정
        transforms.Resize(256),

        # 이미지 중앙을 기준으로 224x224 크기로 크롭함
        transforms.CenterCrop(224),

        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = '/content/drive/MyDrive/hymenoptera_data'

# 학습 데이터와 검증 데이터 로드
# datasets.ImageFolder: 지정된 디렉토리에서 이미지를 로드하고 라벨을 자동으로 할당
# data_transforms[x]: 데이터 변환(전처리) 적용
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}

# 학습데이터와 검증데이터를 로드하는 데이터로더 설정
# torch.utils.data.DataLoader: 데이터셋을 로드하고 배치 단위로 나눔
# image_datasets[x]: 로드된 데이터셋 / # num_workers = 4: 데이터를 로드할 때 4개의 작업자를 사용하여 병렬처리함. 데이터 로드 속도를 높이는데 도움됨
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}

# 학습데이터와 검증데이터의 크기(샘플수) 계산
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

# 학습데이터셋의 클래스 이름 가져오기
# image_datasets['train'].classes는 datasets.ImageFolder 클래스의 속성으로, 데이터셋의 클래스 라벨 이름을 반환함
class_names = image_datasets['train'].classes

# 디바이스 설정
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```

## 2. 이미지 몇개 시각화 해보기


```python
# 텐서 형태의 이미지를 시각화할 수 있는 형태로 변환하고 출력함
def imshow(inp, title=None):
    """Display image for Tensor."""

    # inp.numpy(): PyTorch 텐서를 넘파이 배열로 변환
    # transpose((1,2,0)): 텐서의 축 순서를 (채널,높이,너비)에서 (높이,너비,채널)로 변환함. Matplotlib가 이미지를 올바르게 시각화할 수 있도록 하는 작업임
    inp = inp.numpy().transpose((1, 2, 0))

    # 정규화 해제
    # 시각화를 위해서는 각 채널(R,G,B)에 대한 정규화된 값을 원래의 픽셀 값 범위로 되돌려야 함
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean

    # 이미지의 픽셀 값을 0과 1 사이로 제한함. 정규화 해제 후 일부 값이 이 범위를 벗어날 수 있기 때문
    inp = np.clip(inp, 0, 1)

    # 이미지 출력하고 제목 설정함
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # 플롯이 업데이트 될 수 있도록 잠시 멈춤

# 학습 데이터로더에서 한 배치 가져오기
# input: 배치에 포함된 이미지 텐서 / classes: 각 이미지의 클래스 레벨
inputs, classes = next(iter(dataloaders['train']))

# 이미지 배치를 격자 형태로 배열. 여러 이미지를 한 버에 시각화할 수 있도록 함
out = torchvision.utils.make_grid(inputs)

# 이미지 시각화하고 각 이미지에 해당하는 클래스 이름을 제목으로 표
# title인자는 각 클래스의 라벨을 클래스 이름으로 변환하여 리스트로 전달함
imshow(out, title=[class_names[x] for x in classes])
```


    
![png](240621_PytorchOfficialTransferLearningTutorial_files/240621_PytorchOfficialTransferLearningTutorial_7_0.png)
    


## 3. 모델 학습시키기


```python
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):

    # 학습 시작 시간을 기록(학습이 끝난 후에 전체 학습시간을 계산하기 위함)
    since = time.time()

    # Create a temporary directory to save training checkpoints

    # tempdir라는 이름으로 임시 디렉토리 생성
    # 임시 디렉토리는 with블록이 종료되면 자동으로 삭제되므로 임시로 파일을 저장할 때 유용함
    with TemporaryDirectory() as tempdir:

        # 임시 디렉토리 내에 best_model_params.pt라는 파일 경로 생성. 학습 중에 가장 좋은 파라미터를 저장하는 데 사용됨
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

        # 현재 모델의 상태를 best_model_params_path 경로에 저장함
        # model.state_dict(): 모델의 모든 학습 가능한 파라미터와 버퍼를 포함하는 상태 사전을 반환함. 모델의 가중치와 편향을 포함함
        torch.save(model.state_dict(), best_model_params_path)

        # 최고 검증정확도 변수를 초기화함. 학습 과정에서 검증정확도가 더 높은 에포크가 나타날 때마다 이 값을 업데이트하고 해당 에포크의 모델 파라미터를 저장
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            # 각 에포크는 학습과 검증 두 단계로 나뉨
            for phase in ['train', 'val']:
                if phase == 'train':
                    # 모델을 학습모드로 설정
                    model.train()  # 드롭아웃 및 배치정규화 활성화
                else:
                    # 모델을 평가모드로 설정
                    model.eval()   # 드롭아웃 및 배치정규화 비활성화

                # 변수 초기화하여 각 에포크에서 손실과 정확도를 누적함
                running_loss = 0.0
                running_corrects = 0

                # 배치 반복
                # dataloaders[phase]에서 데이터 배치를 반복해서 가져옴
                for inputs, labels in dataloaders[phase]:

                    # 가져온 데이터 배치 inputs, labels를 device로 전송함
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # 이전 배치의 기울기를 초기화함
                    optimizer.zero_grad()

                    # 순전파
                    # 학습 단계가 True일 때만 기울기를 추적함
                    # torch.set_grad_enabled: 기울기 추적을 설정하는 컨텍스트 관리자
                    with torch.set_grad_enabled(phase == 'train'):

                        # 모델에 입력 데이터를 통과시켜 출력을 계산함
                        # inputs: 배치의 입력이미지 / outputs: 모델의 예측 결과
                        outputs = model(inputs)

                        # 예측 계산
                        # torch.max(outputs, 1):  outputs의 각 행에서 최대값을 가진 인덱스를 반환
                        # _: 최대값(값 자체는 필요 없기 때문에 이렇게 표시한거임) / preds: 예측된 클랙스 인덱스
                        _, preds = torch.max(outputs, 1)

                        # 손실 계산
                        # criterion: 손실함수를 나타냄. 일반적으로 nn.CrossEntropyLoss를 사용함
                        loss = criterion(outputs, labels)

                        # 역전파(학습단계에서만) 및 최적화
                        if phase == 'train':

                            # 역전파. 손실값에 대한 기울기를 계산함. 모델의 모든 학습 가능한 파라미터에 대한 기울기를 계산하여 model.parameters()에 저장함
                            loss.backward()

                            # 최적화. 옵티마이저를 사용하여 모델의 파라미터를 업데이트함. 계산된 기울기를 사용하여 파라미터를 조정함
                            optimizer.step()

                    # 통계 업데이트
                    # 배치손실을 배치크기로 곱해서 전체 손실값을 누적함(∵ 배치 크기가 다를 수 있기 때문에 배치 크기에 따른 가중치를 반영하기 위해)
                    # loss.item(): 현재 배치의 손실값 반환 / input.size(0): 현재 배치의 크기(샘플수)
                    running_loss += loss.item() * inputs.size(0)

                    # preds == labels.data: 예측이 실제 라벨과 일치하는지 확인하여 불리언 텐서 생성
                    # torch.sum()으로 True의 개수 구한 다음 running_corrects에 누적함
                    running_corrects += torch.sum(preds == labels.data)

                # 학습단계일 때만 학습률 스케줄러를 업데이트함
                if phase == 'train':

                    # 스케줄러의 상태를 업데이트해서 학습률 조정
                    scheduler.step()

                # running_loss를 전체 데이터셋 크기로 나눠서 평균 손실을 구함
                epoch_loss = running_loss / dataset_sizes[phase]

                # running_corrects를 전체 데이터셋 크기로 나눠서 평균 정확도를 구함
                # double()은 정수형을 부동소수점형으로 변환해서 나눗셈이 정확하게 이루어지도록 하는 역할임
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # 최적 모델 저장
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc

                    # 현재 모델의 상태사전을 best_momdel_params_path에 저장함
                    torch.save(model.state_dict(), best_model_params_path)

            # 에포크 완료 후 줄바꿈을 출력해서 가독성을 높임
            print()

        # 현재시간에서 학습시작시간을 빼서 전체 학습시간을 계산함
        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        # 저장된 최적 모델 파라미터를 로드해서 모델에 적용함
        model.load_state_dict(torch.load(best_model_params_path))

    # 최적 모델 반환
    return model
```

## 4. 모델이 예측한 거 시각화하기


```python
# num_images: 표시할 이미지 수
def visualize_model(model, num_images=6):

    # was_training: 모델의초기 훈련 상태를 저장
    was_training = model.training

    # 모델을 평가모드로 설정. 드롭아웃과 같은 훈련 전용 동작을 끔
    model.eval()

    # 처리된 이미지의 수를 추적하는 변수
    images_so_far = 0

    # 이미지를 그리기 위한새로운 그림(fig) 생성
    fig = plt.figure()

    # torch.no_grad(): 기울기 계산을 비활성화하여 메모리 사용량을 줄이고 계산 속도를 높임
    with torch.no_grad():
        # dataloaders['val']: 검증데이터셋의 데이터로더로 가정됨
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            # 이미지 예측 및 결과 시각화
            for j in range(inputs.size()[0]):

                # 처리된 이미지수 증가시키기
                images_so_far += 1

                # 그림에 서브플롯을 추가함(num_images//2 행 2열로 배치됨)
                ax = plt.subplot(num_images//2, 2, images_so_far)

                # 서브플롯의 축 끄기
                ax.axis('off')

                ax.set_title(f'predicted: {class_names[preds[j]]}')

                # imshow 함수를 사용하여 이미지 표시. 이미지를 CPU로 이동시키고 NumPy 배열로 변환함
                imshow(inputs.cpu().data[j])

                # 종료조건 및 훈련 모드로 복귀
                # 원하는 수의 이미지가 처리되었다면
                if images_so_far == num_images:

                    # 모델을 훈련 상태로 되돌림
                    model.train(mode=was_training)

                    # 원하는 수의 이미지가 시각화되면 함수를 종료함
                    return

        # 반복문을 벗어난 후에도 모델이 원래의 훈련 상태로 돌아가도록 함
        model.train(mode=was_training)
```

## 5. ConvNet 파인튜닝하기


```python
# 사전학습된 가중치를 사용하여 ResNet-18 모델을 로드함
# IMAGENET1K_V1: ImageNet으로학습된 가중치
model_ft = models.resnet18(weights='IMAGENET1K_V1')

# 모델의 마지막 fully connected layer의 입력 특징 수를 얻음
num_ftrs = model_ft.fc.in_features

# 모델의 마지막 완전연결층을 새로 정의하여 출력 노드수를 2로 설정함. 분류할 클래스 수가 2개임을 의미함
# 더 일반화된 방법으로는 `nn.Linear(num_ftrs, len(class_names))`를 사용해서 클래스 수에 따라 동적으로 설정할 수 있음
model_ft.fc = nn.Linear(num_ftrs, 2)

# 모델을 지정된 장치로 이동
model_ft = model_ft.to(device)

# 손실함수 정의. 손실함수로 CE를 사용함
criterion = nn.CrossEntropyLoss()

# SGD를 옵티마이저로 설정하고 lr을 0.001, 모멘텀을 0.9fh tjfwjdgka
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# 학습률 스케줄러를 설정하여 7에포크마다 학습률을 0.1배로 감소킴
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
```

    Downloading: "https://download.pytorch.org/models/resnet18-f37072fd.pth" to /root/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth
    100%|██████████| 44.7M/44.7M [00:00<00:00, 185MB/s]
    

## 6. 학습시키고 평가하기


```python
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=25)
```

    Epoch 0/24
    ----------
    train Loss: 0.7619 Acc: 0.6844
    val Loss: 0.3862 Acc: 0.8562
    
    Epoch 1/24
    ----------
    train Loss: 0.5507 Acc: 0.7828
    val Loss: 0.3697 Acc: 0.8431
    
    Epoch 2/24
    ----------
    train Loss: 0.4787 Acc: 0.8156
    val Loss: 0.3743 Acc: 0.8693
    
    Epoch 3/24
    ----------
    train Loss: 0.5460 Acc: 0.7910
    val Loss: 0.2683 Acc: 0.9216
    
    Epoch 4/24
    ----------
    train Loss: 0.5598 Acc: 0.7910
    val Loss: 0.3392 Acc: 0.9020
    
    Epoch 5/24
    ----------
    train Loss: 0.5607 Acc: 0.7705
    val Loss: 0.3501 Acc: 0.8627
    
    Epoch 6/24
    ----------
    train Loss: 0.3676 Acc: 0.8648
    val Loss: 0.3048 Acc: 0.8954
    
    Epoch 7/24
    ----------
    train Loss: 0.3396 Acc: 0.8484
    val Loss: 0.2667 Acc: 0.8824
    
    Epoch 8/24
    ----------
    train Loss: 0.2829 Acc: 0.8730
    val Loss: 0.2450 Acc: 0.8758
    
    Epoch 9/24
    ----------
    train Loss: 0.3593 Acc: 0.8361
    val Loss: 0.2182 Acc: 0.9020
    
    Epoch 10/24
    ----------
    train Loss: 0.4098 Acc: 0.8320
    val Loss: 0.2551 Acc: 0.8824
    
    Epoch 11/24
    ----------
    train Loss: 0.3229 Acc: 0.8648
    val Loss: 0.2342 Acc: 0.9346
    
    Epoch 12/24
    ----------
    train Loss: 0.2691 Acc: 0.8811
    val Loss: 0.2093 Acc: 0.9085
    
    Epoch 13/24
    ----------
    train Loss: 0.2560 Acc: 0.8934
    val Loss: 0.2126 Acc: 0.9216
    
    Epoch 14/24
    ----------
    train Loss: 0.2107 Acc: 0.9303
    val Loss: 0.2600 Acc: 0.8824
    
    Epoch 15/24
    ----------
    train Loss: 0.1914 Acc: 0.9344
    val Loss: 0.2175 Acc: 0.9150
    
    Epoch 16/24
    ----------
    train Loss: 0.2525 Acc: 0.8975
    val Loss: 0.2382 Acc: 0.9085
    
    Epoch 17/24
    ----------
    train Loss: 0.3163 Acc: 0.8484
    val Loss: 0.2200 Acc: 0.9150
    
    Epoch 18/24
    ----------
    train Loss: 0.2965 Acc: 0.8730
    val Loss: 0.2094 Acc: 0.8889
    
    Epoch 19/24
    ----------
    train Loss: 0.2902 Acc: 0.8852
    val Loss: 0.2448 Acc: 0.8824
    
    Epoch 20/24
    ----------
    train Loss: 0.2564 Acc: 0.8730
    val Loss: 0.2092 Acc: 0.9216
    
    Epoch 21/24
    ----------
    train Loss: 0.2215 Acc: 0.9098
    val Loss: 0.2169 Acc: 0.9085
    
    Epoch 22/24
    ----------
    train Loss: 0.2826 Acc: 0.8852
    val Loss: 0.2318 Acc: 0.8889
    
    Epoch 23/24
    ----------
    train Loss: 0.2201 Acc: 0.9180
    val Loss: 0.2070 Acc: 0.9085
    
    Epoch 24/24
    ----------
    train Loss: 0.2779 Acc: 0.8852
    val Loss: 0.2239 Acc: 0.9020
    
    Training complete in 1m 48s
    Best val Acc: 0.934641
    


```python
# model_ft 모델의 예측 결과 시각화
visualize_model(model_ft)
```


    
![png](240621_PytorchOfficialTransferLearningTutorial_files/240621_PytorchOfficialTransferLearningTutorial_16_0.png)
    



    
![png](240621_PytorchOfficialTransferLearningTutorial_files/240621_PytorchOfficialTransferLearningTutorial_16_1.png)
    



    
![png](240621_PytorchOfficialTransferLearningTutorial_files/240621_PytorchOfficialTransferLearningTutorial_16_2.png)
    



    
![png](240621_PytorchOfficialTransferLearningTutorial_files/240621_PytorchOfficialTransferLearningTutorial_16_3.png)
    



    
![png](240621_PytorchOfficialTransferLearningTutorial_files/240621_PytorchOfficialTransferLearningTutorial_16_4.png)
    



    
![png](240621_PytorchOfficialTransferLearningTutorial_files/240621_PytorchOfficialTransferLearningTutorial_16_5.png)
    


## 7. CNN을 고정된 특징 추출기로 사용하기
- CNN의 가중치가 업데이트되지 않고 고정된 상태로 사용된다는 뜻
- 이미 학습된 모델을 사용할 때 씀


```python
# ResNet-18모델 로드 및 파라미터 고정
model_conv = torchvision.models.resnet18(weights='IMAGENET1K_V1')
for param in model_conv.parameters():

    # 모델의 모든 파라미터에 대해 기울기 계산을 비활성화하여 파라미터를 고정함
    # 이는 사전 학습된 가중치를 그대로 유지하고, 모델의 일부만 재학습할 때 사용됨
    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
# 모델의 마지막 fully connected layer의 입력 특징 수를 얻음
num_ftrs = model_conv.fc.in_features

# 모델의 마지막 fully connected layer를 새로 정의하여 출력 노드 수를 2로 설정함. 이 부분은 고정된 파라미터와 달리 학습이 가능하게 됨
model_conv.fc = nn.Linear(num_ftrs, 2)

model_conv = model_conv.to(device)

criterion = nn.CrossEntropyLoss()

# 최종 레이어(model_conv.fc)의 파라미터만 최적화의 대상임. 이전 레이어들은 고정되어 학습되지 않음
optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)
```

## 8. 학습시키고 평가하기


```python
model_conv = train_model(model_conv, criterion, optimizer_conv,
                         exp_lr_scheduler, num_epochs=25)
```

    Epoch 0/24
    ----------
    train Loss: 0.6025 Acc: 0.6639
    val Loss: 0.2298 Acc: 0.9085
    
    Epoch 1/24
    ----------
    train Loss: 0.3998 Acc: 0.8156
    val Loss: 0.2041 Acc: 0.9412
    
    Epoch 2/24
    ----------
    train Loss: 0.5002 Acc: 0.7828
    val Loss: 0.2000 Acc: 0.9020
    
    Epoch 3/24
    ----------
    train Loss: 0.4390 Acc: 0.8115
    val Loss: 0.1745 Acc: 0.9412
    
    Epoch 4/24
    ----------
    train Loss: 0.7883 Acc: 0.7213
    val Loss: 0.1662 Acc: 0.9542
    
    Epoch 5/24
    ----------
    train Loss: 0.7272 Acc: 0.7664
    val Loss: 0.1618 Acc: 0.9542
    
    Epoch 6/24
    ----------
    train Loss: 0.3636 Acc: 0.8975
    val Loss: 0.2821 Acc: 0.9020
    
    Epoch 7/24
    ----------
    train Loss: 0.3409 Acc: 0.8566
    val Loss: 0.1715 Acc: 0.9608
    
    Epoch 8/24
    ----------
    train Loss: 0.3525 Acc: 0.8402
    val Loss: 0.1635 Acc: 0.9477
    
    Epoch 9/24
    ----------
    train Loss: 0.4322 Acc: 0.8074
    val Loss: 0.1563 Acc: 0.9542
    
    Epoch 10/24
    ----------
    train Loss: 0.3396 Acc: 0.8852
    val Loss: 0.1697 Acc: 0.9542
    
    Epoch 11/24
    ----------
    train Loss: 0.3084 Acc: 0.8852
    val Loss: 0.1620 Acc: 0.9542
    
    Epoch 12/24
    ----------
    train Loss: 0.3320 Acc: 0.8484
    val Loss: 0.1815 Acc: 0.9477
    
    Epoch 13/24
    ----------
    train Loss: 0.2558 Acc: 0.8811
    val Loss: 0.1616 Acc: 0.9608
    
    Epoch 14/24
    ----------
    train Loss: 0.4073 Acc: 0.8402
    val Loss: 0.1688 Acc: 0.9477
    
    Epoch 15/24
    ----------
    train Loss: 0.3990 Acc: 0.8033
    val Loss: 0.1602 Acc: 0.9542
    
    Epoch 16/24
    ----------
    train Loss: 0.3690 Acc: 0.8279
    val Loss: 0.1929 Acc: 0.9412
    
    Epoch 17/24
    ----------
    train Loss: 0.2916 Acc: 0.8730
    val Loss: 0.1994 Acc: 0.9412
    
    Epoch 18/24
    ----------
    train Loss: 0.2906 Acc: 0.8770
    val Loss: 0.1520 Acc: 0.9542
    
    Epoch 19/24
    ----------
    train Loss: 0.3863 Acc: 0.8484
    val Loss: 0.1732 Acc: 0.9608
    
    Epoch 20/24
    ----------
    train Loss: 0.3606 Acc: 0.8361
    val Loss: 0.1720 Acc: 0.9608
    
    Epoch 21/24
    ----------
    train Loss: 0.3856 Acc: 0.8074
    val Loss: 0.2094 Acc: 0.9281
    
    Epoch 22/24
    ----------
    train Loss: 0.3194 Acc: 0.8525
    val Loss: 0.1697 Acc: 0.9542
    
    Epoch 23/24
    ----------
    train Loss: 0.2841 Acc: 0.8893
    val Loss: 0.1786 Acc: 0.9608
    
    Epoch 24/24
    ----------
    train Loss: 0.2697 Acc: 0.8811
    val Loss: 0.1464 Acc: 0.9542
    
    Training complete in 0m 39s
    Best val Acc: 0.960784
    


```python
visualize_model(model_conv)

# matplotlib의 인터랙티브 모드를 끔. 인터랙티브 모드에서는 플롯이 즉시 업데이트 되고 표시되는데, 끄면 plt.show()를 호출할 때까지 표시되지 않음
plt.ioff()
plt.show()
```


    
![png](240621_PytorchOfficialTransferLearningTutorial_files/240621_PytorchOfficialTransferLearningTutorial_21_0.png)
    



    
![png](240621_PytorchOfficialTransferLearningTutorial_files/240621_PytorchOfficialTransferLearningTutorial_21_1.png)
    



    
![png](240621_PytorchOfficialTransferLearningTutorial_files/240621_PytorchOfficialTransferLearningTutorial_21_2.png)
    



    
![png](240621_PytorchOfficialTransferLearningTutorial_files/240621_PytorchOfficialTransferLearningTutorial_21_3.png)
    



    
![png](240621_PytorchOfficialTransferLearningTutorial_files/240621_PytorchOfficialTransferLearningTutorial_21_4.png)
    



    
![png](240621_PytorchOfficialTransferLearningTutorial_files/240621_PytorchOfficialTransferLearningTutorial_21_5.png)
    


##9. 새로운 데이터 사용해서 예측하기


```python
def visualize_model_predictions(model,img_path):
    was_training = model.training
    model.eval()

    img = Image.open(img_path)
    img = data_transforms['val'](img)
    img = img.unsqueeze(0)
    img = img.to(device)

    with torch.no_grad():
        outputs = model(img)
        _, preds = torch.max(outputs, 1)

        ax = plt.subplot(2,2,1)
        ax.axis('off')
        ax.set_title(f'Predicted: {class_names[preds[0]]}')
        imshow(img.cpu().data[0])

        model.train(mode=was_training)
```


```python
visualize_model_predictions(
    model_conv,
    img_path='/content/drive/MyDrive/hymenoptera_data/val/bees/72100438_73de9f17af.jpg'
)

plt.ioff()
plt.show()
```


    
![png](240621_PytorchOfficialTransferLearningTutorial_files/240621_PytorchOfficialTransferLearningTutorial_24_0.png)
    

