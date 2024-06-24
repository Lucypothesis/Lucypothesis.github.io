---
layout: post
title: "240624(월) [과제] 난이도(중)-2 Recurrent Neural Networks (RNN or LSTM or GRU)를 직접 구성하여 98% 이상의 성능을 내는 MNIST 분류기 만들기"
subtitle: "[Tips]"
date: 2024-06-24 22:21
background: 
tag: [Tips, Github io, Notion]
---

**난이도(중) 2번문제**   
Recurrent Neural Networks (RNN or LSTM or GRU)를 직접 구성하여 98% 이상의 성능을 내는 MNIST 분류기 만들기

1. RNN, LSTM, 그리고 GRU에 대해 이해하기   
2. 각각을 이용하여 MNIST에 분류기를 구현하며 각 모듈이 실무에서의 활용될 때 어떤 장/단점이 있는지 이해하기

# Recurrent Neural Networks (RNN or LSTM or GRU)를 직접 구성하여 98% 이상의 성능을 내는 MNIST 분류기 만들기

- RNN, LSTM, GRU는 시퀀스 데이터를 처리하는 데 사용되는 RNN 계열의 다양한 아키텍처임

## 1. RNN, LSTM, 그리고 GRU에 대해 이해하기

### 1.1 RNN (Recurrent Neural Network)

**특징**
- 각 타임스텝의 입력과 이전 타임스텝의 은닉상태를 결합하여 현재 타임스텝의 은닉 상태를 계산함
- 은닉상태는 시쿼스의 정보가 저장되는 메모리 역할을 하고, 순환적으로 다음 타임스텝으로 전달됨

**장점**
- 구조가 간단하고 이해하기 쉬움
- 순차데이터(ex. 시계열, 텍스트 데이터 등)를 처리하는데 적합함

**단점**
- 장기 의존성 문제: 시간이 지남에 따라 초기 입력 정보가 소멸되어 긴 시퀀스를 처리하기 어려움

### 1.2 LSTM (Long Short-Term Memory)

**특징**
- RNN의 장기 의존성 문제를 해결하기 위해 고안됨
- 셀 상태(cell state)와 게이트 세 가지 게이트(입력, 출력, 망각)을 통해 정보를 유지하고 업데이트함
  - 셀 상태: 정보가 타임스텝 간에 거의 변경되지 않고 전달되는 경로. 장기 의존성 문제를 해결함
  - 입력게이트: 현재 입력을 얼마나 셀 상태에 반영할지 결정
  - 망각게이트: 셀 상태에서 어떤 정보를 잊을지 결정
  - 출력게이트: 은닉 상태를 얼마나 다음 타임스텝에 반영할지 결정

**장점**
- 장기의존성을 잘 처리하여 긴 시퀀스 데이터에도 성능이 우수함

**단점**
- 구조가 복잡하고 계산 비용이 높음

### 1.3 GRU (Gated Recurrent Unit)

**특징**
- RNN의 장기 의존성 문제를 해결하기 위해 고안됨
- LSTM의 단순화된 버전
- 셀 상태를 직접 사용하지 않고 두 개의 게이트(업데이트 게이트, 리셋 게이트)만 사용함
  - 업데이트게이트: 이전 은닉 상태와 현재 입력을 결합하여 새로운 은닉 상태를 생성
  - 리셋게이트: 이전 은닉 상태를 얼마나 무시할지 결정

**장점**
- LSTM과 유사한 성능을 보이면서 더 단순한 구조로 계산 비용이 적음

**단점**
- 일부 경우에서 LSTM보다 덜 유연할 수 있음(하지만 대부분의 작업에 충분히 강력함)

### 1.4 RNN, LSTM, GRU 비교

| 특성       | RNN                       | LSTM                                      | GRU                                      |
|:------------:|:---------------------------:|:-------------------------------------------:|:------------------------------------------:|
| 구조       | 단순                      | 복잡                                      | 단순                                    |
| 장기 의존성| 취약                      | 강함                                      | 강함                                    |
| 계산 비용  | 낮음                      | 높음                                      | 중간                                    |
| 성능       | 짧은 시퀀스에 적합         | 긴 시퀀스에 적합                           | 대부분의 시퀀스 작업에 적합             |
| 게이트     | 없음                      | 입력 게이트, 망각 게이트, 출력 게이트      | 업데이트 게이트, 리셋 게이트             |
| 셀 상태    | 없음                      | 있음                                      | 없음                                    |


## 2. 각각을 이용하여 MNIST에 분류기를 구현하며 각 모듈이 실무에서 활용될 때 어떤 장/단점이 있는지 이해하기

각각을 이용해서 MNIST에 분류기를 구현해보고자 한다. Tensorflow와 PyTorch 중 무엇으로 구현해볼까 하다가 결국 tensorflow로 구현하기로 했다. 각 모듈이 '실무에서 활용될 때' 어떤 장/단점이 있는지 보라는 문구를 봤을 때 실무에서 더 자주 활용되는 프레임워크를 사용하는 게 좋을 것이라고 판단했기 때문이다. Tensorflow는 산업에서, PyTorch는 연구 및 개발환경에서 주로 사용된다고 이해하고 있어서, 이번 코드는 tensorflow로 구현해보고자 한다.

### 2.1 라이브러리 임포트 & 데이터 전처리


```python
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical # 정수 레이블을 원핫 인코딩 형태로 변환하는데 사용됨

# MNIST 데이터셋 로드
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 데이터 전처리
# train_images와 test_images의 모든 값을 255로 나눠서 0과 1사이의 범위로 정규화함. 이렇게 하면 학습 과정에서 수치가 안정되고 학습 속도가 빨라짐
train_images = train_images / 255.0
test_images = test_images / 255.0

# train_labels와 test_labels의 레이블 숫자를 to_categorical을 활용해서 정수 레이블을 원핫인코딩 형식으로 변환함
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
```

    Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
    11490434/11490434 [==============================] - 0s 0us/step
    

### 2.2 RNN 모델 구현


```python
from tensorflow.keras.models import Sequential # 층을 순차적으로 쌓아 올릴 때 사용됨
from tensorflow.keras.layers import SimpleRNN, Dense # SimpleRNN: 간단한 RNN레이어 생성/ Dense: FC레이어 생성

# Sequential 모델을 정의
model_rnn = Sequential([

    # 128개의 은닉유닛을 갖는 SimpleRNN 레이어를 추가함
    # input_shape: 입력데이터의 형태를 정의. MNIST이미지가 28x28이기 때문에 (28,28)임
    SimpleRNN(128, input_shape=(28, 28)),

    # 10개의 출력유닛을 갖는 Dense 레이어를 추가함(0~9까지 숫자 분류하기 위해)
    # Softmax함수를 사용해서 출력값을 확률로 변환함. 출력값의 합은 1이 됨
    Dense(10, activation='softmax')
])

# Compile 메서드 호출해서 모델을 컴파일함. 컴파일 단계에서는 모델이 학습할 때 사용할 최적화 방법과 손실함수, 평가지표를 정의함
# Categorical_crossentropy: 다중 클래스 분류 문제에서 사용됨(+ 그냥 cross entropy라고 불리는 건 binary cross entropy로 이진 분류 문제에서 사용됨)
model_rnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

### 2.3 LSTM 모델 구현


```python
from tensorflow.keras.layers import LSTM

# LSTM 모델 구성
model_lstm = Sequential([
    LSTM(128, input_shape=(28, 28)),
    Dense(10, activation='softmax')
])

model_lstm.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

### 2.4 GRU 모델 구현


```python
from tensorflow.keras.layers import GRU

# GRU 모델 구성
model_gru = Sequential([
    GRU(128, input_shape=(28, 28)),
    Dense(10, activation='softmax')
])

model_gru.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

### 2.5 모델 학습 및 평가


```python
# tf.keras.callbacks.Callback를 상속해서 커스텀 콜백 클래스를 정의함
class EpochLogger(tf.keras.callbacks.Callback):

    # on_epoch_end: 각 에포크가 끝날 때 호출됨 / on_epoch_end라는 메서드 자체가 tf.keras.callbacks.Callback 클래스의 메서드 중 하나로 정의되어 있음
    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch + 1}, Loss: {logs['loss']:.4f}, Accuracy: {logs['accuracy']:.4f}, Val_Loss: {logs['val_loss']:.4f}, Val_Accuracy: {logs['val_accuracy']:.4f}")


def train_and_evaluate(model, train_images, train_labels, test_images, test_labels):

    # epoch_logger 인스턴스를 생성해서 콜백으로 사용함
    epoch_logger = EpochLogger()

    # callbacks의 인자로 epoch_logger를 전달해서 각 에포크가 끝날 때마다 로그를 출력하도록 함
    model.fit(train_images, train_labels, epochs=10, batch_size=64, validation_split=0.2, callbacks=[epoch_logger])

    # 학습이 끝난 후 model.evaluate 메서드를 사용해서 테스트 데이터에 대한 손실과 정확도를 평가함
    test_loss, test_acc = model.evaluate(test_images, test_labels)

    # 테스트 정확도 반환
    return test_acc

# RNN 학습 및 평가
acc_rnn = train_and_evaluate(model_rnn, train_images, train_labels, test_images, test_labels)
print(f"RNN Test Accuracy: {acc_rnn:.4f}")

# LSTM 학습 및 평가
acc_lstm = train_and_evaluate(model_lstm, train_images, train_labels, test_images, test_labels)
print(f"LSTM Test Accuracy: {acc_lstm:.4f}")

# GRU 학습 및 평가
acc_gru = train_and_evaluate(model_gru, train_images, train_labels, test_images, test_labels)
print(f"GRU Test Accuracy: {acc_gru:.4f}")
```

    Epoch 1/10
    748/750 [============================>.] - ETA: 0s - loss: 0.4270 - accuracy: 0.8731Epoch 1, Loss: 0.4265, Accuracy: 0.8732, Val_Loss: 0.3113, Val_Accuracy: 0.9053
    750/750 [==============================] - 9s 10ms/step - loss: 0.4265 - accuracy: 0.8732 - val_loss: 0.3113 - val_accuracy: 0.9053
    Epoch 2/10
    750/750 [==============================] - ETA: 0s - loss: 0.2094 - accuracy: 0.9386Epoch 2, Loss: 0.2094, Accuracy: 0.9386, Val_Loss: 0.1714, Val_Accuracy: 0.9517
    750/750 [==============================] - 7s 9ms/step - loss: 0.2094 - accuracy: 0.9386 - val_loss: 0.1714 - val_accuracy: 0.9517
    Epoch 3/10
    744/750 [============================>.] - ETA: 0s - loss: 0.1691 - accuracy: 0.9503Epoch 3, Loss: 0.1696, Accuracy: 0.9502, Val_Loss: 0.1419, Val_Accuracy: 0.9600
    750/750 [==============================] - 7s 9ms/step - loss: 0.1696 - accuracy: 0.9502 - val_loss: 0.1419 - val_accuracy: 0.9600
    Epoch 4/10
    747/750 [============================>.] - ETA: 0s - loss: 0.1487 - accuracy: 0.9563Epoch 4, Loss: 0.1485, Accuracy: 0.9564, Val_Loss: 0.1420, Val_Accuracy: 0.9591
    750/750 [==============================] - 7s 9ms/step - loss: 0.1485 - accuracy: 0.9564 - val_loss: 0.1420 - val_accuracy: 0.9591
    Epoch 5/10
    744/750 [============================>.] - ETA: 0s - loss: 0.1331 - accuracy: 0.9608Epoch 5, Loss: 0.1328, Accuracy: 0.9609, Val_Loss: 0.1673, Val_Accuracy: 0.9515
    750/750 [==============================] - 7s 9ms/step - loss: 0.1328 - accuracy: 0.9609 - val_loss: 0.1673 - val_accuracy: 0.9515
    Epoch 6/10
    744/750 [============================>.] - ETA: 0s - loss: 0.1246 - accuracy: 0.9636Epoch 6, Loss: 0.1247, Accuracy: 0.9636, Val_Loss: 0.1270, Val_Accuracy: 0.9644
    750/750 [==============================] - 7s 9ms/step - loss: 0.1247 - accuracy: 0.9636 - val_loss: 0.1270 - val_accuracy: 0.9644
    Epoch 7/10
    745/750 [============================>.] - ETA: 0s - loss: 0.1190 - accuracy: 0.9649Epoch 7, Loss: 0.1188, Accuracy: 0.9649, Val_Loss: 0.1351, Val_Accuracy: 0.9616
    750/750 [==============================] - 7s 9ms/step - loss: 0.1188 - accuracy: 0.9649 - val_loss: 0.1351 - val_accuracy: 0.9616
    Epoch 8/10
    744/750 [============================>.] - ETA: 0s - loss: 0.1095 - accuracy: 0.9683Epoch 8, Loss: 0.1094, Accuracy: 0.9683, Val_Loss: 0.1155, Val_Accuracy: 0.9683
    750/750 [==============================] - 7s 9ms/step - loss: 0.1094 - accuracy: 0.9683 - val_loss: 0.1155 - val_accuracy: 0.9683
    Epoch 9/10
    744/750 [============================>.] - ETA: 0s - loss: 0.1059 - accuracy: 0.9697Epoch 9, Loss: 0.1060, Accuracy: 0.9696, Val_Loss: 0.1146, Val_Accuracy: 0.9668
    750/750 [==============================] - 7s 9ms/step - loss: 0.1060 - accuracy: 0.9696 - val_loss: 0.1146 - val_accuracy: 0.9668
    Epoch 10/10
    746/750 [============================>.] - ETA: 0s - loss: 0.1019 - accuracy: 0.9699Epoch 10, Loss: 0.1021, Accuracy: 0.9699, Val_Loss: 0.1127, Val_Accuracy: 0.9681
    750/750 [==============================] - 7s 9ms/step - loss: 0.1021 - accuracy: 0.9699 - val_loss: 0.1127 - val_accuracy: 0.9681
    313/313 [==============================] - 1s 4ms/step - loss: 0.1167 - accuracy: 0.9663
    RNN Test Accuracy: 0.9663
    Epoch 1/10
    736/750 [============================>.] - ETA: 0s - loss: 0.4948 - accuracy: 0.8375Epoch 1, Loss: 0.4904, Accuracy: 0.8390, Val_Loss: 0.1982, Val_Accuracy: 0.9398
    750/750 [==============================] - 5s 5ms/step - loss: 0.4904 - accuracy: 0.8390 - val_loss: 0.1982 - val_accuracy: 0.9398
    Epoch 2/10
    746/750 [============================>.] - ETA: 0s - loss: 0.1477 - accuracy: 0.9559Epoch 2, Loss: 0.1479, Accuracy: 0.9558, Val_Loss: 0.1125, Val_Accuracy: 0.9658
    750/750 [==============================] - 3s 4ms/step - loss: 0.1479 - accuracy: 0.9558 - val_loss: 0.1125 - val_accuracy: 0.9658
    Epoch 3/10
    737/750 [============================>.] - ETA: 0s - loss: 0.1010 - accuracy: 0.9692Epoch 3, Loss: 0.1012, Accuracy: 0.9693, Val_Loss: 0.0908, Val_Accuracy: 0.9727
    750/750 [==============================] - 3s 4ms/step - loss: 0.1012 - accuracy: 0.9693 - val_loss: 0.0908 - val_accuracy: 0.9727
    Epoch 4/10
    739/750 [============================>.] - ETA: 0s - loss: 0.0815 - accuracy: 0.9753Epoch 4, Loss: 0.0813, Accuracy: 0.9754, Val_Loss: 0.0704, Val_Accuracy: 0.9792
    750/750 [==============================] - 3s 4ms/step - loss: 0.0813 - accuracy: 0.9754 - val_loss: 0.0704 - val_accuracy: 0.9792
    Epoch 5/10
    745/750 [============================>.] - ETA: 0s - loss: 0.0619 - accuracy: 0.9810Epoch 5, Loss: 0.0619, Accuracy: 0.9810, Val_Loss: 0.0736, Val_Accuracy: 0.9780
    750/750 [==============================] - 3s 4ms/step - loss: 0.0619 - accuracy: 0.9810 - val_loss: 0.0736 - val_accuracy: 0.9780
    Epoch 6/10
    749/750 [============================>.] - ETA: 0s - loss: 0.0549 - accuracy: 0.9825Epoch 6, Loss: 0.0550, Accuracy: 0.9824, Val_Loss: 0.0674, Val_Accuracy: 0.9810
    750/750 [==============================] - 3s 4ms/step - loss: 0.0550 - accuracy: 0.9824 - val_loss: 0.0674 - val_accuracy: 0.9810
    Epoch 7/10
    739/750 [============================>.] - ETA: 0s - loss: 0.0442 - accuracy: 0.9858Epoch 7, Loss: 0.0442, Accuracy: 0.9859, Val_Loss: 0.0660, Val_Accuracy: 0.9806
    750/750 [==============================] - 3s 4ms/step - loss: 0.0442 - accuracy: 0.9859 - val_loss: 0.0660 - val_accuracy: 0.9806
    Epoch 8/10
    746/750 [============================>.] - ETA: 0s - loss: 0.0396 - accuracy: 0.9876Epoch 8, Loss: 0.0395, Accuracy: 0.9876, Val_Loss: 0.0593, Val_Accuracy: 0.9817
    750/750 [==============================] - 3s 4ms/step - loss: 0.0395 - accuracy: 0.9876 - val_loss: 0.0593 - val_accuracy: 0.9817
    Epoch 9/10
    746/750 [============================>.] - ETA: 0s - loss: 0.0339 - accuracy: 0.9892Epoch 9, Loss: 0.0339, Accuracy: 0.9892, Val_Loss: 0.0539, Val_Accuracy: 0.9847
    750/750 [==============================] - 3s 4ms/step - loss: 0.0339 - accuracy: 0.9892 - val_loss: 0.0539 - val_accuracy: 0.9847
    Epoch 10/10
    746/750 [============================>.] - ETA: 0s - loss: 0.0308 - accuracy: 0.9906Epoch 10, Loss: 0.0309, Accuracy: 0.9906, Val_Loss: 0.0643, Val_Accuracy: 0.9815
    750/750 [==============================] - 3s 4ms/step - loss: 0.0309 - accuracy: 0.9906 - val_loss: 0.0643 - val_accuracy: 0.9815
    313/313 [==============================] - 1s 2ms/step - loss: 0.0633 - accuracy: 0.9810
    LSTM Test Accuracy: 0.9810
    Epoch 1/10
    744/750 [============================>.] - ETA: 0s - loss: 0.5455 - accuracy: 0.8204Epoch 1, Loss: 0.5429, Accuracy: 0.8214, Val_Loss: 0.1900, Val_Accuracy: 0.9421
    750/750 [==============================] - 5s 5ms/step - loss: 0.5429 - accuracy: 0.8214 - val_loss: 0.1900 - val_accuracy: 0.9421
    Epoch 2/10
    743/750 [============================>.] - ETA: 0s - loss: 0.1558 - accuracy: 0.9536Epoch 2, Loss: 0.1551, Accuracy: 0.9538, Val_Loss: 0.0997, Val_Accuracy: 0.9698
    750/750 [==============================] - 3s 4ms/step - loss: 0.1551 - accuracy: 0.9538 - val_loss: 0.0997 - val_accuracy: 0.9698
    Epoch 3/10
    742/750 [============================>.] - ETA: 0s - loss: 0.1016 - accuracy: 0.9690Epoch 3, Loss: 0.1014, Accuracy: 0.9691, Val_Loss: 0.0904, Val_Accuracy: 0.9727
    750/750 [==============================] - 3s 4ms/step - loss: 0.1014 - accuracy: 0.9691 - val_loss: 0.0904 - val_accuracy: 0.9727
    Epoch 4/10
    747/750 [============================>.] - ETA: 0s - loss: 0.0785 - accuracy: 0.9760Epoch 4, Loss: 0.0784, Accuracy: 0.9760, Val_Loss: 0.0651, Val_Accuracy: 0.9811
    750/750 [==============================] - 3s 4ms/step - loss: 0.0784 - accuracy: 0.9760 - val_loss: 0.0651 - val_accuracy: 0.9811
    Epoch 5/10
    741/750 [============================>.] - ETA: 0s - loss: 0.0616 - accuracy: 0.9812Epoch 5, Loss: 0.0618, Accuracy: 0.9812, Val_Loss: 0.0598, Val_Accuracy: 0.9823
    750/750 [==============================] - 3s 4ms/step - loss: 0.0618 - accuracy: 0.9812 - val_loss: 0.0598 - val_accuracy: 0.9823
    Epoch 6/10
    747/750 [============================>.] - ETA: 0s - loss: 0.0511 - accuracy: 0.9842Epoch 6, Loss: 0.0511, Accuracy: 0.9842, Val_Loss: 0.0595, Val_Accuracy: 0.9822
    750/750 [==============================] - 3s 4ms/step - loss: 0.0511 - accuracy: 0.9842 - val_loss: 0.0595 - val_accuracy: 0.9822
    Epoch 7/10
    741/750 [============================>.] - ETA: 0s - loss: 0.0423 - accuracy: 0.9873Epoch 7, Loss: 0.0427, Accuracy: 0.9872, Val_Loss: 0.0525, Val_Accuracy: 0.9847
    750/750 [==============================] - 3s 4ms/step - loss: 0.0427 - accuracy: 0.9872 - val_loss: 0.0525 - val_accuracy: 0.9847
    Epoch 8/10
    750/750 [==============================] - ETA: 0s - loss: 0.0360 - accuracy: 0.9890Epoch 8, Loss: 0.0360, Accuracy: 0.9890, Val_Loss: 0.0556, Val_Accuracy: 0.9847
    750/750 [==============================] - 3s 4ms/step - loss: 0.0360 - accuracy: 0.9890 - val_loss: 0.0556 - val_accuracy: 0.9847
    Epoch 9/10
    747/750 [============================>.] - ETA: 0s - loss: 0.0319 - accuracy: 0.9901Epoch 9, Loss: 0.0319, Accuracy: 0.9901, Val_Loss: 0.0679, Val_Accuracy: 0.9812
    750/750 [==============================] - 3s 4ms/step - loss: 0.0319 - accuracy: 0.9901 - val_loss: 0.0679 - val_accuracy: 0.9812
    Epoch 10/10
    748/750 [============================>.] - ETA: 0s - loss: 0.0274 - accuracy: 0.9909Epoch 10, Loss: 0.0274, Accuracy: 0.9909, Val_Loss: 0.0468, Val_Accuracy: 0.9868
    750/750 [==============================] - 3s 4ms/step - loss: 0.0274 - accuracy: 0.9909 - val_loss: 0.0468 - val_accuracy: 0.9868
    313/313 [==============================] - 1s 2ms/step - loss: 0.0548 - accuracy: 0.9850
    GRU Test Accuracy: 0.9850
    

RNN 모델의 정확도가 0.9663, LSTM은 0.9810, GRU는 0.9850이 나왔다. 뒤로 갈수록 성능이 올라감을 알 수 있었다. LSTM과 GRU는 정확도가 비슷하지만 그래도 GRU가 조금 높은 것도 알 수 있었다.   

각 RNN들의 특성과 이들이 어떤 원리로 만들어졌는지 실습을 통해 직접 느낄 수 있었다. 목표했던 정확도인 0.98을 거뜬히 넘겼으니 이쯤에서 끝내보려고 한다.       

끝~
