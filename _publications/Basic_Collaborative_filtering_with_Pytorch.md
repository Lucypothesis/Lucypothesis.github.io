# Bulding Recommender System with PyTorch using Collaborative Filtering
[Youtube 링크](https://www.youtube.com/watch?v=Wj-nkk7dFS8)   
[데이터셋](https://grouplens.org/datasets/movielens/latest/)

**Collaborative Filtering의 종류**
- Item-based
- User-based
- Matrix Factorization


```python
import pandas as pd
import numpy as np
from sklearn import model_selection, metrics, preprocessing
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
```


```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```


```python
df = pd.read_csv('/content/drive/MyDrive/csv_datas/ml-latest-small/ratings.csv')
```


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 100836 entries, 0 to 100835
    Data columns (total 4 columns):
     #   Column     Non-Null Count   Dtype  
    ---  ------     --------------   -----  
     0   userId     100836 non-null  int64  
     1   movieId    100836 non-null  int64  
     2   rating     100836 non-null  float64
     3   timestamp  100836 non-null  int64  
    dtypes: float64(1), int64(3)
    memory usage: 3.1 MB
    


```python
df.userId.nunique()
```




    610




```python
df.movieId.nunique()
```




    9724




```python
df.rating.value_counts()
```




    rating
    4.0    26818
    3.0    20047
    5.0    13211
    3.5    13136
    4.5     8551
    2.0     7551
    2.5     5550
    1.0     2811
    1.5     1791
    0.5     1370
    Name: count, dtype: int64




```python
df.shape
```




    (100836, 4)



## Training Dataset Class Wrapper
이거 하는 이유: 나중에 pytorch data loader를 사용하여 batch training이나 batch processing을 할 수 있기 때문


```python
class MovieDataset:

  # 초기화 함수
  def __init__(self, users, movies, ratings):
    self.users = users
    self.movies = movies
    self.ratings = ratings

  # 데이터셋의 총 길이를 반환함. PyTorch의 DataLoader가 데이터셋의 크기를 알 필요가 있을 때 호출됨
  def __len__(self):
    return len(self.users)

  # 인덱스를 사용하여 users, movies, ratings 리스트에서 각각의 요소를 추출함
  def __getitem__(self,item):
    users = self.users[item]
    movies = self.movies[item]
    ratings = self.ratings[item]

    # 추출된 각 값은 PyTorch의 torch.tensor로 반환됨. 이렇게 변환된 텐서는 딕셔너리 형태로 구성되어 반환됨
    return {
        'users': torch.tensor(users, dtype = torch.long),
        'movies': torch.tensor(movies, dtype = torch.long),
        'ratings': torch.tensor(ratings, dtype = torch.long)
    }
```

## Create the model
사용자와 영화에 대한 임베딩을 생성하고, 이를 결합하여 평점을 예측


```python
class RecSysModel(nn.Module):

  # 초기화 함수
  def __init__(self, n_users, n_movies):
    super().__init__()

    # 각 사용자에 대한 32차원의 임베딩 벡터를 생성하는 임베딩 레이어임
    self.user_embed = nn.Embedding(n_users, 32)
    # 각 영화에 대한 32차원의 임베딩 벡터를 생성하는 임베딩 레이어임
    self.movie_embed = nn.Embedding(n_movies, 32)
    # 사용자와 영화 임베딩을 결합한 후 이를 입력으로 받아서 최종적으로 평점을 예측하는 선형 레이어임. 입력 차원은 64, 출력차원은 1임
    self.out = nn.Linear(64, 1)

  # 순전파 함수
  def forward(self, users, movies, ratings = None) :
    # 사용자ID에 대해 정의된 임베딩 레이어를 통해 각 사용자ID에 해당하는 임베딩 벡터를 추출함
    user_embeds = self.user_embed(users)
    # 영화ID에 대해 정의된 임베딩 레이어를 통해 각 영화 ID에 해당하는 임베딩 벡터를 추출함
    movie_embeds = self.movie_embed(movies)
    # 사용자와 영화 임베딩을 행방향(dim=1)로 결합함. 이렇게 하면 각 샘플에 대해 64차원의 벡터가 생성됨
    output = torch.cat([user_embeds, movie_embeds], dim = 1)

    # 결합된 임베딩 벡터를 선형 레이어에 통과시켜 최종적으로 평점을 예측함
    output = self.out(output)

    return output
```

### 데이터 전처리, 학습 및검증 데이터셋 생성


```python
# LabelEncoder를 사용하여 userId와 movieId를 숫자 인덱스로 변환함. 인덱스 0부터 시작하도록 해서 index out of bound 방지함
lbl_user = preprocessing.LabelEncoder()
lbl_movie = preprocessing.LabelEncoder()
df.userId = lbl_user.fit_transform(df.userId.values)
df.movieId = lbl_movie.fit_transform(df.movieId.values)

df_train, df_valid = model_selection.train_test_split(
    df,
    test_size = 0.1,
    random_state = 42,
    stratify = df.rating.values # 평점분포가 데이터셋에 균일하게 분포되도록 함
)

# 학습 데이터셋 생성
train_dataset = MovieDataset(
    users = df_train.userId.values,
    movies = df_train.movieId.values,
    ratings = df_train.rating.values
)

# 검증 데이터셋 생성
valid_dataset = MovieDataset(
    users = df_valid.userId.values,
    movies = df_valid.movieId.values,
    ratings = df_valid.rating.values
)
```

### Train/Test 데이터셋을 위한 데이터 로더를 설정하고 데이터를 로드함


```python
train_loader = DataLoader(dataset = train_dataset,
                          batch_size = 4,
                          shuffle = True,
                          num_workers = 2)

validation_loader = DataLoader(dataset = valid_dataset,
                               batch_size = 4,
                               shuffle = True,
                               num_workers = 2)

# 학습 데이터 로더를 반복 가능한 객체로 변환함
dataiter = iter(train_loader)
# 반복자에서 다음 데이터 배치를 로드함
dataloader_data = next(dataiter)
# 출력. 출력된 데이터는 MovieDataset의 __getitem__ 메서드에서 반환된 포맷을 따름. 사용자ID, 영화ID, 평점을 포함하는 텐서가 각각 users, movies, ratings 키에 매핑되어있는 딕셔너리 형태임
print(dataloader_data)

### 실행할 때마다 값이 다르게 나오는 이유: shuffle을 True로 설정해놨기 때문
### 이는 각 iteration에서 다른 데이터를 보여줘서 오버피팅을 방지할 수 있음
```

    {'users': tensor([245, 376,  61, 447]), 'movies': tensor([ 257, 5253, 9335, 7266]), 'ratings': tensor([4, 4, 4, 2])}
    

### 추천시스템 모델의 인스턴스 생성, 옵티마이저 및 학습률 스케줄러 설정, 그리고 손실함수 선택 과정을 다룸


```python
# 모델 인스턴스 생성
# 개수를 구하는 이유: 사용자와 영화의 임베딩 레이어 크기를 결정하는데 사용하기 때문. 각 임베딩 레이어는 해당 ID의개수만큼의 임베디 벡터를 생성할 수 있어야 함
model = RecSysModel(
    # 사용자ID의 총 개수
    n_users = len(lbl_user.classes_),
    # 영화ID의 총 개수
    n_movies = len(lbl_movie.classes_)
).to(device) # 생성된 모델을 device(CPU 또는 GPU)로 이동시킴

# Adam 옵티마이저를 사용하여 모델 파라미터를 최적화함.
optimizer = torch.optim.Adam(model.parameters())
# StepLR 학습률 스케줄러를 사용함. step_size = 3: 3에폭마다 학습률을 감소시킴. gamma = 0.7: 각 스텝에서 적용되는 학습률 감소율
sch = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 3, gamma = 0.7)

# 평균 제곱 오차(Mean Squared Error) 손실함수 사용
loss_func = nn.MSELoss()
```


```python
# LabelEncoder로 인코딩된 사용자ID의 고유 클래스 수 출력
print(len(lbl_user.classes_))
# LabelEncoder로 인코딩된 영화ID의 고유 클래스 수 출력
print(len(lbl_movie.classes_))
# LabelEncoder를 통해 변환된 후의 최대 영화ID 인덱스. 영화ID의 인덱스가 0부터 시작하기 때문에 위의 값에서 1뺀 값이어야 함
print(df.movieId.max())
# MovieDataset클래스에 정의된 __len__ 메소드를 호출하여 학습 데이터셋에 포함된 샘플(사용자-영화평점 쌍)의 총 수를 반환함
print(len(train_dataset))
```

    610
    9724
    9723
    90752
    

## Manually run a forward path


```python
# DataLoader에서 로드한 데이터 배치에서 users 키에 해당하는 텐서를 출력함
print(dataloader_data['users'])
print(dataloader_data['users'].size())

# DataLoader에서 로드한 데이터 배치에서 movies키에 해당하는 텐서를 출력함
print(dataloader_data['movies'])
print(dataloader_data['movies'].size())

# 사용자ID를 32차원 벡터로 변환하는 임베딩 레이어를 생성함/ len(lbl_movie.classess_): 전체 고유 사용자 수. 임베딩 레이어에서 필요한 범주의 총 수를 결정함
user_embed = nn.Embedding(len(lbl_user.classes_), 32)
# 위와 같은 방식임
movie_embed = nn.Embedding(len(lbl_movie.classes_), 32)

# 선형 레이어를 생성함. 64차원 벡터(사용자와 영화 임베딩 벡터의 결합)를 입력으로 받아서 1차원의 출력(예측된 평점)을 생성함
out = nn.Linear(64,1)
```

    tensor([245, 376,  61, 447])
    torch.Size([4])
    tensor([ 257, 5253, 9335, 7266])
    torch.Size([4])
    

### 사용자ID와 영화ID에 대한 임베딩을 생성하고 ㅐㅇ성된 임베딩의 크기와 내용을 출력함


```python
# user_embed 레이어를 사용하여 DataLoader로부터 로드된 사용자ID텐서('dataloader_data['users'])를 입력으로 받아 해당 사용자ID에 대응하는 임베딩 벡터를 생성함. user_embed는 각 사용자ID를 32차원의 임베딩 벡터로 변환함aLoader로부터 로드된 사용자ID텐서('dataloader_data['users'])를 입력으로 받아 해당 사용자ID에 대응하는 임베딩 벡터를 생성함. user_embed는 각 사용자ID를 32차원의 임베딩 벡터로 변환함
user_embeds = user_embed(dataloader_data['users'])
movie_embeds = movie_embed(dataloader_data['movies'])

print(f'user_embeds {user_embeds.size()}')
print(f'user_embeds {user_embeds}')
print(f'movie_embeds {movie_embeds.size()}')
print(f'movie_embeds {movie_embeds}')
```

    user_embeds torch.Size([4, 32])
    user_embeds tensor([[ 0.9972,  1.1471,  1.0510,  1.1593, -0.5045, -0.7886, -0.7916,  0.6648,
             -0.0172, -0.4149,  0.4869,  1.1452, -0.9210,  1.8226, -1.1652, -0.0643,
             -1.1265,  1.0010,  0.1568,  1.1607,  0.9161,  1.4166,  0.7506, -1.1820,
             -0.2918,  0.4858, -0.3314, -0.3576, -0.4945, -0.7662,  2.4870,  0.2585],
            [-0.8675,  0.0312, -0.4306,  0.9021,  0.4350,  1.1825,  0.4318, -0.9660,
              0.3669, -0.2780, -0.1800,  1.0910, -1.0789, -0.0485,  0.7096, -1.1817,
              0.3554,  3.1993, -0.8630, -0.8863,  0.0814, -0.9862,  0.1797,  0.1328,
              0.7367,  0.7950, -0.9138,  0.2562, -0.7533, -0.8860, -0.2857,  0.5081],
            [ 0.6140, -0.4573,  0.9458,  1.3676,  0.2016,  0.3161, -1.2050, -0.7713,
             -0.7093,  0.1033, -0.3130, -0.2255, -0.1435,  0.3969,  0.6179, -0.4618,
              0.4102, -0.6804, -1.2770,  0.9675,  0.6469, -0.4416,  2.5120,  1.1907,
              2.3189,  0.9488, -0.7505, -0.5079, -0.4974, -0.3180, -0.1941,  0.6126],
            [ 1.1051,  0.6967, -0.2670, -0.5855, -2.7891,  0.7236, -0.3593, -0.8838,
             -0.6516, -1.2316,  0.3976, -0.4968,  1.3641,  0.9129,  0.7716, -0.9480,
              0.5692,  1.4294,  0.0374, -0.2226,  0.3224,  0.5089, -0.2201, -1.6224,
              0.1487,  0.9389, -0.3826,  1.4507, -1.2348,  1.1690, -0.9254, -0.0622]],
           grad_fn=<EmbeddingBackward0>)
    movie_embeds torch.Size([4, 32])
    movie_embeds tensor([[-0.2683, -0.1485,  0.3031,  0.4896,  1.5291, -0.1814,  2.0929, -0.5594,
             -0.6423,  1.1120, -1.2248,  2.6090,  0.6509, -0.7230, -0.1177,  0.7635,
              0.3928,  0.2621, -0.4949,  0.6138, -0.1979,  0.1623,  0.0918,  1.0199,
              0.2103,  0.7423, -1.8849, -0.6223, -0.6345,  0.3827, -1.7574, -0.5685],
            [-2.5454, -0.5509,  0.7990,  0.5328, -0.5500, -0.5314, -1.3430,  0.1177,
             -0.4099,  0.2175, -0.7498, -0.5712,  1.1677, -1.3241,  0.2202,  0.7208,
             -0.1302, -0.2905,  1.7866, -0.8702, -0.6748,  0.4336, -0.3997,  0.4549,
              1.3000,  0.1518,  0.5365, -0.2342, -1.6119,  0.5198, -0.3108,  0.5051],
            [ 0.6265, -1.2597, -0.7526,  0.4640,  1.2625, -1.5806,  0.2608,  1.2545,
             -0.2247, -1.3814, -1.7891, -0.5819, -0.5736,  0.1790, -0.3070, -0.0688,
             -0.3316,  0.8432,  0.8671,  1.8282, -0.6675,  0.2882, -1.7265, -0.9682,
              1.4441, -0.5980,  0.2734, -0.2154,  1.3500,  0.2147, -0.6708,  1.5020],
            [ 0.7889, -1.3771, -0.5070, -0.9586, -0.1589, -0.1900, -0.1981, -1.9249,
              1.1425, -0.3738,  0.3237,  0.5599,  0.5750, -0.2005, -0.4418, -0.5218,
              0.2039, -0.6625,  0.9713,  0.3378, -1.6134, -1.0685, -0.7873, -1.3767,
             -1.2653, -0.0591,  0.5675,  0.7881, -1.2838, -0.4206,  1.3964, -1.2815]],
           grad_fn=<EmbeddingBackward0>)
    


```python
output = torch.cat([user_embeds, movie_embeds], dim = 1)
print(f'output: {output.size()}')
print(f'output: {output}')
output = out(output)
print(f'output: {output}')
```

    output: torch.Size([4, 64])
    output: tensor([[ 0.9972,  1.1471,  1.0510,  1.1593, -0.5045, -0.7886, -0.7916,  0.6648,
             -0.0172, -0.4149,  0.4869,  1.1452, -0.9210,  1.8226, -1.1652, -0.0643,
             -1.1265,  1.0010,  0.1568,  1.1607,  0.9161,  1.4166,  0.7506, -1.1820,
             -0.2918,  0.4858, -0.3314, -0.3576, -0.4945, -0.7662,  2.4870,  0.2585,
             -0.2683, -0.1485,  0.3031,  0.4896,  1.5291, -0.1814,  2.0929, -0.5594,
             -0.6423,  1.1120, -1.2248,  2.6090,  0.6509, -0.7230, -0.1177,  0.7635,
              0.3928,  0.2621, -0.4949,  0.6138, -0.1979,  0.1623,  0.0918,  1.0199,
              0.2103,  0.7423, -1.8849, -0.6223, -0.6345,  0.3827, -1.7574, -0.5685],
            [-0.8675,  0.0312, -0.4306,  0.9021,  0.4350,  1.1825,  0.4318, -0.9660,
              0.3669, -0.2780, -0.1800,  1.0910, -1.0789, -0.0485,  0.7096, -1.1817,
              0.3554,  3.1993, -0.8630, -0.8863,  0.0814, -0.9862,  0.1797,  0.1328,
              0.7367,  0.7950, -0.9138,  0.2562, -0.7533, -0.8860, -0.2857,  0.5081,
             -2.5454, -0.5509,  0.7990,  0.5328, -0.5500, -0.5314, -1.3430,  0.1177,
             -0.4099,  0.2175, -0.7498, -0.5712,  1.1677, -1.3241,  0.2202,  0.7208,
             -0.1302, -0.2905,  1.7866, -0.8702, -0.6748,  0.4336, -0.3997,  0.4549,
              1.3000,  0.1518,  0.5365, -0.2342, -1.6119,  0.5198, -0.3108,  0.5051],
            [ 0.6140, -0.4573,  0.9458,  1.3676,  0.2016,  0.3161, -1.2050, -0.7713,
             -0.7093,  0.1033, -0.3130, -0.2255, -0.1435,  0.3969,  0.6179, -0.4618,
              0.4102, -0.6804, -1.2770,  0.9675,  0.6469, -0.4416,  2.5120,  1.1907,
              2.3189,  0.9488, -0.7505, -0.5079, -0.4974, -0.3180, -0.1941,  0.6126,
              0.6265, -1.2597, -0.7526,  0.4640,  1.2625, -1.5806,  0.2608,  1.2545,
             -0.2247, -1.3814, -1.7891, -0.5819, -0.5736,  0.1790, -0.3070, -0.0688,
             -0.3316,  0.8432,  0.8671,  1.8282, -0.6675,  0.2882, -1.7265, -0.9682,
              1.4441, -0.5980,  0.2734, -0.2154,  1.3500,  0.2147, -0.6708,  1.5020],
            [ 1.1051,  0.6967, -0.2670, -0.5855, -2.7891,  0.7236, -0.3593, -0.8838,
             -0.6516, -1.2316,  0.3976, -0.4968,  1.3641,  0.9129,  0.7716, -0.9480,
              0.5692,  1.4294,  0.0374, -0.2226,  0.3224,  0.5089, -0.2201, -1.6224,
              0.1487,  0.9389, -0.3826,  1.4507, -1.2348,  1.1690, -0.9254, -0.0622,
              0.7889, -1.3771, -0.5070, -0.9586, -0.1589, -0.1900, -0.1981, -1.9249,
              1.1425, -0.3738,  0.3237,  0.5599,  0.5750, -0.2005, -0.4418, -0.5218,
              0.2039, -0.6625,  0.9713,  0.3378, -1.6134, -1.0685, -0.7873, -1.3767,
             -1.2653, -0.0591,  0.5675,  0.7881, -1.2838, -0.4206,  1.3964, -1.2815]],
           grad_fn=<CatBackward0>)
    output: tensor([[ 0.3648],
            [-0.5742],
            [ 0.4236],
            [-0.4313]], grad_fn=<AddmmBackward0>)
    


```python
with torch.no_grad():
  model_output = model(dataloader_data['users'],
                       dataloader_data['movies'])

  print(f'model_output: {model_output}, size: {model_output.size()}')
```

    model_output: tensor([[-0.3937],
            [-0.7241],
            [ 0.1695],
            [ 0.2989]]), size: torch.Size([4, 1])
    


```python
rating = dataloader_data['ratings']
print(rating)
print(rating.view(4,-1))
print(model_output)

print(rating.sum())

print(model_output.sum() - rating.sum())
```

    tensor([4, 4, 4, 2])
    tensor([[4],
            [4],
            [4],
            [2]])
    tensor([[-0.3937],
            [-0.7241],
            [ 0.1695],
            [ 0.2989]])
    tensor(14)
    tensor(-14.6495)
    

## Run the training loop


```python
epochs = 1
total_loss = 0
plot_steps, print_steps = 5000, 5000
step_cnt = 0
all_losses_list = []

model.train()
for epoch_i in range(epochs):
  for i, train_data in enumerate(train_loader):
    output = model(train_data['users'],
                   train_data['movies']
                   )
    # .view(4,-1) is to reshape the rating to match the shape of model output which is 4x1
    rating = train_data['ratings'].view(4,-1).to(torch.float32)

    loss = loss_func(output, rating)
    total_loss = total_loss + loss.sum().item()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    step_cnt = step_cnt + len(train_data['users'])

    if(step_cnt % plot_steps == 0):
      avg_loss = total_loss / (len(train_data['users']) * plot_steps)
      print(f'epoch {epoch_i} loss at step: {step_cnt} is {avg_loss}')
      all_losses_list.append(avg_loss)
      total_loss = 0 # reset total loss
```

    epoch 0 loss at step: 5000 is 0.5015925409600138
    epoch 0 loss at step: 10000 is 0.1896826016144827
    epoch 0 loss at step: 15000 is 0.11415501520968974
    epoch 0 loss at step: 20000 is 0.08231562353027984
    epoch 0 loss at step: 25000 is 0.07353478641146793
    epoch 0 loss at step: 30000 is 0.06952268821061588
    epoch 0 loss at step: 35000 is 0.0644703435108997
    epoch 0 loss at step: 40000 is 0.06430983757236973
    epoch 0 loss at step: 45000 is 0.061694897446502
    epoch 0 loss at step: 50000 is 0.060891327755898235
    epoch 0 loss at step: 55000 is 0.05821819767151028
    epoch 0 loss at step: 60000 is 0.059701146042067554
    epoch 0 loss at step: 65000 is 0.05877014695010148
    epoch 0 loss at step: 70000 is 0.0572540772087872
    epoch 0 loss at step: 75000 is 0.058639997307054
    epoch 0 loss at step: 80000 is 0.059518357504159215
    epoch 0 loss at step: 85000 is 0.054845110919023866
    epoch 0 loss at step: 90000 is 0.0568829077469185
    


```python
plt.figure()
plt.plot(all_losses_list)
plt.show()
```


    
![png](Basic_Collaborative_filtering_with_Pytorch_files/Basic_Collaborative_filtering_with_Pytorch_30_0.png)
    


## Evaluation with RMSE


```python
from sklearn.metrics import mean_squared_error

model_output_list = []
target_rating_list = []

model.eval()

with torch.no_grad():
  for i, batched_data in enumerate(validation_loader):
    model_output = model(batched_data['users'],
                         batched_data['movies'])
    model_output_list.append(model_output.sum().item() / len(batched_data['users']))

    target_rating = batched_data['ratings']

    target_rating_list.append(target_rating.sum().item() / len(batched_data['users']))

    print(f'model_output: {model_output}, target_rating: {target_rating}')

# squared If True returns MSE value, if False returns RMSE value
rms = mean_squared_error(target_rating_list, model_output_list, squared = False)
print(f'rms: {rms}')
```

    [1;30;43m스트리밍 출력 내용이 길어서 마지막 5000줄이 삭제되었습니다.[0m
            [3.9404],
            [4.0238],
            [2.9720]]), target_rating: tensor([4, 5, 5, 4])
    model_output: tensor([[4.2150],
            [3.4864],
            [3.3795],
            [3.7746]]), target_rating: tensor([4, 3, 3, 3])
    model_output: tensor([[2.9042],
            [2.7182],
            [3.8404],
            [3.2008]]), target_rating: tensor([3, 3, 4, 3])
    model_output: tensor([[2.1233],
            [3.7899],
            [3.4108],
            [4.0566]]), target_rating: tensor([2, 3, 4, 4])
    model_output: tensor([[2.8808],
            [2.9426],
            [3.1347],
            [3.9617]]), target_rating: tensor([4, 4, 3, 5])
    model_output: tensor([[3.2751],
            [3.8918],
            [2.0954],
            [3.6233]]), target_rating: tensor([2, 5, 3, 3])
    model_output: tensor([[3.1150],
            [3.3777],
            [4.2686],
            [3.3495]]), target_rating: tensor([3, 1, 4, 3])
    model_output: tensor([[3.1412],
            [3.8812],
            [3.5321],
            [4.0314]]), target_rating: tensor([4, 4, 4, 5])
    model_output: tensor([[3.7681],
            [3.3929],
            [1.9936],
            [3.2711]]), target_rating: tensor([3, 4, 2, 4])
    model_output: tensor([[3.6805],
            [2.3835],
            [2.1327],
            [3.0692]]), target_rating: tensor([4, 3, 3, 4])
    model_output: tensor([[3.7017],
            [4.2847],
            [3.7753],
            [3.2238]]), target_rating: tensor([4, 4, 3, 4])
    model_output: tensor([[3.1986],
            [2.9470],
            [3.0506],
            [2.6595]]), target_rating: tensor([2, 3, 5, 3])
    model_output: tensor([[3.1786],
            [3.9521],
            [3.1093],
            [3.3802]]), target_rating: tensor([3, 2, 2, 2])
    model_output: tensor([[3.4795],
            [3.5233],
            [3.1585],
            [3.3408]]), target_rating: tensor([3, 3, 0, 4])
    model_output: tensor([[3.4078],
            [1.9574],
            [3.2436],
            [2.7437]]), target_rating: tensor([4, 4, 4, 2])
    model_output: tensor([[4.1494],
            [3.0195],
            [3.0092],
            [3.6171]]), target_rating: tensor([4, 3, 3, 4])
    model_output: tensor([[3.1696],
            [1.7682],
            [3.5234],
            [3.5773]]), target_rating: tensor([4, 2, 2, 4])
    model_output: tensor([[2.7200],
            [3.3489],
            [3.3097],
            [2.9991]]), target_rating: tensor([3, 3, 2, 2])
    model_output: tensor([[3.0422],
            [3.7252],
            [2.9977],
            [3.6054]]), target_rating: tensor([3, 4, 3, 4])
    model_output: tensor([[3.7463],
            [3.5397],
            [3.4775],
            [3.3284]]), target_rating: tensor([4, 3, 4, 2])
    model_output: tensor([[3.0117],
            [4.3165],
            [3.8407],
            [2.9518]]), target_rating: tensor([4, 5, 3, 1])
    model_output: tensor([[3.6098],
            [3.6947],
            [3.0388],
            [3.5900]]), target_rating: tensor([4, 4, 5, 3])
    model_output: tensor([[3.2552],
            [3.7268],
            [3.0196],
            [3.2450]]), target_rating: tensor([4, 4, 3, 3])
    model_output: tensor([[4.1544],
            [2.2670],
            [3.1624],
            [2.8780]]), target_rating: tensor([5, 1, 3, 4])
    model_output: tensor([[3.3768],
            [3.9898],
            [3.5274],
            [3.2822]]), target_rating: tensor([4, 4, 3, 3])
    model_output: tensor([[3.7211],
            [3.0633],
            [2.6593],
            [3.6200]]), target_rating: tensor([3, 3, 3, 5])
    model_output: tensor([[3.3934],
            [3.2619],
            [3.2162],
            [3.5933]]), target_rating: tensor([4, 4, 4, 3])
    model_output: tensor([[3.8078],
            [2.7378],
            [2.9680],
            [4.2638]]), target_rating: tensor([4, 3, 4, 4])
    model_output: tensor([[3.0155],
            [3.1916],
            [3.0153],
            [3.8873]]), target_rating: tensor([4, 5, 4, 4])
    model_output: tensor([[2.2779],
            [2.8103],
            [3.3846],
            [2.2252]]), target_rating: tensor([2, 2, 1, 0])
    model_output: tensor([[4.1362],
            [3.7831],
            [3.5848],
            [3.0728]]), target_rating: tensor([5, 4, 5, 3])
    model_output: tensor([[3.7519],
            [2.6609],
            [4.0984],
            [4.1388]]), target_rating: tensor([4, 3, 5, 5])
    model_output: tensor([[3.8300],
            [3.1198],
            [3.2623],
            [3.1116]]), target_rating: tensor([4, 1, 3, 3])
    model_output: tensor([[3.6732],
            [4.6113],
            [2.9647],
            [2.4806]]), target_rating: tensor([4, 5, 1, 1])
    model_output: tensor([[4.0522],
            [3.0736],
            [3.5214],
            [2.5254]]), target_rating: tensor([5, 4, 4, 3])
    model_output: tensor([[3.4485],
            [3.1401],
            [3.8835],
            [2.8285]]), target_rating: tensor([3, 2, 4, 3])
    model_output: tensor([[3.3985],
            [2.3773],
            [3.5459],
            [3.7522]]), target_rating: tensor([3, 3, 4, 4])
    model_output: tensor([[3.3935],
            [4.1148],
            [3.3057],
            [3.5641]]), target_rating: tensor([3, 5, 3, 3])
    model_output: tensor([[3.4042],
            [3.2804],
            [3.4398],
            [2.5812]]), target_rating: tensor([5, 4, 3, 3])
    model_output: tensor([[2.8575],
            [3.0086],
            [3.4453],
            [3.0225]]), target_rating: tensor([4, 4, 4, 3])
    model_output: tensor([[3.7798],
            [2.9095],
            [2.9943],
            [3.5682]]), target_rating: tensor([4, 3, 4, 4])
    model_output: tensor([[3.1874],
            [3.7535],
            [3.2455],
            [2.5347]]), target_rating: tensor([3, 3, 1, 3])
    model_output: tensor([[2.9776],
            [4.1550],
            [3.0471],
            [3.6718]]), target_rating: tensor([2, 5, 3, 0])
    model_output: tensor([[2.5212],
            [4.2634],
            [3.9193],
            [3.6380]]), target_rating: tensor([2, 3, 5, 3])
    model_output: tensor([[2.9039],
            [3.7629],
            [2.8985],
            [1.8658]]), target_rating: tensor([2, 5, 3, 1])
    model_output: tensor([[3.0445],
            [2.0669],
            [3.3271],
            [3.6288]]), target_rating: tensor([3, 1, 3, 5])
    model_output: tensor([[3.2348],
            [2.9873],
            [3.7657],
            [2.9963]]), target_rating: tensor([3, 3, 4, 2])
    model_output: tensor([[3.4930],
            [2.6448],
            [3.6322],
            [3.3487]]), target_rating: tensor([4, 3, 4, 4])
    model_output: tensor([[3.3663],
            [3.8700],
            [3.8924],
            [2.8778]]), target_rating: tensor([4, 2, 5, 3])
    model_output: tensor([[2.8741],
            [2.4234],
            [3.6923],
            [3.5613]]), target_rating: tensor([2, 1, 4, 4])
    model_output: tensor([[2.7865],
            [2.3949],
            [3.4547],
            [3.8848]]), target_rating: tensor([3, 2, 5, 5])
    model_output: tensor([[3.1838],
            [3.2804],
            [3.9244],
            [3.8195]]), target_rating: tensor([3, 3, 4, 4])
    model_output: tensor([[4.0663],
            [4.1236],
            [2.8368],
            [3.7164]]), target_rating: tensor([5, 4, 3, 4])
    model_output: tensor([[4.1622],
            [3.1479],
            [2.7424],
            [3.3634]]), target_rating: tensor([4, 4, 3, 5])
    model_output: tensor([[3.6632],
            [3.2625],
            [3.6514],
            [3.7246]]), target_rating: tensor([4, 3, 2, 3])
    model_output: tensor([[3.1033],
            [3.9269],
            [3.3346],
            [4.0794]]), target_rating: tensor([3, 3, 3, 4])
    model_output: tensor([[3.6511],
            [3.1389],
            [3.8385],
            [3.4573]]), target_rating: tensor([5, 3, 5, 3])
    model_output: tensor([[2.7954],
            [2.3166],
            [3.4254],
            [3.8472]]), target_rating: tensor([2, 2, 4, 2])
    model_output: tensor([[1.9673],
            [2.9696],
            [3.6441],
            [2.9188]]), target_rating: tensor([1, 4, 3, 3])
    model_output: tensor([[2.3820],
            [3.2191],
            [2.8943],
            [2.5665]]), target_rating: tensor([2, 5, 3, 5])
    model_output: tensor([[2.5952],
            [4.0970],
            [3.8407],
            [3.1570]]), target_rating: tensor([4, 4, 4, 4])
    model_output: tensor([[2.2183],
            [2.7909],
            [2.9758],
            [3.3923]]), target_rating: tensor([1, 4, 3, 4])
    model_output: tensor([[4.1876],
            [2.9795],
            [3.7206],
            [3.5681]]), target_rating: tensor([4, 3, 3, 4])
    model_output: tensor([[2.7966],
            [4.0405],
            [3.7625],
            [2.6290]]), target_rating: tensor([1, 3, 5, 1])
    model_output: tensor([[2.8168],
            [3.7875],
            [3.6026],
            [3.4362]]), target_rating: tensor([2, 4, 3, 4])
    model_output: tensor([[4.2772],
            [4.1460],
            [2.7555],
            [3.0267]]), target_rating: tensor([4, 5, 2, 2])
    model_output: tensor([[3.7284],
            [3.0944],
            [3.3745],
            [3.1079]]), target_rating: tensor([5, 5, 4, 4])
    model_output: tensor([[3.3981],
            [3.4245],
            [2.6633],
            [3.4389]]), target_rating: tensor([2, 4, 1, 3])
    model_output: tensor([[3.6451],
            [3.1455],
            [3.1344],
            [2.2585]]), target_rating: tensor([3, 3, 4, 2])
    model_output: tensor([[3.5358],
            [3.0296],
            [2.0984],
            [2.6394]]), target_rating: tensor([4, 3, 2, 3])
    model_output: tensor([[2.4481],
            [3.3531],
            [3.5479],
            [3.8336]]), target_rating: tensor([2, 4, 4, 3])
    model_output: tensor([[2.9700],
            [3.4295],
            [3.6322],
            [3.2072]]), target_rating: tensor([3, 4, 4, 3])
    model_output: tensor([[3.0281],
            [4.0212],
            [3.5369],
            [2.1134]]), target_rating: tensor([3, 4, 5, 3])
    model_output: tensor([[3.9305],
            [3.6963],
            [2.5937],
            [3.4309]]), target_rating: tensor([5, 2, 2, 4])
    model_output: tensor([[4.7587],
            [3.3821],
            [3.0899],
            [2.7461]]), target_rating: tensor([4, 5, 3, 3])
    model_output: tensor([[3.5937],
            [2.2992],
            [3.2239],
            [2.4885]]), target_rating: tensor([4, 3, 4, 3])
    model_output: tensor([[2.4926],
            [2.9326],
            [3.6685],
            [2.7763]]), target_rating: tensor([2, 3, 4, 3])
    model_output: tensor([[2.9839],
            [3.3031],
            [2.7929],
            [2.2413]]), target_rating: tensor([4, 4, 2, 2])
    model_output: tensor([[3.2322],
            [2.9240],
            [3.1849],
            [3.5479]]), target_rating: tensor([5, 4, 5, 4])
    model_output: tensor([[3.2456],
            [3.7928],
            [3.4083],
            [4.0707]]), target_rating: tensor([4, 4, 1, 4])
    model_output: tensor([[3.8816],
            [2.2017],
            [3.5756],
            [2.6173]]), target_rating: tensor([4, 2, 4, 2])
    model_output: tensor([[4.0989],
            [3.7207],
            [3.1619],
            [2.2586]]), target_rating: tensor([3, 4, 4, 2])
    model_output: tensor([[4.2622],
            [3.9274],
            [3.0422],
            [3.8903]]), target_rating: tensor([5, 4, 3, 4])
    model_output: tensor([[3.5237],
            [3.4122],
            [3.2252],
            [3.4409]]), target_rating: tensor([4, 4, 2, 4])
    model_output: tensor([[3.4851],
            [2.8602],
            [3.0202],
            [4.0735]]), target_rating: tensor([4, 3, 4, 5])
    model_output: tensor([[2.9733],
            [3.2546],
            [3.6421],
            [2.1369]]), target_rating: tensor([3, 2, 2, 0])
    model_output: tensor([[2.8939],
            [2.8802],
            [3.6467],
            [2.7713]]), target_rating: tensor([2, 4, 4, 2])
    model_output: tensor([[3.0720],
            [3.1742],
            [3.5737],
            [2.0626]]), target_rating: tensor([2, 0, 4, 2])
    model_output: tensor([[3.5963],
            [2.7633],
            [2.9430],
            [2.7529]]), target_rating: tensor([4, 1, 2, 3])
    model_output: tensor([[3.2739],
            [2.5674],
            [2.6118],
            [3.1228]]), target_rating: tensor([4, 3, 3, 3])
    model_output: tensor([[3.1928],
            [3.0561],
            [3.2108],
            [3.7787]]), target_rating: tensor([4, 4, 4, 4])
    model_output: tensor([[3.5716],
            [4.1721],
            [3.0766],
            [4.0289]]), target_rating: tensor([4, 4, 3, 3])
    model_output: tensor([[3.3431],
            [4.3186],
            [2.5670],
            [3.3961]]), target_rating: tensor([2, 2, 3, 4])
    model_output: tensor([[4.0736],
            [3.2072],
            [3.3283],
            [2.6915]]), target_rating: tensor([4, 4, 4, 3])
    model_output: tensor([[4.0187],
            [3.2548],
            [4.0879],
            [4.0399]]), target_rating: tensor([5, 2, 5, 5])
    model_output: tensor([[3.4050],
            [3.4242],
            [3.0351],
            [3.4075]]), target_rating: tensor([4, 5, 3, 3])
    model_output: tensor([[3.2855],
            [3.2419],
            [3.6041],
            [4.5709]]), target_rating: tensor([3, 4, 5, 5])
    model_output: tensor([[3.5578],
            [4.0958],
            [3.9231],
            [3.3245]]), target_rating: tensor([4, 2, 4, 5])
    model_output: tensor([[3.3894],
            [2.4656],
            [3.4282],
            [3.7103]]), target_rating: tensor([3, 3, 3, 5])
    model_output: tensor([[2.2939],
            [2.5497],
            [2.6914],
            [3.7883]]), target_rating: tensor([3, 3, 2, 4])
    model_output: tensor([[3.5631],
            [3.5101],
            [3.8866],
            [4.5428]]), target_rating: tensor([4, 4, 4, 3])
    model_output: tensor([[3.4951],
            [3.3775],
            [3.5227],
            [3.3906]]), target_rating: tensor([3, 3, 4, 4])
    model_output: tensor([[3.3366],
            [3.8396],
            [4.0870],
            [2.9386]]), target_rating: tensor([4, 4, 5, 3])
    model_output: tensor([[3.6586],
            [3.4576],
            [3.6432],
            [3.7711]]), target_rating: tensor([3, 5, 4, 5])
    model_output: tensor([[2.9053],
            [3.0785],
            [4.0916],
            [3.3718]]), target_rating: tensor([4, 3, 4, 3])
    model_output: tensor([[3.0542],
            [3.4679],
            [2.8157],
            [3.9525]]), target_rating: tensor([3, 5, 3, 4])
    model_output: tensor([[4.2252],
            [3.4122],
            [3.7757],
            [3.6884]]), target_rating: tensor([4, 2, 4, 3])
    model_output: tensor([[3.7440],
            [3.1019],
            [3.9479],
            [3.4310]]), target_rating: tensor([5, 4, 4, 0])
    model_output: tensor([[2.5225],
            [3.8113],
            [3.0797],
            [3.1705]]), target_rating: tensor([3, 4, 4, 4])
    model_output: tensor([[3.2755],
            [3.2776],
            [3.2613],
            [3.0420]]), target_rating: tensor([3, 0, 4, 2])
    model_output: tensor([[3.9977],
            [3.7009],
            [3.3686],
            [2.9818]]), target_rating: tensor([4, 4, 2, 4])
    model_output: tensor([[2.9536],
            [3.1835],
            [3.6150],
            [3.3577]]), target_rating: tensor([3, 3, 5, 4])
    model_output: tensor([[2.9545],
            [3.8636],
            [3.4778],
            [3.5466]]), target_rating: tensor([2, 5, 3, 3])
    model_output: tensor([[3.5171],
            [3.7978],
            [3.4866],
            [2.9860]]), target_rating: tensor([4, 3, 1, 3])
    model_output: tensor([[3.7886],
            [3.0645],
            [4.1533],
            [3.5276]]), target_rating: tensor([4, 3, 4, 4])
    model_output: tensor([[2.6594],
            [3.6404],
            [3.5624],
            [3.3151]]), target_rating: tensor([3, 4, 3, 2])
    model_output: tensor([[3.2418],
            [3.7649],
            [3.2739],
            [3.1188]]), target_rating: tensor([3, 3, 4, 4])
    model_output: tensor([[4.0971],
            [2.6793],
            [3.0711],
            [3.8853]]), target_rating: tensor([4, 2, 2, 4])
    model_output: tensor([[3.7766],
            [3.5129],
            [3.0136],
            [4.2347]]), target_rating: tensor([4, 3, 4, 5])
    model_output: tensor([[4.4161],
            [3.3182],
            [3.0230],
            [2.9533]]), target_rating: tensor([4, 3, 4, 3])
    model_output: tensor([[4.2087],
            [3.7483],
            [2.7034],
            [3.9372]]), target_rating: tensor([4, 4, 3, 4])
    model_output: tensor([[3.5180],
            [3.6097],
            [3.8404],
            [3.6151]]), target_rating: tensor([3, 4, 4, 5])
    model_output: tensor([[2.1712],
            [3.3108],
            [3.1402],
            [2.8566]]), target_rating: tensor([1, 4, 2, 1])
    model_output: tensor([[2.5132],
            [1.9906],
            [2.9454],
            [3.0763]]), target_rating: tensor([3, 2, 3, 3])
    model_output: tensor([[3.0011],
            [2.8326],
            [4.6452],
            [2.6263]]), target_rating: tensor([4, 5, 5, 3])
    model_output: tensor([[3.3078],
            [3.6709],
            [3.6421],
            [3.2633]]), target_rating: tensor([4, 4, 3, 3])
    model_output: tensor([[3.2915],
            [3.1756],
            [3.2587],
            [2.2502]]), target_rating: tensor([1, 3, 3, 4])
    model_output: tensor([[2.6671],
            [3.2432],
            [2.8635],
            [3.1109]]), target_rating: tensor([3, 3, 4, 3])
    model_output: tensor([[3.5325],
            [3.4346],
            [3.1795],
            [3.0679]]), target_rating: tensor([4, 4, 2, 3])
    model_output: tensor([[3.0342],
            [3.1720],
            [3.6723],
            [3.0798]]), target_rating: tensor([4, 4, 5, 2])
    model_output: tensor([[2.9741],
            [3.7961],
            [3.3820],
            [2.7017]]), target_rating: tensor([3, 3, 4, 3])
    model_output: tensor([[3.4084],
            [2.1771],
            [3.1225],
            [3.1338]]), target_rating: tensor([2, 3, 3, 4])
    model_output: tensor([[3.9453],
            [3.0285],
            [3.2881],
            [4.0982]]), target_rating: tensor([5, 2, 2, 4])
    model_output: tensor([[4.0471],
            [3.2769],
            [3.3411],
            [3.6750]]), target_rating: tensor([4, 1, 4, 3])
    model_output: tensor([[3.0512],
            [2.3269],
            [3.3956],
            [1.9353]]), target_rating: tensor([3, 2, 4, 2])
    model_output: tensor([[2.7298],
            [3.3046],
            [2.8900],
            [2.5596]]), target_rating: tensor([3, 5, 4, 4])
    model_output: tensor([[3.1089],
            [2.3642],
            [3.0227],
            [3.4842]]), target_rating: tensor([1, 2, 4, 4])
    model_output: tensor([[3.0641],
            [3.6458],
            [3.2454],
            [2.9287]]), target_rating: tensor([4, 4, 1, 4])
    model_output: tensor([[3.1769],
            [4.1486],
            [3.6831],
            [2.7053]]), target_rating: tensor([3, 3, 4, 2])
    model_output: tensor([[3.4871],
            [4.3566],
            [3.3993],
            [3.0552]]), target_rating: tensor([4, 4, 2, 3])
    model_output: tensor([[2.7975],
            [3.8002],
            [2.3481],
            [3.4470]]), target_rating: tensor([1, 4, 3, 1])
    model_output: tensor([[3.9575],
            [3.0525],
            [2.6775],
            [4.1652]]), target_rating: tensor([4, 5, 0, 4])
    model_output: tensor([[3.2995],
            [2.7006],
            [2.9630],
            [3.5982]]), target_rating: tensor([2, 3, 4, 4])
    model_output: tensor([[2.9402],
            [4.0092],
            [3.5286],
            [3.7592]]), target_rating: tensor([2, 4, 1, 3])
    model_output: tensor([[2.7635],
            [3.2702],
            [3.0867],
            [2.7409]]), target_rating: tensor([3, 2, 3, 4])
    model_output: tensor([[2.5050],
            [3.9263],
            [3.3078],
            [3.7062]]), target_rating: tensor([3, 4, 3, 4])
    model_output: tensor([[2.6781],
            [3.0639],
            [3.6858],
            [3.2511]]), target_rating: tensor([3, 4, 4, 4])
    model_output: tensor([[3.0990],
            [3.6238],
            [3.1598],
            [3.3269]]), target_rating: tensor([4, 3, 3, 4])
    model_output: tensor([[3.7842],
            [3.5948],
            [4.3514],
            [2.3699]]), target_rating: tensor([4, 3, 5, 4])
    model_output: tensor([[3.1383],
            [3.6293],
            [3.8775],
            [3.0511]]), target_rating: tensor([3, 5, 5, 3])
    model_output: tensor([[3.1659],
            [2.8826],
            [2.8557],
            [3.3612]]), target_rating: tensor([5, 3, 3, 1])
    model_output: tensor([[4.0871],
            [3.2383],
            [3.6427],
            [3.0464]]), target_rating: tensor([4, 3, 3, 3])
    model_output: tensor([[3.6098],
            [3.2755],
            [4.0410],
            [3.3283]]), target_rating: tensor([4, 2, 5, 4])
    model_output: tensor([[2.7195],
            [3.7207],
            [3.4048],
            [3.0639]]), target_rating: tensor([3, 4, 3, 5])
    model_output: tensor([[3.8505],
            [3.4254],
            [3.9616],
            [3.7347]]), target_rating: tensor([4, 4, 4, 3])
    model_output: tensor([[3.7511],
            [3.8116],
            [2.9418],
            [3.8948]]), target_rating: tensor([3, 5, 3, 4])
    model_output: tensor([[2.9649],
            [2.7702],
            [2.2376],
            [2.3106]]), target_rating: tensor([1, 3, 2, 2])
    model_output: tensor([[3.7112],
            [2.6897],
            [2.9654],
            [2.6164]]), target_rating: tensor([4, 4, 3, 1])
    model_output: tensor([[3.3672],
            [3.8000],
            [3.2305],
            [3.7669]]), target_rating: tensor([4, 5, 4, 3])
    model_output: tensor([[3.7332],
            [3.1570],
            [3.1928],
            [3.8306]]), target_rating: tensor([5, 2, 4, 4])
    model_output: tensor([[3.5034],
            [3.6737],
            [1.8798],
            [2.8610]]), target_rating: tensor([3, 4, 0, 3])
    model_output: tensor([[3.7502],
            [3.5281],
            [3.5112],
            [2.1757]]), target_rating: tensor([4, 2, 2, 2])
    model_output: tensor([[2.6019],
            [3.7323],
            [3.1186],
            [3.2026]]), target_rating: tensor([4, 4, 2, 3])
    model_output: tensor([[2.7790],
            [3.7814],
            [3.5525],
            [3.8300]]), target_rating: tensor([3, 3, 5, 3])
    model_output: tensor([[2.6074],
            [2.7000],
            [3.3864],
            [3.8552]]), target_rating: tensor([2, 4, 4, 3])
    model_output: tensor([[3.2889],
            [2.8335],
            [3.5964],
            [4.0633]]), target_rating: tensor([5, 3, 3, 5])
    model_output: tensor([[1.7862],
            [3.0817],
            [3.2374],
            [4.5452]]), target_rating: tensor([0, 3, 3, 4])
    model_output: tensor([[3.5905],
            [3.5363],
            [2.7934],
            [3.8768]]), target_rating: tensor([3, 3, 3, 5])
    model_output: tensor([[3.4383],
            [2.8928],
            [2.1002],
            [3.3534]]), target_rating: tensor([3, 3, 2, 4])
    model_output: tensor([[2.1008],
            [3.2485],
            [2.1040],
            [2.5219]]), target_rating: tensor([2, 4, 2, 1])
    model_output: tensor([[3.4602],
            [1.7022],
            [2.6800],
            [3.2823]]), target_rating: tensor([4, 1, 4, 3])
    model_output: tensor([[2.6754],
            [4.6750],
            [2.8369],
            [4.4930]]), target_rating: tensor([3, 4, 2, 5])
    model_output: tensor([[2.8243],
            [3.5267],
            [2.4793],
            [3.8223]]), target_rating: tensor([2, 4, 2, 2])
    model_output: tensor([[2.9933],
            [4.0214],
            [2.2879],
            [2.6667]]), target_rating: tensor([2, 4, 2, 5])
    model_output: tensor([[3.4954],
            [2.5285],
            [3.0252],
            [4.0376]]), target_rating: tensor([5, 3, 4, 4])
    model_output: tensor([[3.7076],
            [3.6695],
            [3.7042],
            [3.1049]]), target_rating: tensor([4, 4, 3, 2])
    model_output: tensor([[3.0497],
            [3.8813],
            [3.7757],
            [3.2312]]), target_rating: tensor([3, 5, 4, 3])
    model_output: tensor([[2.4900],
            [3.2683],
            [3.4772],
            [2.6620]]), target_rating: tensor([2, 4, 4, 2])
    model_output: tensor([[3.6746],
            [4.1254],
            [3.7335],
            [2.9296]]), target_rating: tensor([5, 3, 5, 4])
    model_output: tensor([[3.3817],
            [2.3794],
            [3.8332],
            [2.6072]]), target_rating: tensor([4, 3, 4, 1])
    model_output: tensor([[3.3452],
            [2.7224],
            [4.3917],
            [3.4953]]), target_rating: tensor([3, 2, 4, 3])
    model_output: tensor([[3.9946],
            [3.9215],
            [2.6566],
            [3.1857]]), target_rating: tensor([4, 5, 2, 3])
    model_output: tensor([[4.3410],
            [3.7961],
            [3.2245],
            [3.3554]]), target_rating: tensor([4, 2, 2, 3])
    model_output: tensor([[3.4123],
            [3.8630],
            [3.3202],
            [3.7341]]), target_rating: tensor([1, 5, 3, 4])
    model_output: tensor([[3.5786],
            [4.2124],
            [3.7060],
            [3.2805]]), target_rating: tensor([4, 4, 1, 5])
    model_output: tensor([[4.0942],
            [3.2724],
            [2.9885],
            [3.4792]]), target_rating: tensor([4, 2, 3, 3])
    model_output: tensor([[3.7176],
            [2.6777],
            [3.0986],
            [2.5678]]), target_rating: tensor([4, 3, 5, 2])
    model_output: tensor([[2.6883],
            [2.5558],
            [2.7833],
            [3.3709]]), target_rating: tensor([3, 1, 4, 4])
    model_output: tensor([[4.4331],
            [3.3966],
            [3.2683],
            [3.9272]]), target_rating: tensor([5, 2, 5, 4])
    model_output: tensor([[2.6960],
            [3.3411],
            [3.1947],
            [3.5422]]), target_rating: tensor([1, 3, 3, 3])
    model_output: tensor([[3.8746],
            [3.5201],
            [3.3504],
            [3.5043]]), target_rating: tensor([5, 5, 3, 3])
    model_output: tensor([[2.7170],
            [3.3210],
            [3.6313],
            [2.9941]]), target_rating: tensor([2, 3, 4, 3])
    model_output: tensor([[3.2131],
            [3.2531],
            [3.2982],
            [2.3097]]), target_rating: tensor([1, 4, 3, 3])
    model_output: tensor([[3.8326],
            [3.7995],
            [3.3550],
            [3.9197]]), target_rating: tensor([4, 5, 3, 4])
    model_output: tensor([[3.3184],
            [2.8873],
            [3.4854],
            [3.0611]]), target_rating: tensor([4, 3, 3, 2])
    model_output: tensor([[3.3087],
            [3.4489],
            [3.9524],
            [4.2690]]), target_rating: tensor([4, 4, 4, 4])
    model_output: tensor([[3.7321],
            [2.4068],
            [4.4530],
            [3.4132]]), target_rating: tensor([3, 0, 4, 4])
    model_output: tensor([[3.7953],
            [2.1296],
            [2.7817],
            [3.5619]]), target_rating: tensor([5, 1, 2, 3])
    model_output: tensor([[3.3364],
            [3.5616],
            [3.1567],
            [3.2091]]), target_rating: tensor([3, 3, 2, 3])
    model_output: tensor([[3.0893],
            [3.5771],
            [3.3708],
            [3.0621]]), target_rating: tensor([4, 4, 3, 3])
    model_output: tensor([[3.1365],
            [3.1666],
            [3.7942],
            [2.8735]]), target_rating: tensor([4, 3, 4, 4])
    model_output: tensor([[3.3863],
            [3.1840],
            [2.7192],
            [2.7056]]), target_rating: tensor([4, 5, 4, 3])
    model_output: tensor([[2.8972],
            [3.5784],
            [4.0698],
            [4.0655]]), target_rating: tensor([3, 4, 4, 4])
    model_output: tensor([[2.6448],
            [3.0281],
            [4.1104],
            [4.2815]]), target_rating: tensor([4, 3, 4, 4])
    model_output: tensor([[2.7280],
            [2.7661],
            [2.9754],
            [3.7258]]), target_rating: tensor([3, 3, 1, 4])
    model_output: tensor([[4.6947],
            [3.5503],
            [3.5405],
            [3.1858]]), target_rating: tensor([4, 4, 4, 2])
    model_output: tensor([[4.0124],
            [4.0239],
            [3.2078],
            [3.6071]]), target_rating: tensor([5, 3, 3, 4])
    model_output: tensor([[4.3458],
            [3.6699],
            [4.2462],
            [3.9716]]), target_rating: tensor([4, 5, 5, 5])
    model_output: tensor([[3.3765],
            [2.4017],
            [3.4645],
            [3.4362]]), target_rating: tensor([3, 3, 2, 4])
    model_output: tensor([[3.2807],
            [2.8295],
            [3.4681],
            [3.6033]]), target_rating: tensor([3, 4, 2, 3])
    model_output: tensor([[3.0431],
            [3.5472],
            [3.5909],
            [1.8953]]), target_rating: tensor([3, 4, 4, 2])
    model_output: tensor([[3.4387],
            [3.6293],
            [2.4826],
            [2.4795]]), target_rating: tensor([3, 4, 2, 0])
    model_output: tensor([[2.8477],
            [2.7090],
            [2.9256],
            [4.1771]]), target_rating: tensor([3, 3, 1, 4])
    model_output: tensor([[3.8565],
            [3.2258],
            [3.4162],
            [2.0220]]), target_rating: tensor([3, 4, 3, 1])
    model_output: tensor([[2.6237],
            [2.5824],
            [3.3109],
            [3.6539]]), target_rating: tensor([2, 3, 3, 4])
    model_output: tensor([[3.9359],
            [3.5781],
            [3.1673],
            [2.9236]]), target_rating: tensor([4, 4, 3, 4])
    model_output: tensor([[3.5983],
            [3.7292],
            [3.4434],
            [3.4434]]), target_rating: tensor([4, 3, 4, 4])
    model_output: tensor([[3.1681],
            [3.7874],
            [3.6366],
            [3.2199]]), target_rating: tensor([3, 4, 2, 3])
    model_output: tensor([[2.4320],
            [3.1878],
            [3.6995],
            [4.4580]]), target_rating: tensor([0, 3, 4, 5])
    model_output: tensor([[3.1336],
            [3.1839],
            [3.0929],
            [4.3437]]), target_rating: tensor([3, 3, 3, 5])
    model_output: tensor([[2.9234],
            [2.6976],
            [3.2616],
            [3.9105]]), target_rating: tensor([3, 3, 3, 3])
    model_output: tensor([[3.6258],
            [2.8900],
            [3.8279],
            [2.5105]]), target_rating: tensor([3, 3, 2, 1])
    model_output: tensor([[4.1125],
            [3.4067],
            [3.4336],
            [4.5937]]), target_rating: tensor([4, 3, 4, 4])
    model_output: tensor([[3.6147],
            [2.9286],
            [3.1633],
            [2.8586]]), target_rating: tensor([4, 4, 2, 3])
    model_output: tensor([[3.2279],
            [3.1780],
            [3.2156],
            [4.1156]]), target_rating: tensor([4, 4, 3, 4])
    model_output: tensor([[2.5359],
            [2.8292],
            [2.5442],
            [4.3636]]), target_rating: tensor([2, 2, 2, 4])
    model_output: tensor([[3.7234],
            [3.4005],
            [2.6188],
            [3.1963]]), target_rating: tensor([5, 3, 1, 5])
    model_output: tensor([[2.6031],
            [3.6573],
            [2.7868],
            [3.1091]]), target_rating: tensor([4, 4, 3, 0])
    model_output: tensor([[3.8156],
            [2.5742],
            [3.5476],
            [2.7477]]), target_rating: tensor([3, 0, 4, 2])
    model_output: tensor([[3.3855],
            [3.4318],
            [3.2656],
            [2.6435]]), target_rating: tensor([2, 5, 2, 2])
    model_output: tensor([[3.5168],
            [2.8398],
            [3.4805],
            [3.6044]]), target_rating: tensor([4, 2, 4, 4])
    model_output: tensor([[4.0906],
            [3.0305],
            [3.2713],
            [3.4071]]), target_rating: tensor([5, 4, 1, 4])
    model_output: tensor([[2.2670],
            [3.7477],
            [2.0263],
            [3.6883]]), target_rating: tensor([0, 4, 2, 3])
    model_output: tensor([[2.3099],
            [3.5162],
            [3.0956],
            [3.6018]]), target_rating: tensor([2, 5, 3, 2])
    model_output: tensor([[4.0107],
            [3.5292],
            [3.7836],
            [2.7016]]), target_rating: tensor([5, 3, 3, 2])
    model_output: tensor([[3.5222],
            [3.5177],
            [2.7625],
            [2.8520]]), target_rating: tensor([5, 4, 4, 2])
    model_output: tensor([[3.5869],
            [3.7281],
            [3.2041],
            [3.0676]]), target_rating: tensor([3, 3, 3, 2])
    model_output: tensor([[3.4055],
            [3.4838],
            [3.1457],
            [3.3759]]), target_rating: tensor([2, 4, 3, 5])
    model_output: tensor([[3.7507],
            [2.1628],
            [3.3853],
            [3.4527]]), target_rating: tensor([4, 1, 4, 4])
    model_output: tensor([[3.5304],
            [3.2138],
            [3.7003],
            [3.2157]]), target_rating: tensor([3, 4, 2, 2])
    model_output: tensor([[3.3090],
            [2.9513],
            [3.1131],
            [3.6846]]), target_rating: tensor([3, 5, 1, 4])
    model_output: tensor([[3.2503],
            [2.6105],
            [3.6704],
            [3.0040]]), target_rating: tensor([1, 3, 2, 2])
    model_output: tensor([[4.0205],
            [2.9790],
            [2.7734],
            [3.2819]]), target_rating: tensor([3, 4, 3, 4])
    model_output: tensor([[3.1427],
            [4.2240],
            [3.8801],
            [3.6429]]), target_rating: tensor([4, 5, 3, 4])
    model_output: tensor([[3.4957],
            [2.2307],
            [2.3401],
            [3.5502]]), target_rating: tensor([4, 1, 2, 5])
    model_output: tensor([[3.0030],
            [2.2484],
            [2.3545],
            [3.3640]]), target_rating: tensor([4, 3, 3, 3])
    model_output: tensor([[3.5904],
            [3.5846],
            [3.3970],
            [2.2155]]), target_rating: tensor([4, 3, 3, 2])
    model_output: tensor([[1.7484],
            [4.5071],
            [3.8522],
            [3.5660]]), target_rating: tensor([0, 4, 5, 3])
    model_output: tensor([[2.8776],
            [2.7622],
            [4.1359],
            [2.2169]]), target_rating: tensor([3, 2, 4, 2])
    model_output: tensor([[3.5784],
            [3.5546],
            [2.4236],
            [2.4407]]), target_rating: tensor([3, 4, 2, 3])
    model_output: tensor([[2.0761],
            [2.4106],
            [3.5983],
            [3.6875]]), target_rating: tensor([4, 2, 3, 4])
    model_output: tensor([[3.5768],
            [3.1065],
            [2.4096],
            [3.9390]]), target_rating: tensor([4, 3, 1, 4])
    model_output: tensor([[2.1244],
            [4.0828],
            [3.6377],
            [3.3053]]), target_rating: tensor([0, 5, 4, 4])
    model_output: tensor([[3.0998],
            [3.7961],
            [3.1134],
            [3.0151]]), target_rating: tensor([3, 4, 3, 2])
    model_output: tensor([[2.6979],
            [3.6268],
            [3.0737],
            [4.1595]]), target_rating: tensor([4, 3, 5, 4])
    model_output: tensor([[3.5766],
            [3.0020],
            [2.8982],
            [4.1077]]), target_rating: tensor([3, 3, 3, 3])
    model_output: tensor([[3.0954],
            [3.1603],
            [2.5663],
            [3.6595]]), target_rating: tensor([4, 4, 1, 4])
    model_output: tensor([[3.7672],
            [3.2599],
            [2.8745],
            [3.1914]]), target_rating: tensor([5, 4, 1, 4])
    model_output: tensor([[3.4841],
            [3.2410],
            [2.9234],
            [2.7817]]), target_rating: tensor([4, 4, 1, 2])
    model_output: tensor([[3.0594],
            [3.4181],
            [3.5350],
            [3.8737]]), target_rating: tensor([3, 4, 4, 4])
    model_output: tensor([[3.5729],
            [2.8699],
            [4.2519],
            [3.1237]]), target_rating: tensor([3, 2, 4, 4])
    model_output: tensor([[3.8499],
            [3.3972],
            [4.5144],
            [3.2957]]), target_rating: tensor([2, 4, 5, 0])
    model_output: tensor([[1.9654],
            [2.6104],
            [3.2855],
            [3.7366]]), target_rating: tensor([2, 2, 1, 1])
    model_output: tensor([[3.0899],
            [4.0505],
            [2.5243],
            [3.9877]]), target_rating: tensor([4, 3, 3, 5])
    model_output: tensor([[3.1294],
            [3.4286],
            [3.3495],
            [3.1014]]), target_rating: tensor([5, 4, 4, 1])
    model_output: tensor([[2.7490],
            [2.4244],
            [2.0187],
            [3.3159]]), target_rating: tensor([2, 2, 3, 4])
    model_output: tensor([[4.1014],
            [1.9024],
            [2.6596],
            [4.7043]]), target_rating: tensor([4, 5, 3, 5])
    model_output: tensor([[3.5566],
            [3.5141],
            [3.8755],
            [3.6207]]), target_rating: tensor([3, 3, 3, 3])
    model_output: tensor([[3.7713],
            [3.5980],
            [3.8261],
            [3.6093]]), target_rating: tensor([3, 4, 3, 4])
    model_output: tensor([[3.5003],
            [3.5109],
            [3.7510],
            [3.9439]]), target_rating: tensor([4, 3, 4, 4])
    model_output: tensor([[2.9636],
            [3.3916],
            [2.8527],
            [3.1548]]), target_rating: tensor([3, 3, 3, 3])
    model_output: tensor([[2.5746],
            [3.0876],
            [3.2053],
            [2.8220]]), target_rating: tensor([2, 3, 4, 3])
    model_output: tensor([[3.3134],
            [2.9091],
            [3.3993],
            [3.2680]]), target_rating: tensor([5, 3, 3, 3])
    model_output: tensor([[2.5171],
            [2.9135],
            [3.1772],
            [2.9223]]), target_rating: tensor([3, 1, 4, 4])
    model_output: tensor([[2.6930],
            [2.8492],
            [3.6556],
            [3.3881]]), target_rating: tensor([4, 3, 4, 3])
    model_output: tensor([[3.3500],
            [2.4775],
            [3.1975],
            [2.1410]]), target_rating: tensor([4, 3, 3, 2])
    model_output: tensor([[3.8146],
            [3.0469],
            [2.8719],
            [3.3978]]), target_rating: tensor([4, 3, 4, 4])
    model_output: tensor([[3.2684],
            [3.3220],
            [3.0083],
            [1.8604]]), target_rating: tensor([5, 2, 3, 0])
    model_output: tensor([[3.8918],
            [3.9088],
            [3.6067],
            [3.8057]]), target_rating: tensor([4, 3, 4, 5])
    model_output: tensor([[4.6019],
            [3.6961],
            [2.8348],
            [4.2513]]), target_rating: tensor([5, 4, 3, 4])
    model_output: tensor([[3.8964],
            [3.3057],
            [3.2822],
            [3.0294]]), target_rating: tensor([4, 3, 4, 3])
    model_output: tensor([[2.7481],
            [3.6820],
            [2.4333],
            [3.8561]]), target_rating: tensor([3, 2, 2, 4])
    model_output: tensor([[3.6022],
            [3.8224],
            [3.4809],
            [3.2770]]), target_rating: tensor([4, 5, 4, 3])
    model_output: tensor([[2.9162],
            [3.1757],
            [2.9198],
            [3.0338]]), target_rating: tensor([4, 3, 3, 3])
    model_output: tensor([[2.8550],
            [3.8618],
            [3.1712],
            [4.4582]]), target_rating: tensor([5, 3, 4, 4])
    model_output: tensor([[3.3528],
            [3.0581],
            [3.8276],
            [3.3861]]), target_rating: tensor([3, 3, 4, 4])
    model_output: tensor([[3.4408],
            [3.1286],
            [3.9081],
            [3.4032]]), target_rating: tensor([5, 3, 5, 4])
    model_output: tensor([[2.9996],
            [3.9312],
            [3.1727],
            [3.0543]]), target_rating: tensor([3, 3, 2, 3])
    model_output: tensor([[4.1365],
            [3.1098],
            [3.4068],
            [2.9762]]), target_rating: tensor([5, 3, 1, 4])
    model_output: tensor([[2.6696],
            [3.2474],
            [2.9742],
            [3.2863]]), target_rating: tensor([1, 3, 4, 4])
    model_output: tensor([[3.8998],
            [2.8413],
            [4.3328],
            [3.4176]]), target_rating: tensor([5, 4, 2, 3])
    model_output: tensor([[3.1558],
            [2.2823],
            [3.3388],
            [2.9577]]), target_rating: tensor([3, 1, 3, 3])
    model_output: tensor([[2.9814],
            [3.1468],
            [1.9224],
            [3.6986]]), target_rating: tensor([3, 4, 3, 3])
    model_output: tensor([[3.7453],
            [2.9734],
            [3.1829],
            [3.3705]]), target_rating: tensor([5, 4, 2, 4])
    model_output: tensor([[3.4836],
            [3.8926],
            [3.1088],
            [3.6623]]), target_rating: tensor([4, 5, 1, 3])
    model_output: tensor([[3.5788],
            [3.2141],
            [3.5309],
            [3.2898]]), target_rating: tensor([3, 3, 4, 4])
    model_output: tensor([[3.6413],
            [3.4852],
            [4.6125],
            [3.8443]]), target_rating: tensor([4, 3, 5, 4])
    model_output: tensor([[3.1366],
            [3.0392],
            [3.4802],
            [3.7408]]), target_rating: tensor([3, 4, 4, 5])
    model_output: tensor([[2.6643],
            [3.7987],
            [3.6529],
            [3.3383]]), target_rating: tensor([4, 4, 3, 3])
    model_output: tensor([[3.3890],
            [3.1805],
            [2.7963],
            [3.1935]]), target_rating: tensor([4, 3, 3, 4])
    model_output: tensor([[3.6315],
            [3.3501],
            [3.7526],
            [3.6869]]), target_rating: tensor([4, 3, 4, 4])
    model_output: tensor([[3.2245],
            [2.5727],
            [2.6633],
            [4.0992]]), target_rating: tensor([3, 3, 4, 5])
    model_output: tensor([[3.0789],
            [3.9235],
            [3.3482],
            [3.3324]]), target_rating: tensor([0, 4, 4, 3])
    model_output: tensor([[3.2011],
            [3.2069],
            [3.1688],
            [4.6390]]), target_rating: tensor([3, 4, 3, 3])
    model_output: tensor([[4.0434],
            [2.8017],
            [3.6462],
            [3.7407]]), target_rating: tensor([5, 3, 4, 5])
    model_output: tensor([[2.9767],
            [2.9364],
            [3.3449],
            [4.6877]]), target_rating: tensor([3, 4, 5, 3])
    model_output: tensor([[3.4386],
            [2.9524],
            [3.2499],
            [2.7586]]), target_rating: tensor([3, 4, 4, 2])
    model_output: tensor([[3.9216],
            [3.9732],
            [3.9105],
            [3.6706]]), target_rating: tensor([4, 5, 1, 4])
    model_output: tensor([[3.5000],
            [2.9941],
            [4.0470],
            [2.6833]]), target_rating: tensor([4, 5, 5, 3])
    model_output: tensor([[4.2133],
            [2.7147],
            [2.9133],
            [2.3699]]), target_rating: tensor([4, 4, 4, 1])
    model_output: tensor([[3.2436],
            [3.2816],
            [4.4293],
            [3.3322]]), target_rating: tensor([5, 3, 5, 1])
    model_output: tensor([[2.7991],
            [3.4787],
            [2.9652],
            [3.2716]]), target_rating: tensor([2, 3, 4, 4])
    model_output: tensor([[2.9766],
            [2.9961],
            [3.0013],
            [3.4601]]), target_rating: tensor([3, 3, 5, 0])
    model_output: tensor([[3.2232],
            [1.9076],
            [3.7714],
            [3.5390]]), target_rating: tensor([3, 2, 4, 4])
    model_output: tensor([[3.1375],
            [3.4923],
            [3.1367],
            [3.2599]]), target_rating: tensor([4, 3, 3, 3])
    model_output: tensor([[3.7820],
            [3.3921],
            [3.7224],
            [2.3510]]), target_rating: tensor([3, 3, 3, 1])
    model_output: tensor([[3.1827],
            [4.1038],
            [3.4621],
            [2.8728]]), target_rating: tensor([4, 4, 4, 4])
    model_output: tensor([[3.2365],
            [3.4628],
            [3.4777],
            [3.8025]]), target_rating: tensor([4, 3, 4, 4])
    model_output: tensor([[2.7213],
            [3.0060],
            [3.9215],
            [3.2983]]), target_rating: tensor([3, 3, 5, 4])
    model_output: tensor([[3.8399],
            [3.3727],
            [3.9262],
            [3.1172]]), target_rating: tensor([4, 3, 4, 2])
    model_output: tensor([[3.6007],
            [3.0850],
            [2.5773],
            [4.0883]]), target_rating: tensor([5, 2, 2, 5])
    model_output: tensor([[2.4673],
            [3.3056],
            [2.8641],
            [2.3516]]), target_rating: tensor([3, 3, 2, 1])
    model_output: tensor([[3.5057],
            [2.8016],
            [2.7042],
            [3.5332]]), target_rating: tensor([4, 2, 3, 3])
    model_output: tensor([[2.4301],
            [4.0483],
            [2.9700],
            [3.5539]]), target_rating: tensor([2, 4, 3, 5])
    model_output: tensor([[4.1336],
            [3.7120],
            [2.5558],
            [3.2733]]), target_rating: tensor([3, 5, 1, 3])
    model_output: tensor([[3.6952],
            [2.7595],
            [2.9260],
            [3.0838]]), target_rating: tensor([3, 1, 4, 2])
    model_output: tensor([[3.5112],
            [3.6979],
            [2.8962],
            [3.5835]]), target_rating: tensor([5, 5, 3, 3])
    model_output: tensor([[3.8746],
            [3.1654],
            [3.8250],
            [3.0433]]), target_rating: tensor([4, 4, 4, 3])
    model_output: tensor([[3.9372],
            [3.7655],
            [3.1186],
            [2.7255]]), target_rating: tensor([4, 3, 4, 2])
    model_output: tensor([[2.3786],
            [2.6305],
            [2.9495],
            [2.6275]]), target_rating: tensor([0, 3, 3, 1])
    model_output: tensor([[3.5370],
            [2.8880],
            [4.1823],
            [3.9254]]), target_rating: tensor([3, 2, 4, 3])
    model_output: tensor([[3.0389],
            [3.3814],
            [2.4826],
            [3.2148]]), target_rating: tensor([3, 2, 4, 4])
    model_output: tensor([[3.7073],
            [3.7527],
            [3.4898],
            [4.0855]]), target_rating: tensor([5, 3, 3, 4])
    model_output: tensor([[3.4749],
            [2.8571],
            [4.1018],
            [3.0031]]), target_rating: tensor([3, 4, 0, 1])
    model_output: tensor([[3.8800],
            [3.0136],
            [3.4627],
            [3.9198]]), target_rating: tensor([4, 3, 4, 4])
    model_output: tensor([[4.3126],
            [3.3194],
            [3.6373],
            [2.9259]]), target_rating: tensor([2, 2, 4, 2])
    model_output: tensor([[3.1083],
            [1.8858],
            [2.3139],
            [2.8899]]), target_rating: tensor([3, 3, 3, 4])
    model_output: tensor([[2.8101],
            [2.0604],
            [3.0745],
            [3.1579]]), target_rating: tensor([3, 2, 3, 3])
    model_output: tensor([[3.8836],
            [3.7050],
            [2.0378],
            [3.3136]]), target_rating: tensor([4, 5, 2, 4])
    model_output: tensor([[2.8641],
            [2.7036],
            [4.4406],
            [3.8494]]), target_rating: tensor([3, 2, 4, 4])
    model_output: tensor([[3.4749],
            [3.5362],
            [2.8248],
            [3.0292]]), target_rating: tensor([4, 2, 3, 1])
    model_output: tensor([[3.9914],
            [3.8314],
            [3.2607],
            [2.3740]]), target_rating: tensor([4, 5, 5, 3])
    model_output: tensor([[2.6227],
            [2.9397],
            [3.0594],
            [2.9094]]), target_rating: tensor([3, 2, 3, 2])
    model_output: tensor([[3.1847],
            [3.1728],
            [3.4328],
            [3.5066]]), target_rating: tensor([5, 5, 4, 3])
    model_output: tensor([[3.7934],
            [3.7398],
            [4.4984],
            [2.6416]]), target_rating: tensor([5, 4, 5, 1])
    model_output: tensor([[3.8542],
            [3.4391],
            [3.1390],
            [3.3118]]), target_rating: tensor([5, 4, 3, 4])
    model_output: tensor([[3.6202],
            [3.5325],
            [4.3945],
            [3.5129]]), target_rating: tensor([4, 4, 5, 5])
    model_output: tensor([[3.3212],
            [3.7195],
            [2.9307],
            [3.1360]]), target_rating: tensor([4, 3, 5, 3])
    model_output: tensor([[3.6911],
            [3.3163],
            [2.8070],
            [3.1880]]), target_rating: tensor([4, 4, 3, 3])
    model_output: tensor([[2.8505],
            [2.9142],
            [3.1797],
            [3.2374]]), target_rating: tensor([1, 3, 3, 3])
    model_output: tensor([[3.0032],
            [3.4562],
            [3.7176],
            [2.7018]]), target_rating: tensor([3, 4, 5, 3])
    model_output: tensor([[3.4425],
            [3.9252],
            [3.8374],
            [3.5379]]), target_rating: tensor([4, 5, 4, 4])
    model_output: tensor([[3.9697],
            [2.1100],
            [4.9822],
            [3.3284]]), target_rating: tensor([4, 2, 5, 2])
    model_output: tensor([[2.4062],
            [2.9752],
            [3.6614],
            [3.4899]]), target_rating: tensor([4, 4, 5, 3])
    model_output: tensor([[3.6338],
            [3.4863],
            [2.3966],
            [3.6325]]), target_rating: tensor([2, 4, 3, 4])
    model_output: tensor([[3.2023],
            [2.7367],
            [3.4031],
            [3.5265]]), target_rating: tensor([4, 2, 3, 4])
    model_output: tensor([[3.7051],
            [3.4130],
            [3.4492],
            [3.9807]]), target_rating: tensor([5, 1, 2, 5])
    model_output: tensor([[3.5316],
            [3.2423],
            [3.3868],
            [3.2170]]), target_rating: tensor([5, 2, 4, 4])
    model_output: tensor([[3.8532],
            [3.6122],
            [2.9503],
            [4.0773]]), target_rating: tensor([3, 3, 0, 3])
    model_output: tensor([[3.6598],
            [3.4814],
            [2.8570],
            [2.9114]]), target_rating: tensor([4, 4, 3, 4])
    model_output: tensor([[3.1067],
            [2.9775],
            [3.2704],
            [3.4742]]), target_rating: tensor([5, 3, 2, 4])
    model_output: tensor([[2.4012],
            [3.1447],
            [2.9001],
            [3.3774]]), target_rating: tensor([0, 3, 3, 4])
    model_output: tensor([[4.5720],
            [3.2278],
            [2.9234],
            [2.6673]]), target_rating: tensor([5, 3, 3, 3])
    model_output: tensor([[4.0125],
            [3.7446],
            [3.7923],
            [3.0492]]), target_rating: tensor([4, 4, 3, 3])
    model_output: tensor([[3.5964],
            [2.4229],
            [3.3134],
            [3.1326]]), target_rating: tensor([3, 0, 4, 4])
    model_output: tensor([[2.2490],
            [3.2084],
            [2.4203],
            [2.7889]]), target_rating: tensor([1, 4, 3, 1])
    model_output: tensor([[2.3046],
            [3.2342],
            [2.8848],
            [3.3137]]), target_rating: tensor([3, 2, 3, 2])
    model_output: tensor([[3.3076],
            [4.1439],
            [3.7723],
            [3.7346]]), target_rating: tensor([3, 4, 5, 5])
    model_output: tensor([[3.1607],
            [2.5871],
            [2.0957],
            [3.4630]]), target_rating: tensor([4, 3, 2, 4])
    model_output: tensor([[2.6516],
            [4.7381],
            [3.3721],
            [3.0492]]), target_rating: tensor([3, 5, 3, 4])
    model_output: tensor([[2.7202],
            [4.1853],
            [3.6989],
            [3.4368]]), target_rating: tensor([1, 4, 5, 3])
    model_output: tensor([[2.8185],
            [3.1857],
            [3.7555],
            [3.1803]]), target_rating: tensor([3, 4, 2, 3])
    model_output: tensor([[2.8821],
            [3.6730],
            [2.7857],
            [3.6800]]), target_rating: tensor([3, 4, 2, 4])
    model_output: tensor([[3.0583],
            [2.7319],
            [2.5550],
            [4.5897]]), target_rating: tensor([5, 3, 3, 5])
    model_output: tensor([[3.5412],
            [3.4773],
            [3.8916],
            [3.3920]]), target_rating: tensor([4, 4, 5, 4])
    model_output: tensor([[1.7472],
            [3.1908],
            [3.3497],
            [3.3596]]), target_rating: tensor([3, 2, 3, 2])
    model_output: tensor([[2.0889],
            [3.3667],
            [2.4346],
            [2.6735]]), target_rating: tensor([2, 4, 3, 2])
    model_output: tensor([[3.2671],
            [2.8435],
            [2.9503],
            [3.3909]]), target_rating: tensor([3, 4, 1, 3])
    model_output: tensor([[3.4841],
            [2.6397],
            [3.7965],
            [3.6626]]), target_rating: tensor([3, 1, 5, 3])
    model_output: tensor([[2.9518],
            [3.3589],
            [3.7331],
            [3.1140]]), target_rating: tensor([2, 4, 4, 3])
    model_output: tensor([[3.3904],
            [3.5113],
            [3.0875],
            [3.3432]]), target_rating: tensor([4, 4, 3, 3])
    model_output: tensor([[3.4834],
            [2.8342],
            [2.5460],
            [3.4962]]), target_rating: tensor([3, 3, 2, 4])
    model_output: tensor([[3.1664],
            [2.7092],
            [2.5669],
            [3.6504]]), target_rating: tensor([1, 3, 1, 5])
    model_output: tensor([[1.7447],
            [4.0429],
            [3.5326],
            [2.7277]]), target_rating: tensor([2, 3, 1, 3])
    model_output: tensor([[3.5060],
            [2.8331],
            [3.6073],
            [2.8569]]), target_rating: tensor([3, 4, 3, 4])
    model_output: tensor([[2.8368],
            [3.0766],
            [2.7657],
            [3.6098]]), target_rating: tensor([4, 3, 3, 4])
    model_output: tensor([[3.1347],
            [4.3928],
            [2.6187],
            [3.4630]]), target_rating: tensor([4, 5, 2, 4])
    model_output: tensor([[3.2791],
            [3.4850],
            [3.1661],
            [3.3444]]), target_rating: tensor([3, 5, 4, 5])
    model_output: tensor([[2.9210],
            [3.5540],
            [3.6680],
            [4.2121]]), target_rating: tensor([3, 3, 5, 5])
    model_output: tensor([[4.2093],
            [3.5308],
            [3.2671],
            [2.5613]]), target_rating: tensor([4, 4, 3, 2])
    model_output: tensor([[3.6227],
            [3.2194],
            [3.3109],
            [2.6927]]), target_rating: tensor([5, 5, 3, 2])
    model_output: tensor([[4.3541],
            [3.2431],
            [2.7657],
            [3.2452]]), target_rating: tensor([5, 4, 2, 4])
    model_output: tensor([[2.7057],
            [2.8170],
            [3.1341],
            [4.1615]]), target_rating: tensor([3, 3, 4, 4])
    model_output: tensor([[2.2584],
            [2.7843],
            [2.8219],
            [2.2559]]), target_rating: tensor([3, 2, 4, 4])
    model_output: tensor([[3.2665],
            [3.5624],
            [3.7432],
            [3.6026]]), target_rating: tensor([4, 4, 3, 3])
    model_output: tensor([[2.9553],
            [3.5753],
            [4.1657],
            [2.7087]]), target_rating: tensor([3, 3, 5, 2])
    model_output: tensor([[4.2013],
            [3.7192],
            [3.0466],
            [4.0783]]), target_rating: tensor([5, 5, 4, 4])
    model_output: tensor([[3.4686],
            [3.4023],
            [4.5459],
            [2.8493]]), target_rating: tensor([4, 3, 4, 4])
    model_output: tensor([[3.7173],
            [3.4765],
            [3.7101],
            [3.4657]]), target_rating: tensor([4, 3, 4, 4])
    model_output: tensor([[3.2480],
            [3.4906],
            [3.0183],
            [3.2704]]), target_rating: tensor([4, 3, 4, 3])
    model_output: tensor([[3.2935],
            [3.5260],
            [3.3845],
            [2.9268]]), target_rating: tensor([2, 2, 4, 4])
    model_output: tensor([[4.5321],
            [3.3041],
            [3.3506],
            [3.1439]]), target_rating: tensor([5, 4, 5, 3])
    model_output: tensor([[3.4119],
            [2.9351],
            [2.8724],
            [3.7411]]), target_rating: tensor([4, 3, 4, 1])
    model_output: tensor([[2.5385],
            [2.9582],
            [3.0131],
            [3.4473]]), target_rating: tensor([4, 3, 4, 4])
    model_output: tensor([[2.7613],
            [3.2586],
            [4.1070],
            [2.2918]]), target_rating: tensor([1, 4, 5, 2])
    model_output: tensor([[3.5331],
            [3.0341],
            [4.0476],
            [3.2496]]), target_rating: tensor([3, 3, 3, 3])
    model_output: tensor([[3.5847],
            [3.6662],
            [2.3896],
            [3.5543]]), target_rating: tensor([4, 3, 4, 4])
    model_output: tensor([[2.9905],
            [2.0593],
            [3.6197],
            [3.7753]]), target_rating: tensor([3, 2, 4, 5])
    model_output: tensor([[3.8740],
            [3.0992],
            [2.9602],
            [3.3939]]), target_rating: tensor([4, 3, 2, 4])
    model_output: tensor([[2.4643],
            [3.2977],
            [2.1657],
            [3.4858]]), target_rating: tensor([2, 4, 1, 3])
    model_output: tensor([[3.0631],
            [3.6093],
            [2.0272],
            [3.3819]]), target_rating: tensor([2, 4, 2, 3])
    model_output: tensor([[2.9745],
            [4.4645],
            [3.1678],
            [2.1908]]), target_rating: tensor([4, 5, 3, 2])
    model_output: tensor([[2.9144],
            [3.8169],
            [3.1552],
            [3.7539]]), target_rating: tensor([4, 5, 2, 3])
    model_output: tensor([[3.0149],
            [2.9110],
            [3.2701],
            [3.2614]]), target_rating: tensor([4, 4, 3, 4])
    model_output: tensor([[2.3679],
            [3.5789],
            [3.4829],
            [3.0782]]), target_rating: tensor([3, 3, 4, 2])
    model_output: tensor([[2.5100],
            [3.4060],
            [3.0107],
            [2.6584]]), target_rating: tensor([3, 5, 2, 2])
    model_output: tensor([[2.3802],
            [3.1520],
            [3.5776],
            [2.8230]]), target_rating: tensor([1, 1, 2, 3])
    model_output: tensor([[3.4065],
            [3.7239],
            [2.8754],
            [3.3657]]), target_rating: tensor([1, 4, 1, 4])
    model_output: tensor([[3.5051],
            [3.8359],
            [2.9262],
            [3.0455]]), target_rating: tensor([5, 5, 5, 3])
    model_output: tensor([[3.5723],
            [3.0731],
            [2.7123],
            [3.5257]]), target_rating: tensor([4, 4, 3, 4])
    model_output: tensor([[3.3306],
            [3.6492],
            [2.6177],
            [3.9089]]), target_rating: tensor([5, 2, 3, 5])
    model_output: tensor([[3.7441],
            [3.9022],
            [3.4779],
            [4.1389]]), target_rating: tensor([5, 5, 4, 2])
    model_output: tensor([[2.3175],
            [2.9614],
            [3.2420],
            [3.8156]]), target_rating: tensor([2, 2, 3, 5])
    model_output: tensor([[3.6875],
            [3.2393],
            [2.7542],
            [3.7137]]), target_rating: tensor([5, 4, 3, 3])
    model_output: tensor([[3.0744],
            [3.8487],
            [2.6694],
            [3.1202]]), target_rating: tensor([4, 2, 2, 4])
    model_output: tensor([[2.4615],
            [3.5481],
            [3.4337],
            [3.7369]]), target_rating: tensor([1, 2, 3, 3])
    model_output: tensor([[3.4066],
            [3.7110],
            [3.1730],
            [3.5078]]), target_rating: tensor([4, 4, 2, 2])
    model_output: tensor([[3.5802],
            [3.9997],
            [3.2193],
            [3.0005]]), target_rating: tensor([4, 3, 3, 1])
    model_output: tensor([[2.6274],
            [1.7571],
            [2.8817],
            [2.4636]]), target_rating: tensor([4, 2, 4, 3])
    model_output: tensor([[3.3907],
            [3.1497],
            [3.0073],
            [3.8771]]), target_rating: tensor([4, 2, 2, 4])
    model_output: tensor([[3.1483],
            [3.2553],
            [2.0721],
            [4.0417]]), target_rating: tensor([3, 3, 3, 5])
    model_output: tensor([[3.5666],
            [3.2707],
            [3.7942],
            [3.9063]]), target_rating: tensor([4, 5, 4, 5])
    model_output: tensor([[3.7794],
            [2.9294],
            [3.4202],
            [3.2461]]), target_rating: tensor([3, 1, 4, 0])
    model_output: tensor([[3.4339],
            [3.0406],
            [3.0920],
            [2.9558]]), target_rating: tensor([3, 3, 3, 3])
    model_output: tensor([[3.3118],
            [4.2191],
            [2.7179],
            [3.7698]]), target_rating: tensor([3, 5, 2, 4])
    model_output: tensor([[3.4035],
            [4.2802],
            [4.0243],
            [4.0039]]), target_rating: tensor([3, 5, 5, 4])
    model_output: tensor([[2.8042],
            [2.4011],
            [3.4356],
            [4.0614]]), target_rating: tensor([3, 2, 3, 4])
    model_output: tensor([[3.6245],
            [3.4031],
            [3.6619],
            [3.0873]]), target_rating: tensor([0, 4, 4, 1])
    model_output: tensor([[2.8632],
            [4.5107],
            [3.1102],
            [4.3369]]), target_rating: tensor([3, 5, 4, 4])
    model_output: tensor([[3.6762],
            [3.1743],
            [2.8063],
            [3.4332]]), target_rating: tensor([4, 4, 2, 3])
    model_output: tensor([[2.7047],
            [4.1790],
            [2.1347],
            [4.0007]]), target_rating: tensor([4, 5, 2, 2])
    model_output: tensor([[3.9440],
            [3.2245],
            [3.4590],
            [3.7349]]), target_rating: tensor([1, 3, 3, 3])
    model_output: tensor([[3.7027],
            [4.3812],
            [1.9880],
            [3.9230]]), target_rating: tensor([4, 4, 2, 5])
    model_output: tensor([[3.4818],
            [3.3947],
            [2.2785],
            [3.3603]]), target_rating: tensor([2, 2, 1, 3])
    model_output: tensor([[3.7712],
            [3.6632],
            [3.3709],
            [3.1022]]), target_rating: tensor([2, 3, 1, 3])
    model_output: tensor([[3.2803],
            [2.2378],
            [4.1954],
            [4.2899]]), target_rating: tensor([0, 2, 5, 5])
    model_output: tensor([[2.6286],
            [3.4573],
            [2.1939],
            [3.3157]]), target_rating: tensor([2, 3, 3, 3])
    model_output: tensor([[2.4606],
            [3.9129],
            [3.9244],
            [3.4446]]), target_rating: tensor([2, 4, 4, 4])
    model_output: tensor([[3.7393],
            [3.3190],
            [3.1669],
            [3.5761]]), target_rating: tensor([4, 4, 4, 4])
    model_output: tensor([[3.4308],
            [3.5503],
            [3.4761],
            [3.6357]]), target_rating: tensor([3, 3, 4, 2])
    model_output: tensor([[3.2338],
            [2.9108],
            [2.9760],
            [3.1600]]), target_rating: tensor([4, 4, 3, 4])
    model_output: tensor([[3.0952],
            [3.2319],
            [3.5512],
            [2.7642]]), target_rating: tensor([3, 3, 5, 4])
    model_output: tensor([[3.1929],
            [3.6346],
            [3.5656],
            [2.5442]]), target_rating: tensor([3, 3, 3, 4])
    model_output: tensor([[3.2185],
            [3.3280],
            [3.2328],
            [3.2123]]), target_rating: tensor([3, 4, 2, 3])
    model_output: tensor([[2.1663],
            [3.9072],
            [3.2289],
            [2.5481]]), target_rating: tensor([1, 4, 3, 2])
    model_output: tensor([[2.9887],
            [3.6584],
            [3.1237],
            [4.0140]]), target_rating: tensor([0, 3, 4, 4])
    model_output: tensor([[2.9516],
            [4.0168],
            [3.1950],
            [3.4500]]), target_rating: tensor([4, 4, 4, 4])
    model_output: tensor([[3.6602],
            [2.2264],
            [3.4713],
            [3.8477]]), target_rating: tensor([5, 2, 4, 3])
    model_output: tensor([[3.8291],
            [2.8652],
            [2.6518],
            [3.9813]]), target_rating: tensor([4, 2, 4, 2])
    model_output: tensor([[3.7325],
            [3.1901],
            [3.0861],
            [2.8723]]), target_rating: tensor([2, 2, 4, 1])
    model_output: tensor([[3.7314],
            [3.4015],
            [2.8444],
            [3.2567]]), target_rating: tensor([4, 4, 4, 4])
    model_output: tensor([[3.3543],
            [3.7151],
            [3.8719],
            [2.6901]]), target_rating: tensor([4, 4, 4, 1])
    model_output: tensor([[3.5056],
            [3.3776],
            [3.1580],
            [2.6960]]), target_rating: tensor([1, 5, 2, 4])
    model_output: tensor([[3.7025],
            [2.0405],
            [3.4936],
            [4.0004]]), target_rating: tensor([4, 1, 4, 5])
    model_output: tensor([[4.2477],
            [2.3105],
            [2.6162],
            [2.8142]]), target_rating: tensor([5, 2, 3, 3])
    model_output: tensor([[2.6662],
            [2.8565],
            [3.2761],
            [3.4472]]), target_rating: tensor([3, 3, 4, 4])
    model_output: tensor([[3.4104],
            [3.4642],
            [3.2649],
            [3.6404]]), target_rating: tensor([5, 4, 4, 3])
    model_output: tensor([[2.9172],
            [2.8209],
            [4.2150],
            [4.1141]]), target_rating: tensor([2, 3, 0, 3])
    model_output: tensor([[3.5342],
            [3.3997],
            [3.3108],
            [2.8016]]), target_rating: tensor([5, 3, 4, 4])
    model_output: tensor([[2.9272],
            [3.1342],
            [4.0127],
            [3.1170]]), target_rating: tensor([3, 3, 5, 4])
    model_output: tensor([[3.4835],
            [3.9545],
            [3.2174],
            [3.3780]]), target_rating: tensor([1, 5, 3, 3])
    model_output: tensor([[2.4034],
            [3.6646],
            [4.1086],
            [3.1013]]), target_rating: tensor([3, 4, 5, 4])
    model_output: tensor([[3.3219],
            [4.0912],
            [2.8208],
            [3.3699]]), target_rating: tensor([3, 4, 3, 3])
    model_output: tensor([[4.2854],
            [3.4972],
            [2.6353],
            [3.6230]]), target_rating: tensor([5, 3, 3, 4])
    model_output: tensor([[2.7262],
            [3.6700],
            [3.4444],
            [3.6899]]), target_rating: tensor([3, 5, 4, 5])
    model_output: tensor([[3.4377],
            [3.1215],
            [3.2444],
            [2.8890]]), target_rating: tensor([4, 5, 4, 2])
    model_output: tensor([[2.7112],
            [3.7055],
            [3.0071],
            [2.8670]]), target_rating: tensor([3, 5, 3, 3])
    model_output: tensor([[3.0871],
            [3.2511],
            [4.1001],
            [3.5817]]), target_rating: tensor([3, 4, 4, 2])
    model_output: tensor([[2.0663],
            [3.1382],
            [1.9263],
            [3.3420]]), target_rating: tensor([3, 2, 2, 3])
    model_output: tensor([[3.0473],
            [3.3763],
            [2.7460],
            [3.8942]]), target_rating: tensor([2, 3, 3, 4])
    model_output: tensor([[3.1479],
            [4.0362],
            [3.5891],
            [3.5981]]), target_rating: tensor([4, 4, 4, 4])
    model_output: tensor([[3.4825],
            [3.7448],
            [2.9401],
            [2.4128]]), target_rating: tensor([5, 2, 4, 2])
    model_output: tensor([[2.6873],
            [3.4440],
            [3.2067],
            [2.2952]]), target_rating: tensor([2, 4, 3, 4])
    model_output: tensor([[3.4505],
            [3.6226],
            [3.0823],
            [3.6570]]), target_rating: tensor([4, 4, 2, 3])
    model_output: tensor([[4.2588],
            [3.1763],
            [2.8543],
            [2.9218]]), target_rating: tensor([4, 3, 2, 3])
    model_output: tensor([[2.5094],
            [3.0901],
            [3.3157],
            [2.3396]]), target_rating: tensor([4, 4, 4, 4])
    model_output: tensor([[3.9285],
            [2.8335],
            [3.3339],
            [3.2962]]), target_rating: tensor([5, 3, 3, 4])
    model_output: tensor([[3.1434],
            [2.8019],
            [3.1406],
            [2.2684]]), target_rating: tensor([2, 4, 3, 2])
    model_output: tensor([[2.3025],
            [4.0473],
            [3.7180],
            [3.6658]]), target_rating: tensor([2, 4, 4, 4])
    model_output: tensor([[3.1338],
            [2.6682],
            [3.1853],
            [2.6377]]), target_rating: tensor([3, 0, 2, 3])
    model_output: tensor([[3.6274],
            [3.1801],
            [3.2123],
            [3.2039]]), target_rating: tensor([5, 3, 3, 3])
    model_output: tensor([[3.1369],
            [2.7865],
            [2.6531],
            [4.4705]]), target_rating: tensor([1, 4, 4, 5])
    model_output: tensor([[3.1702],
            [2.4796],
            [3.2073],
            [2.1986]]), target_rating: tensor([4, 2, 3, 2])
    model_output: tensor([[3.3524],
            [2.9868],
            [3.3149],
            [2.9983]]), target_rating: tensor([3, 3, 5, 2])
    model_output: tensor([[4.3077],
            [3.2270],
            [3.4436],
            [2.0922]]), target_rating: tensor([5, 2, 3, 4])
    model_output: tensor([[2.9445],
            [3.5920],
            [3.4057],
            [2.9954]]), target_rating: tensor([2, 4, 5, 3])
    model_output: tensor([[2.5906],
            [2.4684],
            [3.3401],
            [3.5899]]), target_rating: tensor([3, 2, 4, 4])
    model_output: tensor([[3.9817],
            [2.0478],
            [4.4097],
            [3.2167]]), target_rating: tensor([5, 2, 5, 3])
    model_output: tensor([[3.8190],
            [3.1393],
            [4.0741],
            [3.5821]]), target_rating: tensor([5, 3, 4, 5])
    model_output: tensor([[2.6820],
            [2.9790],
            [3.2139],
            [2.4998]]), target_rating: tensor([4, 3, 2, 3])
    model_output: tensor([[4.2394],
            [3.4746],
            [2.1821],
            [3.6292]]), target_rating: tensor([5, 3, 1, 5])
    model_output: tensor([[3.9657],
            [3.3970],
            [3.4266],
            [3.1105]]), target_rating: tensor([5, 3, 4, 3])
    model_output: tensor([[2.7424],
            [2.6419],
            [3.2850],
            [3.9058]]), target_rating: tensor([3, 3, 5, 4])
    model_output: tensor([[3.5765],
            [4.3648],
            [3.7228],
            [4.1677]]), target_rating: tensor([5, 4, 4, 4])
    model_output: tensor([[3.3776],
            [3.4065],
            [2.9374],
            [3.7093]]), target_rating: tensor([3, 4, 3, 5])
    model_output: tensor([[4.0529],
            [3.1253],
            [4.3733],
            [3.1600]]), target_rating: tensor([5, 2, 4, 3])
    model_output: tensor([[3.7084],
            [4.1347],
            [3.4969],
            [3.0505]]), target_rating: tensor([4, 2, 2, 1])
    model_output: tensor([[2.3086],
            [3.2698],
            [3.3059],
            [2.9028]]), target_rating: tensor([1, 1, 4, 3])
    model_output: tensor([[2.2885],
            [3.7613],
            [2.4516],
            [4.3899]]), target_rating: tensor([2, 3, 4, 2])
    model_output: tensor([[3.0325],
            [4.1180],
            [2.9946],
            [3.2579]]), target_rating: tensor([2, 4, 4, 3])
    model_output: tensor([[3.2416],
            [3.0694],
            [2.5142],
            [2.3160]]), target_rating: tensor([4, 2, 3, 1])
    model_output: tensor([[3.4398],
            [3.4514],
            [3.1042],
            [2.7095]]), target_rating: tensor([4, 3, 3, 4])
    model_output: tensor([[3.0307],
            [3.6065],
            [2.9122],
            [4.1794]]), target_rating: tensor([4, 4, 3, 4])
    model_output: tensor([[3.8600],
            [2.6587],
            [2.4358],
            [3.4157]]), target_rating: tensor([4, 3, 3, 3])
    model_output: tensor([[3.1608],
            [2.5199],
            [2.6184],
            [3.1075]]), target_rating: tensor([3, 3, 3, 3])
    model_output: tensor([[3.4921],
            [3.6797],
            [3.1559],
            [4.2236]]), target_rating: tensor([4, 4, 3, 4])
    model_output: tensor([[2.1794],
            [2.8213],
            [2.0367],
            [3.4857]]), target_rating: tensor([4, 1, 3, 5])
    model_output: tensor([[2.2427],
            [3.1215],
            [2.8434],
            [3.4365]]), target_rating: tensor([2, 4, 4, 3])
    model_output: tensor([[3.5698],
            [2.4763],
            [3.0269],
            [3.1019]]), target_rating: tensor([4, 1, 2, 3])
    model_output: tensor([[2.6462],
            [3.5854],
            [3.5786],
            [3.7992]]), target_rating: tensor([3, 4, 4, 4])
    model_output: tensor([[3.9512],
            [3.9446],
            [3.7400],
            [3.4433]]), target_rating: tensor([3, 4, 5, 4])
    model_output: tensor([[3.6525],
            [3.0812],
            [2.6801],
            [3.4479]]), target_rating: tensor([4, 3, 3, 2])
    model_output: tensor([[3.5427],
            [3.0409],
            [2.2843],
            [3.8275]]), target_rating: tensor([4, 3, 3, 5])
    model_output: tensor([[3.0248],
            [3.3123],
            [3.3742],
            [2.7080]]), target_rating: tensor([3, 4, 2, 3])
    model_output: tensor([[3.5942],
            [2.6517],
            [4.2349],
            [3.4345]]), target_rating: tensor([2, 5, 4, 4])
    model_output: tensor([[3.6882],
            [3.8990],
            [2.1261],
            [4.2689]]), target_rating: tensor([4, 5, 0, 4])
    model_output: tensor([[2.3113],
            [3.7051],
            [3.6330],
            [4.1985]]), target_rating: tensor([3, 5, 3, 5])
    model_output: tensor([[4.1673],
            [3.0070],
            [3.9378],
            [3.0446]]), target_rating: tensor([4, 3, 1, 4])
    model_output: tensor([[3.5565],
            [3.2571],
            [3.4154],
            [2.9422]]), target_rating: tensor([4, 5, 4, 2])
    model_output: tensor([[3.4072],
            [3.5938],
            [3.3386],
            [3.5211]]), target_rating: tensor([2, 4, 4, 3])
    model_output: tensor([[3.8889],
            [3.4925],
            [4.0488],
            [3.0680]]), target_rating: tensor([3, 5, 2, 3])
    model_output: tensor([[2.4109],
            [3.3917],
            [3.4399],
            [3.1700]]), target_rating: tensor([2, 4, 3, 2])
    model_output: tensor([[3.0724],
            [3.4552],
            [3.1326],
            [3.5700]]), target_rating: tensor([3, 4, 3, 3])
    model_output: tensor([[4.3811],
            [3.8863],
            [3.2781],
            [3.0444]]), target_rating: tensor([5, 3, 3, 3])
    model_output: tensor([[2.8640],
            [3.3663],
            [2.9828],
            [3.1518]]), target_rating: tensor([4, 4, 3, 3])
    model_output: tensor([[3.6022],
            [3.1389],
            [2.6288],
            [3.2624]]), target_rating: tensor([4, 3, 4, 5])
    model_output: tensor([[3.2410],
            [3.5131],
            [3.2838],
            [3.4374]]), target_rating: tensor([3, 3, 3, 5])
    model_output: tensor([[3.4200],
            [3.5746],
            [2.8606],
            [2.6255]]), target_rating: tensor([4, 4, 3, 4])
    model_output: tensor([[1.9907],
            [2.7501],
            [3.6470],
            [4.4682]]), target_rating: tensor([2, 2, 3, 5])
    model_output: tensor([[2.9942],
            [3.7327],
            [2.1289],
            [3.4015]]), target_rating: tensor([2, 4, 5, 4])
    model_output: tensor([[3.7298],
            [3.2111],
            [3.4082],
            [2.9773]]), target_rating: tensor([3, 3, 4, 3])
    model_output: tensor([[2.8704],
            [3.5292],
            [2.6150],
            [3.1239]]), target_rating: tensor([3, 5, 3, 3])
    model_output: tensor([[3.7832],
            [3.3665],
            [3.6161],
            [2.1695]]), target_rating: tensor([4, 4, 4, 3])
    model_output: tensor([[3.3415],
            [3.7749],
            [2.8361],
            [3.9605]]), target_rating: tensor([4, 4, 4, 4])
    model_output: tensor([[3.1228],
            [2.8345],
            [2.8445],
            [3.7516]]), target_rating: tensor([4, 3, 3, 4])
    model_output: tensor([[3.5826],
            [3.5517],
            [2.4187],
            [3.8432]]), target_rating: tensor([3, 3, 3, 4])
    model_output: tensor([[2.6908],
            [4.0620],
            [3.2934],
            [3.9932]]), target_rating: tensor([3, 4, 5, 4])
    model_output: tensor([[4.5814],
            [3.8061],
            [3.1100],
            [3.5963]]), target_rating: tensor([5, 4, 3, 3])
    model_output: tensor([[3.2540],
            [3.2782],
            [2.6135],
            [3.1411]]), target_rating: tensor([4, 2, 3, 3])
    model_output: tensor([[3.6808],
            [3.4875],
            [2.8040],
            [3.4042]]), target_rating: tensor([4, 3, 3, 4])
    model_output: tensor([[3.8707],
            [3.8909],
            [2.6522],
            [3.3984]]), target_rating: tensor([5, 2, 3, 5])
    model_output: tensor([[3.2365],
            [2.5444],
            [3.9595],
            [3.4478]]), target_rating: tensor([2, 3, 5, 1])
    model_output: tensor([[3.5621],
            [3.5669],
            [4.0120],
            [2.9885]]), target_rating: tensor([3, 3, 4, 3])
    model_output: tensor([[3.5231],
            [2.6087],
            [3.2368],
            [4.3709]]), target_rating: tensor([3, 4, 2, 5])
    model_output: tensor([[3.6571],
            [3.6944],
            [2.9029],
            [3.7547]]), target_rating: tensor([3, 5, 3, 3])
    model_output: tensor([[3.1928],
            [3.9727],
            [2.7595],
            [3.7964]]), target_rating: tensor([4, 3, 3, 4])
    model_output: tensor([[3.8199],
            [3.6450],
            [3.9405],
            [3.6035]]), target_rating: tensor([4, 5, 3, 4])
    model_output: tensor([[3.2926],
            [2.7028],
            [4.0124],
            [3.3417]]), target_rating: tensor([2, 3, 3, 3])
    model_output: tensor([[2.9897],
            [3.8954],
            [2.8848],
            [3.6353]]), target_rating: tensor([4, 4, 3, 2])
    model_output: tensor([[4.0063],
            [2.1308],
            [2.5860],
            [2.7409]]), target_rating: tensor([4, 2, 1, 0])
    model_output: tensor([[3.7605],
            [3.3146],
            [3.8714],
            [3.2931]]), target_rating: tensor([4, 2, 4, 3])
    model_output: tensor([[2.9845],
            [3.2315],
            [3.4471],
            [2.0696]]), target_rating: tensor([3, 3, 4, 2])
    model_output: tensor([[3.4248],
            [3.5541],
            [3.0003],
            [3.7227]]), target_rating: tensor([3, 0, 4, 3])
    model_output: tensor([[3.3824],
            [3.6395],
            [2.9397],
            [3.7082]]), target_rating: tensor([4, 3, 4, 3])
    model_output: tensor([[3.9628],
            [2.7837],
            [3.9794],
            [2.4903]]), target_rating: tensor([4, 2, 4, 2])
    model_output: tensor([[2.1742],
            [3.4633],
            [3.8212],
            [3.2814]]), target_rating: tensor([1, 4, 4, 2])
    model_output: tensor([[3.4341],
            [3.6384],
            [3.2603],
            [3.4271]]), target_rating: tensor([2, 3, 5, 3])
    model_output: tensor([[3.6160],
            [3.2554],
            [2.3622],
            [3.1587]]), target_rating: tensor([4, 4, 5, 4])
    model_output: tensor([[2.5244],
            [4.0669],
            [3.3658],
            [3.3110]]), target_rating: tensor([3, 5, 2, 0])
    model_output: tensor([[3.5474],
            [3.3338],
            [3.6471],
            [2.8524]]), target_rating: tensor([4, 4, 4, 0])
    model_output: tensor([[2.1873],
            [3.3037],
            [3.8067],
            [3.6897]]), target_rating: tensor([2, 3, 3, 4])
    model_output: tensor([[3.2202],
            [3.5035],
            [3.0558],
            [3.5375]]), target_rating: tensor([2, 3, 3, 4])
    model_output: tensor([[3.1285],
            [3.2707],
            [3.6344],
            [3.7157]]), target_rating: tensor([3, 3, 4, 3])
    model_output: tensor([[3.3318],
            [3.5726],
            [2.0009],
            [2.0670]]), target_rating: tensor([3, 3, 2, 2])
    model_output: tensor([[3.1175],
            [3.6520],
            [2.5528],
            [3.6607]]), target_rating: tensor([3, 4, 2, 4])
    model_output: tensor([[2.9065],
            [3.1516],
            [4.1343],
            [3.3289]]), target_rating: tensor([1, 5, 3, 4])
    model_output: tensor([[3.0599],
            [3.3669],
            [3.0284],
            [3.2234]]), target_rating: tensor([1, 4, 3, 4])
    model_output: tensor([[3.0659],
            [3.8694],
            [3.4360],
            [2.7920]]), target_rating: tensor([5, 4, 4, 2])
    model_output: tensor([[4.0296],
            [3.6862],
            [2.8606],
            [2.8719]]), target_rating: tensor([4, 3, 3, 2])
    model_output: tensor([[4.7633],
            [4.0087],
            [3.2925],
            [3.3054]]), target_rating: tensor([5, 4, 2, 3])
    model_output: tensor([[4.1638],
            [3.5274],
            [2.7691],
            [3.1900]]), target_rating: tensor([5, 5, 3, 3])
    model_output: tensor([[2.8351],
            [3.5723],
            [3.4026],
            [3.6459]]), target_rating: tensor([3, 3, 4, 4])
    model_output: tensor([[2.7309],
            [2.7923],
            [2.7975],
            [3.0457]]), target_rating: tensor([3, 3, 3, 4])
    model_output: tensor([[3.4502],
            [3.4111],
            [2.6355],
            [4.0609]]), target_rating: tensor([4, 5, 2, 4])
    model_output: tensor([[3.2887],
            [3.4231],
            [3.6793],
            [3.3464]]), target_rating: tensor([1, 4, 4, 5])
    model_output: tensor([[2.5138],
            [3.5161],
            [3.3647],
            [3.1349]]), target_rating: tensor([3, 4, 4, 5])
    model_output: tensor([[2.7723],
            [2.8687],
            [3.5284],
            [3.3218]]), target_rating: tensor([4, 3, 3, 4])
    model_output: tensor([[3.4785],
            [3.2504],
            [2.9498],
            [2.3822]]), target_rating: tensor([2, 3, 4, 3])
    model_output: tensor([[2.6950],
            [3.7504],
            [2.2027],
            [4.4434]]), target_rating: tensor([4, 4, 2, 4])
    model_output: tensor([[3.8856],
            [2.8935],
            [3.2257],
            [2.8216]]), target_rating: tensor([4, 2, 3, 3])
    model_output: tensor([[3.5734],
            [4.6025],
            [2.4354],
            [2.8309]]), target_rating: tensor([4, 4, 1, 2])
    model_output: tensor([[2.4868],
            [2.4637],
            [3.4738],
            [2.8696]]), target_rating: tensor([2, 2, 4, 3])
    model_output: tensor([[3.1092],
            [2.8218],
            [2.6947],
            [3.3034]]), target_rating: tensor([4, 3, 3, 5])
    model_output: tensor([[2.7127],
            [3.1139],
            [3.3458],
            [3.7451]]), target_rating: tensor([4, 3, 3, 4])
    model_output: tensor([[3.1228],
            [2.8182],
            [3.2513],
            [4.1401]]), target_rating: tensor([3, 3, 4, 5])
    model_output: tensor([[2.6046],
            [3.3381],
            [3.6104],
            [3.9055]]), target_rating: tensor([1, 3, 3, 3])
    model_output: tensor([[2.7659],
            [2.9959],
            [2.4986],
            [4.1482]]), target_rating: tensor([4, 4, 3, 4])
    model_output: tensor([[3.8199],
            [2.9556],
            [2.4382],
            [2.8578]]), target_rating: tensor([5, 2, 3, 5])
    model_output: tensor([[3.1905],
            [2.8561],
            [3.8812],
            [3.4190]]), target_rating: tensor([3, 3, 4, 3])
    model_output: tensor([[3.5900],
            [2.8042],
            [3.9299],
            [3.6129]]), target_rating: tensor([5, 2, 4, 1])
    model_output: tensor([[2.9863],
            [2.9927],
            [3.6960],
            [3.0683]]), target_rating: tensor([3, 4, 4, 3])
    model_output: tensor([[2.8603],
            [3.0332],
            [2.8430],
            [2.9471]]), target_rating: tensor([3, 3, 3, 3])
    model_output: tensor([[2.6243],
            [3.1023],
            [3.1138],
            [3.8706]]), target_rating: tensor([4, 1, 3, 4])
    model_output: tensor([[3.0673],
            [3.2135],
            [2.8332],
            [2.4675]]), target_rating: tensor([4, 3, 2, 3])
    model_output: tensor([[3.3204],
            [3.7474],
            [3.7060],
            [3.6038]]), target_rating: tensor([4, 4, 4, 2])
    model_output: tensor([[4.0613],
            [2.9036],
            [3.8589],
            [2.5367]]), target_rating: tensor([4, 4, 5, 3])
    model_output: tensor([[3.6112],
            [2.7450],
            [3.3049],
            [3.4963]]), target_rating: tensor([4, 4, 3, 4])
    model_output: tensor([[2.9944],
            [3.4894],
            [3.0205],
            [4.2656]]), target_rating: tensor([4, 4, 3, 5])
    model_output: tensor([[3.9473],
            [2.7229],
            [3.1404],
            [2.7240]]), target_rating: tensor([4, 4, 4, 1])
    model_output: tensor([[3.2705],
            [2.9092],
            [3.2148],
            [2.8287]]), target_rating: tensor([3, 2, 3, 3])
    model_output: tensor([[3.4698],
            [2.7505],
            [2.8094],
            [3.2133]]), target_rating: tensor([4, 4, 2, 4])
    model_output: tensor([[3.6694],
            [3.5943],
            [3.4189],
            [3.1505]]), target_rating: tensor([5, 3, 4, 3])
    model_output: tensor([[3.6076],
            [3.9105],
            [3.5059],
            [2.3410]]), target_rating: tensor([3, 5, 4, 1])
    model_output: tensor([[3.8340],
            [3.2032],
            [3.2304],
            [3.5110]]), target_rating: tensor([4, 5, 3, 5])
    model_output: tensor([[3.4557],
            [3.0828],
            [3.5827],
            [3.7027]]), target_rating: tensor([4, 3, 4, 3])
    model_output: tensor([[2.2472],
            [4.5734],
            [2.8192],
            [3.5273]]), target_rating: tensor([3, 5, 3, 4])
    model_output: tensor([[2.7274],
            [3.1970],
            [2.7954],
            [3.0766]]), target_rating: tensor([3, 3, 2, 4])
    model_output: tensor([[4.0728],
            [3.4904],
            [3.4999],
            [3.0583]]), target_rating: tensor([5, 4, 3, 2])
    model_output: tensor([[3.7478],
            [3.4036],
            [2.9538],
            [3.1542]]), target_rating: tensor([4, 3, 4, 4])
    model_output: tensor([[3.3462],
            [2.8951],
            [3.3910],
            [3.3546]]), target_rating: tensor([4, 3, 3, 1])
    model_output: tensor([[3.2620],
            [2.1522],
            [3.0634],
            [3.7821]]), target_rating: tensor([5, 0, 4, 4])
    model_output: tensor([[3.9583],
            [2.8032],
            [3.8076],
            [3.4739]]), target_rating: tensor([3, 3, 4, 4])
    model_output: tensor([[3.2708],
            [4.5897],
            [2.7757],
            [2.5327]]), target_rating: tensor([4, 5, 1, 2])
    model_output: tensor([[2.7822],
            [3.3128],
            [2.7057],
            [3.3010]]), target_rating: tensor([2, 2, 3, 3])
    model_output: tensor([[3.5966],
            [3.1363],
            [3.9024],
            [3.7077]]), target_rating: tensor([4, 2, 4, 5])
    model_output: tensor([[3.0329],
            [2.2434],
            [3.3807],
            [3.1689]]), target_rating: tensor([3, 2, 4, 3])
    model_output: tensor([[2.7997],
            [2.8470],
            [3.2077],
            [3.5588]]), target_rating: tensor([3, 4, 4, 4])
    model_output: tensor([[2.7718],
            [3.3024],
            [3.8546],
            [2.6949]]), target_rating: tensor([2, 3, 4, 3])
    model_output: tensor([[3.4544],
            [2.4828],
            [2.4229],
            [3.4457]]), target_rating: tensor([3, 2, 2, 2])
    model_output: tensor([[2.9973],
            [2.2758],
            [3.1300],
            [3.7327]]), target_rating: tensor([3, 3, 3, 3])
    model_output: tensor([[3.0311],
            [3.5044],
            [3.7397],
            [3.3795]]), target_rating: tensor([4, 5, 4, 3])
    model_output: tensor([[3.7619],
            [3.4524],
            [3.1747],
            [2.8129]]), target_rating: tensor([3, 4, 3, 2])
    model_output: tensor([[3.6372],
            [2.9029],
            [2.7855],
            [3.2152]]), target_rating: tensor([4, 3, 4, 4])
    model_output: tensor([[3.5779],
            [3.0868],
            [3.5268],
            [3.3283]]), target_rating: tensor([2, 3, 4, 3])
    model_output: tensor([[3.3094],
            [3.1892],
            [2.0451],
            [1.7499]]), target_rating: tensor([4, 4, 3, 2])
    model_output: tensor([[3.5915],
            [4.6051],
            [3.2381],
            [3.4501]]), target_rating: tensor([3, 4, 3, 3])
    model_output: tensor([[3.5931],
            [3.2743],
            [2.7905],
            [3.2353]]), target_rating: tensor([4, 4, 4, 5])
    model_output: tensor([[3.8779],
            [3.3679],
            [2.2691],
            [3.6548]]), target_rating: tensor([5, 4, 2, 3])
    model_output: tensor([[3.6696],
            [3.0062],
            [3.8895],
            [3.7307]]), target_rating: tensor([4, 4, 5, 4])
    model_output: tensor([[3.2371],
            [4.0342],
            [3.7712],
            [3.5992]]), target_rating: tensor([3, 2, 4, 4])
    model_output: tensor([[2.7306],
            [3.3748],
            [3.7467],
            [3.4000]]), target_rating: tensor([1, 4, 4, 3])
    model_output: tensor([[3.1659],
            [4.0341],
            [2.9872],
            [3.5491]]), target_rating: tensor([3, 1, 3, 2])
    model_output: tensor([[3.0293],
            [3.0659],
            [3.6525],
            [3.7249]]), target_rating: tensor([2, 4, 3, 5])
    model_output: tensor([[3.5850],
            [3.2996],
            [2.6918],
            [4.2168]]), target_rating: tensor([3, 3, 3, 4])
    model_output: tensor([[3.3047],
            [3.9480],
            [3.4178],
            [2.9783]]), target_rating: tensor([4, 5, 4, 2])
    model_output: tensor([[2.8901],
            [3.6347],
            [4.0640],
            [3.2808]]), target_rating: tensor([3, 3, 5, 4])
    model_output: tensor([[2.5446],
            [3.6596],
            [3.7026],
            [2.7171]]), target_rating: tensor([3, 5, 4, 2])
    model_output: tensor([[3.1730],
            [2.6378],
            [2.6036],
            [3.1910]]), target_rating: tensor([3, 3, 3, 4])
    model_output: tensor([[3.6854],
            [2.2695],
            [2.8633],
            [3.0324]]), target_rating: tensor([5, 4, 4, 4])
    model_output: tensor([[3.8307],
            [3.8736],
            [2.9653],
            [3.7026]]), target_rating: tensor([4, 5, 3, 3])
    model_output: tensor([[3.8928],
            [3.3229],
            [3.2110],
            [3.6557]]), target_rating: tensor([4, 3, 2, 4])
    model_output: tensor([[3.2290],
            [3.1102],
            [3.3352],
            [3.0935]]), target_rating: tensor([3, 3, 2, 4])
    model_output: tensor([[2.9183],
            [3.5063],
            [3.6753],
            [3.3073]]), target_rating: tensor([3, 4, 4, 4])
    model_output: tensor([[3.4670],
            [3.0342],
            [3.3866],
            [3.9517]]), target_rating: tensor([3, 5, 4, 4])
    model_output: tensor([[3.4245],
            [3.5377],
            [3.3347],
            [3.1049]]), target_rating: tensor([3, 3, 1, 4])
    model_output: tensor([[2.4966],
            [2.8346],
            [3.1046],
            [3.6654]]), target_rating: tensor([3, 3, 2, 3])
    model_output: tensor([[3.4247],
            [3.3996],
            [2.8111],
            [3.4770]]), target_rating: tensor([2, 4, 1, 4])
    model_output: tensor([[3.7025],
            [3.5673],
            [3.5190],
            [2.9708]]), target_rating: tensor([5, 4, 3, 2])
    model_output: tensor([[2.3598],
            [4.0041],
            [2.8251],
            [3.4141]]), target_rating: tensor([1, 5, 4, 4])
    model_output: tensor([[3.1715],
            [2.4634],
            [3.7098],
            [3.7838]]), target_rating: tensor([4, 2, 5, 3])
    model_output: tensor([[3.4220],
            [3.4126],
            [3.8863],
            [3.1400]]), target_rating: tensor([4, 5, 5, 3])
    model_output: tensor([[3.2348],
            [2.9358],
            [2.8564],
            [3.2067]]), target_rating: tensor([4, 4, 4, 3])
    model_output: tensor([[2.1034],
            [2.7557],
            [3.5184],
            [2.8713]]), target_rating: tensor([2, 3, 4, 4])
    model_output: tensor([[3.1000],
            [2.9785],
            [3.7030],
            [3.7864]]), target_rating: tensor([1, 3, 3, 4])
    model_output: tensor([[2.9507],
            [3.3423],
            [4.0671],
            [3.6165]]), target_rating: tensor([5, 3, 4, 3])
    model_output: tensor([[3.3200],
            [3.7378],
            [4.0278],
            [3.3236]]), target_rating: tensor([4, 4, 4, 4])
    model_output: tensor([[3.1661],
            [3.2201],
            [3.7980],
            [3.1543]]), target_rating: tensor([4, 4, 3, 4])
    model_output: tensor([[4.1656],
            [4.3102],
            [2.0849],
            [2.1729]]), target_rating: tensor([3, 4, 3, 3])
    model_output: tensor([[4.2892],
            [3.4319],
            [3.1363],
            [3.8733]]), target_rating: tensor([3, 4, 2, 4])
    model_output: tensor([[3.3856],
            [3.6885],
            [2.8723],
            [2.7170]]), target_rating: tensor([5, 3, 3, 3])
    model_output: tensor([[3.6334],
            [3.2483],
            [3.2582],
            [3.4961]]), target_rating: tensor([4, 4, 3, 3])
    model_output: tensor([[2.5480],
            [3.4354],
            [3.1889],
            [2.9007]]), target_rating: tensor([3, 4, 5, 3])
    model_output: tensor([[2.9321],
            [3.3141],
            [2.3674],
            [2.9720]]), target_rating: tensor([5, 4, 4, 4])
    model_output: tensor([[3.7448],
            [3.8144],
            [2.9231],
            [3.7165]]), target_rating: tensor([3, 5, 4, 5])
    model_output: tensor([[3.5524],
            [2.9472],
            [3.1319],
            [2.7841]]), target_rating: tensor([4, 4, 4, 4])
    model_output: tensor([[3.8613],
            [2.3952],
            [3.5039],
            [3.1177]]), target_rating: tensor([4, 3, 3, 4])
    model_output: tensor([[2.9343],
            [3.0534],
            [4.9716],
            [3.3736]]), target_rating: tensor([0, 3, 5, 3])
    model_output: tensor([[3.8965],
            [3.2865],
            [3.7861],
            [3.7013]]), target_rating: tensor([5, 4, 3, 3])
    model_output: tensor([[3.2141],
            [3.2733],
            [3.2839],
            [2.3903]]), target_rating: tensor([3, 5, 3, 2])
    model_output: tensor([[2.3149],
            [2.4938],
            [3.0805],
            [1.9689]]), target_rating: tensor([2, 3, 4, 2])
    model_output: tensor([[3.3781],
            [3.1456],
            [4.0881],
            [3.8877]]), target_rating: tensor([4, 5, 4, 5])
    model_output: tensor([[2.6027],
            [4.0732],
            [3.7816],
            [3.5829]]), target_rating: tensor([1, 5, 4, 4])
    model_output: tensor([[3.4953],
            [3.4278],
            [4.5047],
            [2.9795]]), target_rating: tensor([4, 4, 5, 3])
    model_output: tensor([[3.3913],
            [3.1123],
            [2.7864],
            [2.9606]]), target_rating: tensor([5, 3, 1, 5])
    model_output: tensor([[3.6673],
            [2.8157],
            [2.7941],
            [3.2818]]), target_rating: tensor([5, 3, 3, 4])
    model_output: tensor([[3.7347],
            [3.0880],
            [2.8936],
            [3.0789]]), target_rating: tensor([3, 5, 3, 4])
    model_output: tensor([[2.4502],
            [3.7669],
            [4.2963],
            [3.4551]]), target_rating: tensor([2, 3, 4, 4])
    model_output: tensor([[3.9445],
            [3.2416],
            [2.5072],
            [3.3263]]), target_rating: tensor([5, 3, 2, 2])
    model_output: tensor([[3.3471],
            [2.4723],
            [2.8950],
            [3.4555]]), target_rating: tensor([4, 3, 2, 4])
    model_output: tensor([[2.8311],
            [2.7984],
            [3.3371],
            [3.4216]]), target_rating: tensor([2, 2, 3, 4])
    model_output: tensor([[3.3964],
            [2.6010],
            [3.1846],
            [4.0862]]), target_rating: tensor([4, 3, 3, 4])
    model_output: tensor([[3.3167],
            [3.5467],
            [3.8073],
            [2.8528]]), target_rating: tensor([3, 3, 5, 5])
    model_output: tensor([[2.9660],
            [2.9831],
            [3.2424],
            [2.9619]]), target_rating: tensor([4, 3, 4, 5])
    model_output: tensor([[3.2425],
            [2.2911],
            [3.4297],
            [2.7230]]), target_rating: tensor([5, 3, 4, 2])
    model_output: tensor([[3.0431],
            [3.6564],
            [3.1440],
            [3.2007]]), target_rating: tensor([3, 4, 4, 3])
    model_output: tensor([[4.0969],
            [3.1992],
            [2.9019],
            [3.3014]]), target_rating: tensor([4, 4, 4, 3])
    model_output: tensor([[3.8715],
            [3.7951],
            [3.9864],
            [3.5790]]), target_rating: tensor([5, 5, 5, 3])
    model_output: tensor([[3.1932],
            [3.6881],
            [3.4938],
            [3.0061]]), target_rating: tensor([4, 3, 3, 3])
    model_output: tensor([[3.1280],
            [2.7793],
            [2.4258],
            [3.9201]]), target_rating: tensor([3, 2, 3, 3])
    model_output: tensor([[3.5571],
            [3.2297],
            [3.4621],
            [2.7574]]), target_rating: tensor([3, 3, 5, 3])
    model_output: tensor([[3.1897],
            [3.9316],
            [3.2930],
            [3.8960]]), target_rating: tensor([3, 4, 3, 3])
    model_output: tensor([[2.8936],
            [4.5653],
            [3.1850],
            [3.9961]]), target_rating: tensor([4, 4, 5, 5])
    model_output: tensor([[3.7568],
            [3.2260],
            [4.4927],
            [3.6428]]), target_rating: tensor([2, 3, 3, 4])
    model_output: tensor([[3.4345],
            [4.0984],
            [3.5425],
            [2.7421]]), target_rating: tensor([3, 4, 4, 4])
    model_output: tensor([[2.7480],
            [4.0571],
            [3.7221],
            [3.2898]]), target_rating: tensor([2, 5, 5, 4])
    model_output: tensor([[3.0732],
            [4.2809],
            [4.2077],
            [3.5456]]), target_rating: tensor([2, 5, 4, 4])
    model_output: tensor([[4.8603],
            [2.6813],
            [3.4334],
            [3.7196]]), target_rating: tensor([4, 4, 3, 4])
    model_output: tensor([[2.8208],
            [3.4026],
            [3.3459],
            [3.3184]]), target_rating: tensor([1, 4, 3, 4])
    model_output: tensor([[4.0591],
            [3.5748],
            [3.3903],
            [3.2625]]), target_rating: tensor([4, 2, 3, 3])
    model_output: tensor([[2.6775],
            [4.0311],
            [4.4737],
            [3.0025]]), target_rating: tensor([3, 4, 5, 3])
    model_output: tensor([[3.5647],
            [2.1027],
            [3.5538],
            [4.4843]]), target_rating: tensor([5, 3, 0, 4])
    model_output: tensor([[3.9372],
            [2.5542],
            [3.8842],
            [3.1039]]), target_rating: tensor([4, 3, 4, 2])
    model_output: tensor([[3.5017],
            [3.4671],
            [3.5764],
            [4.1314]]), target_rating: tensor([5, 4, 3, 4])
    model_output: tensor([[2.7076],
            [3.1851],
            [3.1168],
            [3.7209]]), target_rating: tensor([3, 4, 4, 4])
    model_output: tensor([[3.4779],
            [2.2486],
            [3.6576],
            [3.1945]]), target_rating: tensor([4, 2, 4, 3])
    model_output: tensor([[3.6291],
            [3.3284],
            [3.7403],
            [3.5151]]), target_rating: tensor([3, 5, 3, 4])
    model_output: tensor([[4.3703],
            [3.7372],
            [2.5887],
            [4.3932]]), target_rating: tensor([5, 4, 1, 4])
    model_output: tensor([[4.3914],
            [3.1157],
            [2.7184],
            [3.2414]]), target_rating: tensor([5, 3, 3, 3])
    model_output: tensor([[4.2232],
            [4.2691],
            [1.8797],
            [3.3756]]), target_rating: tensor([4, 2, 3, 3])
    model_output: tensor([[3.6466],
            [3.6813],
            [3.3040],
            [3.0629]]), target_rating: tensor([4, 3, 5, 4])
    model_output: tensor([[3.1838],
            [3.1759],
            [3.1901],
            [3.2970]]), target_rating: tensor([1, 4, 4, 4])
    model_output: tensor([[3.2708],
            [3.7077],
            [3.3239],
            [3.3451]]), target_rating: tensor([3, 4, 4, 5])
    model_output: tensor([[3.3788],
            [3.4475],
            [3.2082],
            [3.2258]]), target_rating: tensor([3, 5, 3, 3])
    model_output: tensor([[3.9761],
            [3.6079],
            [3.2837],
            [3.4603]]), target_rating: tensor([4, 1, 3, 4])
    model_output: tensor([[3.9266],
            [3.0761],
            [3.6553],
            [2.6350]]), target_rating: tensor([5, 3, 4, 3])
    model_output: tensor([[3.3519],
            [2.4778],
            [3.2206],
            [2.9918]]), target_rating: tensor([1, 2, 3, 4])
    model_output: tensor([[3.5292],
            [3.5082],
            [3.0614],
            [3.5677]]), target_rating: tensor([3, 5, 4, 2])
    model_output: tensor([[3.5053],
            [3.9313],
            [3.7914],
            [3.4068]]), target_rating: tensor([3, 4, 5, 1])
    model_output: tensor([[3.6353],
            [3.0622],
            [2.6205],
            [2.1766]]), target_rating: tensor([4, 3, 3, 2])
    model_output: tensor([[3.0918],
            [3.1111],
            [3.3679],
            [3.1137]]), target_rating: tensor([4, 4, 3, 4])
    model_output: tensor([[4.0632],
            [2.7913],
            [3.8366],
            [3.3669]]), target_rating: tensor([4, 3, 4, 3])
    model_output: tensor([[3.6431],
            [3.3944],
            [2.2794],
            [3.3289]]), target_rating: tensor([3, 3, 3, 3])
    model_output: tensor([[3.4448],
            [3.0412],
            [3.7824],
            [3.4164]]), target_rating: tensor([4, 2, 4, 3])
    model_output: tensor([[3.6955],
            [3.4849],
            [3.5252],
            [4.0490]]), target_rating: tensor([4, 0, 4, 4])
    model_output: tensor([[3.9794],
            [4.0064],
            [2.5771],
            [3.9816]]), target_rating: tensor([5, 4, 4, 4])
    model_output: tensor([[3.0372],
            [4.6120],
            [3.1729],
            [3.2280]]), target_rating: tensor([3, 4, 2, 3])
    model_output: tensor([[3.1695],
            [3.5551],
            [3.9885],
            [4.0181]]), target_rating: tensor([5, 3, 3, 3])
    model_output: tensor([[3.2923],
            [4.0831],
            [3.1638],
            [3.2292]]), target_rating: tensor([3, 4, 3, 1])
    model_output: tensor([[2.7776],
            [3.4133],
            [3.7689],
            [3.3445]]), target_rating: tensor([3, 5, 4, 2])
    model_output: tensor([[3.0375],
            [3.2737],
            [4.3460],
            [3.3574]]), target_rating: tensor([2, 3, 4, 3])
    model_output: tensor([[3.2950],
            [2.8403],
            [4.3722],
            [3.6419]]), target_rating: tensor([2, 3, 5, 4])
    model_output: tensor([[3.3685],
            [2.5714],
            [3.7234],
            [2.9840]]), target_rating: tensor([2, 3, 3, 2])
    model_output: tensor([[3.8351],
            [2.9417],
            [4.1903],
            [2.9483]]), target_rating: tensor([4, 3, 5, 2])
    model_output: tensor([[4.1005],
            [3.6249],
            [3.7992],
            [3.3196]]), target_rating: tensor([4, 3, 5, 4])
    model_output: tensor([[4.4877],
            [3.6509],
            [3.1280],
            [4.4533]]), target_rating: tensor([5, 5, 3, 4])
    model_output: tensor([[3.7252],
            [3.5981],
            [3.6820],
            [3.8199]]), target_rating: tensor([4, 4, 4, 3])
    model_output: tensor([[2.7373],
            [3.7647],
            [3.5261],
            [3.7010]]), target_rating: tensor([3, 4, 4, 3])
    model_output: tensor([[3.4149],
            [2.1133],
            [3.3523],
            [3.9931]]), target_rating: tensor([4, 2, 3, 5])
    model_output: tensor([[3.4702],
            [2.9955],
            [2.9575],
            [3.4264]]), target_rating: tensor([3, 4, 3, 3])
    model_output: tensor([[3.2321],
            [2.0784],
            [3.5610],
            [3.2185]]), target_rating: tensor([4, 3, 1, 2])
    model_output: tensor([[3.5119],
            [3.8072],
            [3.6480],
            [3.9126]]), target_rating: tensor([4, 5, 5, 5])
    model_output: tensor([[3.2193],
            [3.7253],
            [3.4817],
            [3.9332]]), target_rating: tensor([4, 3, 4, 4])
    model_output: tensor([[2.4186],
            [3.1709],
            [3.1902],
            [2.9958]]), target_rating: tensor([2, 3, 5, 3])
    model_output: tensor([[3.8083],
            [3.7369],
            [3.5314],
            [3.7183]]), target_rating: tensor([4, 4, 3, 4])
    model_output: tensor([[3.6896],
            [3.6804],
            [3.0016],
            [2.9429]]), target_rating: tensor([3, 3, 2, 1])
    model_output: tensor([[4.1294],
            [3.5598],
            [2.9680],
            [3.9285]]), target_rating: tensor([5, 4, 2, 4])
    model_output: tensor([[3.6494],
            [3.3414],
            [2.8576],
            [3.6091]]), target_rating: tensor([4, 3, 4, 4])
    model_output: tensor([[3.0584],
            [3.6533],
            [2.7478],
            [4.1949]]), target_rating: tensor([4, 1, 3, 2])
    model_output: tensor([[2.7304],
            [3.3268],
            [3.4368],
            [3.6316]]), target_rating: tensor([3, 4, 4, 4])
    model_output: tensor([[3.8251],
            [2.6016],
            [3.7317],
            [2.9125]]), target_rating: tensor([3, 3, 5, 3])
    model_output: tensor([[3.3817],
            [3.2126],
            [3.8203],
            [2.9330]]), target_rating: tensor([2, 0, 3, 4])
    model_output: tensor([[2.1329],
            [2.9195],
            [2.7493],
            [3.3366]]), target_rating: tensor([1, 3, 3, 3])
    model_output: tensor([[3.7488],
            [4.1149],
            [3.6271],
            [2.4499]]), target_rating: tensor([4, 4, 3, 3])
    model_output: tensor([[3.6401],
            [3.5622],
            [3.4194],
            [4.0129]]), target_rating: tensor([3, 5, 3, 4])
    model_output: tensor([[3.8732],
            [4.3475],
            [2.0329],
            [3.0506]]), target_rating: tensor([5, 4, 2, 4])
    model_output: tensor([[2.9995],
            [3.3770],
            [4.3085],
            [3.4080]]), target_rating: tensor([3, 3, 5, 4])
    model_output: tensor([[4.2701],
            [2.7979],
            [3.9551],
            [3.0060]]), target_rating: tensor([2, 3, 4, 3])
    model_output: tensor([[2.4375],
            [3.8419],
            [3.5110],
            [2.6388]]), target_rating: tensor([0, 3, 4, 5])
    model_output: tensor([[3.5677],
            [3.9233],
            [3.3527],
            [3.5078]]), target_rating: tensor([5, 3, 2, 3])
    model_output: tensor([[3.6664],
            [2.9873],
            [3.3365],
            [2.5285]]), target_rating: tensor([3, 2, 3, 3])
    model_output: tensor([[2.5141],
            [3.6108],
            [2.9699],
            [2.1999]]), target_rating: tensor([2, 2, 4, 2])
    model_output: tensor([[3.7653],
            [3.4830],
            [2.6043],
            [3.1293]]), target_rating: tensor([4, 4, 5, 2])
    model_output: tensor([[3.9101],
            [3.3251],
            [2.9378],
            [2.9094]]), target_rating: tensor([5, 2, 3, 3])
    model_output: tensor([[3.3502],
            [3.3217],
            [3.6046],
            [2.4785]]), target_rating: tensor([2, 2, 2, 4])
    model_output: tensor([[3.4007],
            [2.8361],
            [3.6186],
            [3.7227]]), target_rating: tensor([4, 2, 4, 4])
    model_output: tensor([[2.6075],
            [3.2977],
            [3.3959],
            [4.3436]]), target_rating: tensor([3, 3, 4, 4])
    model_output: tensor([[3.8186],
            [2.8540],
            [4.5917],
            [3.3754]]), target_rating: tensor([5, 3, 5, 4])
    model_output: tensor([[3.0937],
            [3.2709],
            [2.6725],
            [3.4246]]), target_rating: tensor([3, 4, 4, 5])
    model_output: tensor([[4.3852],
            [2.6307],
            [2.0771],
            [2.9092]]), target_rating: tensor([5, 2, 2, 3])
    model_output: tensor([[3.5997],
            [3.2513],
            [3.4988],
            [3.0843]]), target_rating: tensor([3, 3, 1, 1])
    model_output: tensor([[3.8086],
            [3.1103],
            [2.6412],
            [3.2123]]), target_rating: tensor([5, 3, 3, 4])
    model_output: tensor([[2.5293],
            [3.7181],
            [3.3064],
            [3.3648]]), target_rating: tensor([1, 3, 4, 3])
    model_output: tensor([[2.7936],
            [3.3225],
            [4.5670],
            [3.6837]]), target_rating: tensor([2, 4, 5, 3])
    model_output: tensor([[3.2102],
            [2.4353],
            [3.7231],
            [4.0092]]), target_rating: tensor([5, 0, 2, 5])
    model_output: tensor([[4.3365],
            [2.9411],
            [2.3261],
            [3.4708]]), target_rating: tensor([4, 4, 4, 4])
    model_output: tensor([[4.5328],
            [3.5591],
            [3.3025],
            [2.9535]]), target_rating: tensor([5, 4, 3, 3])
    model_output: tensor([[3.5269],
            [2.8604],
            [2.9968],
            [3.4664]]), target_rating: tensor([4, 4, 3, 3])
    model_output: tensor([[2.7511],
            [3.3483],
            [3.8608],
            [3.2818]]), target_rating: tensor([3, 3, 4, 2])
    model_output: tensor([[3.3718],
            [3.0969],
            [4.1623],
            [3.2955]]), target_rating: tensor([3, 3, 5, 1])
    model_output: tensor([[3.5442],
            [3.2976],
            [2.3891],
            [3.4583]]), target_rating: tensor([5, 0, 2, 3])
    model_output: tensor([[3.4326],
            [3.5080],
            [3.8359],
            [1.4323]]), target_rating: tensor([4, 3, 4, 1])
    model_output: tensor([[3.1323],
            [2.7799],
            [4.1212],
            [3.7770]]), target_rating: tensor([4, 3, 3, 5])
    model_output: tensor([[3.0448],
            [3.9923],
            [2.8493],
            [3.2443]]), target_rating: tensor([4, 4, 4, 4])
    model_output: tensor([[2.8469],
            [3.1691],
            [2.5037],
            [2.6565]]), target_rating: tensor([4, 2, 2, 4])
    model_output: tensor([[2.4654],
            [2.7738],
            [4.1106],
            [3.1750]]), target_rating: tensor([1, 3, 4, 2])
    model_output: tensor([[3.9186],
            [3.5742],
            [3.4514],
            [3.8741]]), target_rating: tensor([3, 5, 2, 5])
    model_output: tensor([[3.1123],
            [3.3245],
            [3.8123],
            [3.3234]]), target_rating: tensor([3, 3, 4, 5])
    model_output: tensor([[2.3494],
            [3.5223],
            [4.3001],
            [3.4742]]), target_rating: tensor([1, 4, 5, 4])
    model_output: tensor([[3.4522],
            [2.6794],
            [3.9584],
            [3.0637]]), target_rating: tensor([3, 3, 3, 4])
    model_output: tensor([[3.9842],
            [2.7649],
            [2.4543],
            [3.6835]]), target_rating: tensor([3, 4, 3, 5])
    model_output: tensor([[1.9690],
            [3.9455],
            [3.1019],
            [3.5247]]), target_rating: tensor([2, 3, 4, 3])
    model_output: tensor([[2.3822],
            [3.3114],
            [3.0622],
            [2.7374]]), target_rating: tensor([2, 2, 3, 3])
    model_output: tensor([[3.3138],
            [3.5750],
            [2.5600],
            [3.1895]]), target_rating: tensor([3, 3, 4, 2])
    model_output: tensor([[3.6265],
            [3.3322],
            [3.2468],
            [3.2888]]), target_rating: tensor([2, 5, 4, 4])
    model_output: tensor([[2.1910],
            [3.2913],
            [2.9233],
            [3.5225]]), target_rating: tensor([2, 3, 3, 5])
    model_output: tensor([[2.7109],
            [2.6815],
            [3.0511],
            [3.5304]]), target_rating: tensor([3, 2, 3, 3])
    model_output: tensor([[3.3994],
            [2.0064],
            [4.0442],
            [2.8728]]), target_rating: tensor([3, 3, 3, 1])
    model_output: tensor([[3.4573],
            [1.6874],
            [2.8673],
            [3.3539]]), target_rating: tensor([3, 3, 4, 2])
    model_output: tensor([[3.0589],
            [3.2144],
            [3.5820],
            [3.6503]]), target_rating: tensor([4, 4, 4, 4])
    model_output: tensor([[3.4531],
            [3.5070],
            [3.5350],
            [3.3413]]), target_rating: tensor([3, 4, 3, 3])
    model_output: tensor([[2.9316],
            [4.4781],
            [3.5800],
            [2.0645]]), target_rating: tensor([3, 4, 4, 3])
    model_output: tensor([[3.6896],
            [2.1550],
            [4.1020],
            [4.2651]]), target_rating: tensor([3, 1, 5, 4])
    model_output: tensor([[2.8306],
            [3.4397],
            [2.7997],
            [2.2472]]), target_rating: tensor([4, 4, 4, 1])
    model_output: tensor([[3.7613],
            [2.9935],
            [3.3798],
            [3.3914]]), target_rating: tensor([2, 4, 4, 3])
    model_output: tensor([[3.3694],
            [1.8724],
            [2.1551],
            [3.1487]]), target_rating: tensor([2, 2, 3, 3])
    model_output: tensor([[2.3345],
            [2.7402],
            [2.5142],
            [2.9792]]), target_rating: tensor([2, 3, 4, 2])
    model_output: tensor([[3.5519],
            [3.4340],
            [3.3894],
            [3.0994]]), target_rating: tensor([5, 5, 4, 3])
    model_output: tensor([[2.6901],
            [3.4933],
            [3.3546],
            [3.0234]]), target_rating: tensor([1, 4, 4, 2])
    model_output: tensor([[3.3142],
            [2.7453],
            [3.2005],
            [3.7015]]), target_rating: tensor([4, 2, 4, 5])
    model_output: tensor([[3.5083],
            [3.8074],
            [3.2269],
            [2.8306]]), target_rating: tensor([4, 4, 2, 4])
    model_output: tensor([[2.6450],
            [2.8601],
            [3.4396],
            [2.1540]]), target_rating: tensor([3, 4, 4, 2])
    model_output: tensor([[2.5010],
            [3.3023],
            [3.4476],
            [3.8435]]), target_rating: tensor([3, 5, 4, 5])
    model_output: tensor([[2.9174],
            [3.0787],
            [3.0731],
            [3.8101]]), target_rating: tensor([2, 3, 5, 5])
    model_output: tensor([[3.6089],
            [3.6769],
            [3.9073],
            [3.6137]]), target_rating: tensor([5, 4, 3, 2])
    model_output: tensor([[3.5853],
            [3.2262],
            [3.7078],
            [3.0806]]), target_rating: tensor([2, 1, 5, 3])
    model_output: tensor([[3.3069],
            [2.3922],
            [3.1841],
            [2.6632]]), target_rating: tensor([3, 2, 2, 3])
    model_output: tensor([[2.0302],
            [3.3311],
            [2.9936],
            [3.6873]]), target_rating: tensor([2, 5, 3, 3])
    model_output: tensor([[2.9082],
            [3.5471],
            [2.7398],
            [3.2201]]), target_rating: tensor([4, 4, 1, 3])
    model_output: tensor([[3.1496],
            [3.9232],
            [2.8619],
            [3.1192]]), target_rating: tensor([2, 3, 2, 4])
    model_output: tensor([[3.4364],
            [1.8637],
            [3.9699],
            [4.1828]]), target_rating: tensor([4, 1, 3, 4])
    model_output: tensor([[2.1129],
            [2.2763],
            [4.2820],
            [3.7618]]), target_rating: tensor([3, 2, 1, 4])
    model_output: tensor([[3.0013],
            [4.0760],
            [3.1204],
            [3.6001]]), target_rating: tensor([0, 5, 3, 3])
    model_output: tensor([[3.0722],
            [2.8684],
            [3.0480],
            [3.4432]]), target_rating: tensor([4, 4, 3, 2])
    model_output: tensor([[2.3555],
            [3.7592],
            [3.6015],
            [2.6556]]), target_rating: tensor([3, 5, 3, 3])
    model_output: tensor([[3.7924],
            [3.6932],
            [3.3521],
            [2.7235]]), target_rating: tensor([4, 4, 5, 3])
    model_output: tensor([[2.7571],
            [3.6080],
            [3.1846],
            [3.9925]]), target_rating: tensor([4, 4, 4, 4])
    model_output: tensor([[3.5653],
            [3.2202],
            [2.9329],
            [3.2387]]), target_rating: tensor([3, 3, 3, 4])
    model_output: tensor([[4.0455],
            [2.8039],
            [3.6131],
            [2.8819]]), target_rating: tensor([4, 3, 4, 3])
    model_output: tensor([[3.8824],
            [3.0710],
            [4.0742],
            [3.6305]]), target_rating: tensor([5, 3, 5, 4])
    model_output: tensor([[2.6482],
            [3.6961],
            [4.2475],
            [3.9668]]), target_rating: tensor([4, 2, 5, 5])
    model_output: tensor([[4.1807],
            [3.5661],
            [3.0480],
            [3.8150]]), target_rating: tensor([5, 3, 3, 5])
    model_output: tensor([[3.2704],
            [3.5916],
            [2.3747],
            [3.1612]]), target_rating: tensor([4, 4, 3, 2])
    model_output: tensor([[3.6852],
            [3.2633],
            [3.6638],
            [3.7518]]), target_rating: tensor([4, 3, 4, 4])
    model_output: tensor([[3.9357],
            [3.5724],
            [3.7634],
            [3.9613]]), target_rating: tensor([4, 2, 4, 4])
    model_output: tensor([[3.3616],
            [2.6437],
            [3.1210],
            [3.5761]]), target_rating: tensor([3, 1, 4, 3])
    model_output: tensor([[2.9333],
            [3.7519],
            [4.2889],
            [2.8730]]), target_rating: tensor([2, 4, 5, 3])
    model_output: tensor([[3.0585],
            [3.0119],
            [3.4198],
            [3.0519]]), target_rating: tensor([4, 2, 0, 5])
    model_output: tensor([[3.1722],
            [2.5521],
            [3.6997],
            [3.5543]]), target_rating: tensor([3, 3, 3, 5])
    model_output: tensor([[3.8052],
            [3.0135],
            [3.2203],
            [3.9883]]), target_rating: tensor([4, 4, 2, 5])
    model_output: tensor([[2.8793],
            [3.1606],
            [3.1028],
            [3.3536]]), target_rating: tensor([2, 4, 4, 4])
    model_output: tensor([[2.7637],
            [3.3708],
            [3.4243],
            [3.3112]]), target_rating: tensor([4, 1, 2, 2])
    model_output: tensor([[3.4815],
            [2.4004],
            [3.3021],
            [4.1684]]), target_rating: tensor([4, 2, 2, 4])
    model_output: tensor([[3.2729],
            [3.5788],
            [3.4566],
            [2.9184]]), target_rating: tensor([1, 3, 4, 2])
    model_output: tensor([[2.9569],
            [3.6844],
            [3.1229],
            [3.0148]]), target_rating: tensor([3, 1, 4, 3])
    model_output: tensor([[3.1936],
            [2.8468],
            [3.3326],
            [4.8808]]), target_rating: tensor([4, 3, 3, 4])
    model_output: tensor([[3.3959],
            [3.4069],
            [3.2230],
            [2.5852]]), target_rating: tensor([3, 3, 2, 3])
    model_output: tensor([[3.4599],
            [4.0026],
            [3.0687],
            [3.2576]]), target_rating: tensor([3, 4, 4, 2])
    model_output: tensor([[2.4380],
            [3.6913],
            [2.7395],
            [3.3869]]), target_rating: tensor([3, 2, 4, 4])
    model_output: tensor([[3.8017],
            [3.1829],
            [4.2225],
            [3.4120]]), target_rating: tensor([3, 4, 4, 4])
    model_output: tensor([[2.5470],
            [1.7075],
            [3.3668],
            [3.1824]]), target_rating: tensor([4, 1, 5, 3])
    model_output: tensor([[2.9908],
            [2.4220],
            [4.2819],
            [2.8377]]), target_rating: tensor([2, 3, 4, 3])
    model_output: tensor([[2.7336],
            [4.7887],
            [3.4611],
            [4.1677]]), target_rating: tensor([3, 4, 2, 3])
    model_output: tensor([[3.8839],
            [3.1506],
            [4.0014],
            [3.5983]]), target_rating: tensor([2, 2, 4, 5])
    model_output: tensor([[3.7731],
            [3.4419],
            [3.5117],
            [3.3763]]), target_rating: tensor([4, 3, 4, 4])
    model_output: tensor([[3.2746],
            [3.8156],
            [3.8527],
            [2.5063]]), target_rating: tensor([2, 2, 5, 3])
    model_output: tensor([[2.7314],
            [2.6470],
            [3.7950],
            [2.6263]]), target_rating: tensor([4, 4, 4, 1])
    model_output: tensor([[2.3802],
            [3.2337],
            [3.6840],
            [2.8246]]), target_rating: tensor([3, 3, 4, 3])
    model_output: tensor([[2.7564],
            [3.1630],
            [2.9714],
            [3.8617]]), target_rating: tensor([1, 3, 4, 5])
    model_output: tensor([[4.1571],
            [3.7915],
            [3.8569],
            [3.5984]]), target_rating: tensor([5, 5, 5, 4])
    model_output: tensor([[3.7461],
            [3.3651],
            [3.4543],
            [3.4627]]), target_rating: tensor([3, 5, 3, 4])
    model_output: tensor([[3.6831],
            [2.4237],
            [3.2794],
            [3.4475]]), target_rating: tensor([4, 3, 3, 4])
    model_output: tensor([[3.5027],
            [3.3037],
            [3.0906],
            [3.4923]]), target_rating: tensor([4, 3, 3, 3])
    model_output: tensor([[2.6253],
            [3.5918],
            [3.0627],
            [2.3636]]), target_rating: tensor([4, 5, 3, 2])
    model_output: tensor([[3.3280],
            [3.1523],
            [3.5265],
            [3.3390]]), target_rating: tensor([3, 3, 4, 2])
    model_output: tensor([[3.0532],
            [4.1079],
            [3.0710],
            [3.4371]]), target_rating: tensor([4, 4, 3, 0])
    model_output: tensor([[2.5850],
            [2.3437],
            [3.0594],
            [3.4977]]), target_rating: tensor([3, 4, 3, 1])
    model_output: tensor([[4.1221],
            [3.8147],
            [2.0785],
            [3.6738]]), target_rating: tensor([4, 0, 3, 4])
    model_output: tensor([[3.8152],
            [3.5667],
            [3.4516],
            [3.5401]]), target_rating: tensor([3, 4, 4, 4])
    model_output: tensor([[3.4549],
            [1.9675],
            [3.2763],
            [1.9599]]), target_rating: tensor([5, 2, 3, 4])
    model_output: tensor([[2.6966],
            [3.4430],
            [3.5845],
            [3.8313]]), target_rating: tensor([2, 3, 1, 4])
    model_output: tensor([[3.7270],
            [3.7738],
            [3.3898],
            [4.0007]]), target_rating: tensor([3, 5, 2, 4])
    model_output: tensor([[3.1226],
            [3.8823],
            [2.5012],
            [3.8665]]), target_rating: tensor([3, 3, 3, 4])
    model_output: tensor([[3.6536],
            [3.6185],
            [2.6578],
            [3.5167]]), target_rating: tensor([4, 5, 4, 3])
    model_output: tensor([[3.7731],
            [2.1533],
            [2.2683],
            [3.6733]]), target_rating: tensor([4, 1, 2, 5])
    model_output: tensor([[3.1709],
            [3.8969],
            [4.1576],
            [3.9403]]), target_rating: tensor([3, 3, 5, 4])
    model_output: tensor([[3.5435],
            [3.4907],
            [3.3205],
            [4.2878]]), target_rating: tensor([2, 4, 3, 5])
    model_output: tensor([[4.0741],
            [3.8642],
            [3.5263],
            [3.8934]]), target_rating: tensor([4, 3, 4, 2])
    model_output: tensor([[2.2105],
            [3.9612],
            [3.3463],
            [3.2714]]), target_rating: tensor([2, 4, 3, 3])
    model_output: tensor([[3.6588],
            [3.5380],
            [3.0205],
            [3.7273]]), target_rating: tensor([3, 4, 3, 3])
    model_output: tensor([[3.5078],
            [3.0297],
            [3.1926],
            [4.0034]]), target_rating: tensor([4, 3, 4, 2])
    model_output: tensor([[4.2433],
            [3.1829],
            [3.7585],
            [3.8111]]), target_rating: tensor([5, 1, 4, 3])
    model_output: tensor([[3.1161],
            [3.1195],
            [3.4899],
            [2.6267]]), target_rating: tensor([3, 4, 0, 2])
    model_output: tensor([[3.2741],
            [2.9700],
            [3.3476],
            [2.9567]]), target_rating: tensor([1, 3, 4, 3])
    model_output: tensor([[3.1988],
            [2.1285],
            [3.3519],
            [3.6221]]), target_rating: tensor([4, 3, 5, 4])
    model_output: tensor([[3.6121],
            [3.3696],
            [2.1197],
            [2.5224]]), target_rating: tensor([4, 2, 2, 3])
    model_output: tensor([[3.0218],
            [4.3407],
            [3.2794],
            [3.0321]]), target_rating: tensor([2, 5, 4, 3])
    model_output: tensor([[3.4803],
            [2.7040],
            [3.8643],
            [4.0632]]), target_rating: tensor([4, 4, 3, 4])
    model_output: tensor([[3.1772],
            [4.1326],
            [2.5263],
            [3.7999]]), target_rating: tensor([3, 4, 3, 5])
    model_output: tensor([[4.0440],
            [3.2590],
            [3.1826],
            [4.1829]]), target_rating: tensor([3, 2, 4, 4])
    model_output: tensor([[3.5052],
            [3.2182],
            [3.3577],
            [3.1099]]), target_rating: tensor([3, 2, 4, 2])
    model_output: tensor([[3.4631],
            [3.7198],
            [4.9623],
            [2.6237]]), target_rating: tensor([3, 3, 5, 2])
    model_output: tensor([[3.9628],
            [2.2480],
            [4.5167],
            [3.1475]]), target_rating: tensor([4, 1, 5, 4])
    model_output: tensor([[2.9810],
            [3.2323],
            [4.0834],
            [2.8495]]), target_rating: tensor([1, 4, 5, 3])
    model_output: tensor([[3.4896],
            [3.1042],
            [2.3104],
            [2.8716]]), target_rating: tensor([4, 1, 2, 4])
    model_output: tensor([[3.5630],
            [2.6853],
            [2.9627],
            [2.8352]]), target_rating: tensor([4, 3, 1, 3])
    model_output: tensor([[2.9748],
            [3.2809],
            [3.8714],
            [2.3631]]), target_rating: tensor([3, 3, 4, 3])
    model_output: tensor([[3.1009],
            [3.3097],
            [3.7728],
            [3.7665]]), target_rating: tensor([2, 4, 5, 3])
    model_output: tensor([[3.2602],
            [2.5319],
            [3.2147],
            [3.5157]]), target_rating: tensor([4, 3, 4, 4])
    model_output: tensor([[3.2331],
            [3.3965],
            [3.2870],
            [3.0961]]), target_rating: tensor([3, 5, 3, 4])
    model_output: tensor([[3.2023],
            [3.1241],
            [3.8816],
            [2.8072]]), target_rating: tensor([4, 3, 3, 3])
    model_output: tensor([[3.6437],
            [3.1128],
            [3.6154],
            [3.2682]]), target_rating: tensor([3, 3, 4, 3])
    model_output: tensor([[3.6954],
            [3.1806],
            [4.0505],
            [2.9946]]), target_rating: tensor([4, 3, 4, 4])
    model_output: tensor([[3.9342],
            [3.6756],
            [2.4778],
            [3.2578]]), target_rating: tensor([5, 3, 3, 3])
    model_output: tensor([[2.8871],
            [3.0363],
            [2.9445],
            [3.6430]]), target_rating: tensor([3, 2, 4, 3])
    model_output: tensor([[3.3131],
            [2.6415],
            [3.2763],
            [1.8987]]), target_rating: tensor([2, 3, 4, 0])
    model_output: tensor([[2.9194],
            [3.5476],
            [4.1749],
            [4.3569]]), target_rating: tensor([1, 3, 4, 5])
    model_output: tensor([[3.3644],
            [2.9023],
            [3.6803],
            [2.9529]]), target_rating: tensor([1, 2, 4, 3])
    model_output: tensor([[2.6139],
            [3.4737],
            [3.6470],
            [3.4114]]), target_rating: tensor([4, 4, 1, 5])
    model_output: tensor([[3.4593],
            [4.2947],
            [3.7402],
            [3.4078]]), target_rating: tensor([2, 5, 5, 4])
    model_output: tensor([[3.7310],
            [3.1890],
            [3.5498],
            [2.1971]]), target_rating: tensor([4, 4, 2, 2])
    model_output: tensor([[3.5928],
            [3.6676],
            [4.3367],
            [4.0654]]), target_rating: tensor([1, 4, 4, 4])
    model_output: tensor([[3.1236],
            [2.7170],
            [3.9818],
            [3.7293]]), target_rating: tensor([1, 3, 4, 5])
    model_output: tensor([[3.0430],
            [3.1119],
            [3.6482],
            [3.2114]]), target_rating: tensor([3, 4, 3, 4])
    model_output: tensor([[3.5284],
            [3.4692],
            [3.7294],
            [3.0009]]), target_rating: tensor([4, 4, 4, 3])
    model_output: tensor([[3.7554],
            [3.5130],
            [3.9054],
            [3.4454]]), target_rating: tensor([5, 3, 5, 4])
    model_output: tensor([[3.6180],
            [3.2286],
            [2.8916],
            [3.8008]]), target_rating: tensor([5, 3, 4, 4])
    model_output: tensor([[3.7554],
            [3.1998],
            [3.5716],
            [3.1620]]), target_rating: tensor([5, 4, 1, 4])
    model_output: tensor([[3.9834],
            [3.9642],
            [2.9043],
            [3.8956]]), target_rating: tensor([4, 4, 3, 4])
    model_output: tensor([[4.0658],
            [3.0571],
            [3.7542],
            [3.6185]]), target_rating: tensor([3, 2, 4, 3])
    model_output: tensor([[3.5799],
            [3.3816],
            [3.5039],
            [3.0642]]), target_rating: tensor([4, 3, 3, 3])
    model_output: tensor([[3.4215],
            [3.6820],
            [3.7450],
            [2.2979]]), target_rating: tensor([3, 4, 5, 4])
    model_output: tensor([[2.1387],
            [3.1988],
            [2.7272],
            [3.8526]]), target_rating: tensor([4, 4, 3, 4])
    model_output: tensor([[3.2360],
            [3.0020],
            [3.7727],
            [3.5602]]), target_rating: tensor([4, 3, 2, 3])
    model_output: tensor([[3.2399],
            [3.8233],
            [4.0488],
            [1.9716]]), target_rating: tensor([4, 5, 4, 0])
    model_output: tensor([[2.0989],
            [1.9737],
            [3.0370],
            [2.7682]]), target_rating: tensor([3, 2, 3, 2])
    model_output: tensor([[3.0656],
            [3.2580],
            [4.1814],
            [3.5400]]), target_rating: tensor([2, 4, 5, 3])
    model_output: tensor([[3.1445],
            [3.5926],
            [2.9308],
            [3.2785]]), target_rating: tensor([3, 5, 3, 4])
    model_output: tensor([[3.5356],
            [3.0497],
            [2.3662],
            [3.2265]]), target_rating: tensor([4, 2, 2, 4])
    model_output: tensor([[3.0270],
            [2.6908],
            [4.0945],
            [2.9355]]), target_rating: tensor([3, 3, 4, 5])
    model_output: tensor([[2.7624],
            [2.5361],
            [3.3405],
            [2.9581]]), target_rating: tensor([5, 1, 4, 3])
    model_output: tensor([[3.4109],
            [2.5050],
            [2.7266],
            [2.2351]]), target_rating: tensor([4, 2, 4, 1])
    model_output: tensor([[3.4700],
            [3.4836],
            [2.6677],
            [3.7174]]), target_rating: tensor([3, 5, 1, 3])
    model_output: tensor([[3.7250],
            [3.4766],
            [2.8624],
            [3.5488]]), target_rating: tensor([4, 4, 3, 4])
    model_output: tensor([[2.8546],
            [3.3647],
            [3.7957],
            [2.9211]]), target_rating: tensor([2, 4, 3, 3])
    model_output: tensor([[3.7676],
            [3.4696],
            [3.5187],
            [3.0333]]), target_rating: tensor([5, 3, 3, 2])
    model_output: tensor([[3.0151],
            [2.6098],
            [2.2362],
            [3.9288]]), target_rating: tensor([4, 4, 3, 4])
    model_output: tensor([[2.6058],
            [3.9175],
            [3.7307],
            [3.1070]]), target_rating: tensor([5, 4, 2, 4])
    model_output: tensor([[3.3110],
            [3.2496],
            [3.2178],
            [2.8924]]), target_rating: tensor([3, 3, 3, 3])
    model_output: tensor([[3.0698],
            [3.9836],
            [3.6862],
            [3.4238]]), target_rating: tensor([4, 4, 4, 3])
    model_output: tensor([[3.9298],
            [2.5889],
            [3.4123],
            [3.3450]]), target_rating: tensor([5, 3, 2, 3])
    model_output: tensor([[2.7040],
            [2.0732],
            [2.9280],
            [3.4065]]), target_rating: tensor([0, 3, 4, 3])
    model_output: tensor([[3.0370],
            [3.3020],
            [3.7218],
            [3.7103]]), target_rating: tensor([3, 3, 4, 4])
    model_output: tensor([[2.6747],
            [3.8467],
            [3.4775],
            [2.9046]]), target_rating: tensor([3, 4, 3, 3])
    model_output: tensor([[3.7957],
            [4.1018],
            [3.1877],
            [1.6613]]), target_rating: tensor([1, 4, 4, 1])
    model_output: tensor([[3.3126],
            [3.5955],
            [2.6439],
            [3.2752]]), target_rating: tensor([4, 5, 2, 4])
    model_output: tensor([[3.1111],
            [2.7334],
            [3.5325],
            [3.1456]]), target_rating: tensor([3, 3, 3, 2])
    model_output: tensor([[3.5201],
            [3.2674],
            [2.9232],
            [3.2655]]), target_rating: tensor([2, 4, 4, 4])
    model_output: tensor([[2.8146],
            [3.4993],
            [3.0268],
            [2.5230]]), target_rating: tensor([2, 3, 4, 2])
    model_output: tensor([[3.7427],
            [3.7623],
            [2.4455],
            [4.0393]]), target_rating: tensor([4, 4, 2, 4])
    model_output: tensor([[3.9014],
            [3.3835],
            [2.7693],
            [2.5591]]), target_rating: tensor([3, 4, 2, 3])
    model_output: tensor([[2.9001],
            [2.9461],
            [3.3139],
            [3.6138]]), target_rating: tensor([3, 3, 3, 4])
    model_output: tensor([[2.9658],
            [3.9828],
            [2.6447],
            [2.8332]]), target_rating: tensor([4, 4, 5, 3])
    model_output: tensor([[3.2988],
            [3.6347],
            [3.3114],
            [3.1933]]), target_rating: tensor([4, 4, 3, 4])
    model_output: tensor([[2.9886],
            [3.7207],
            [3.2869],
            [4.6984]]), target_rating: tensor([2, 5, 2, 5])
    model_output: tensor([[2.7572],
            [2.3367],
            [3.0784],
            [3.1247]]), target_rating: tensor([2, 3, 3, 4])
    model_output: tensor([[3.1302],
            [2.8041],
            [2.5995],
            [2.9604]]), target_rating: tensor([2, 1, 2, 4])
    model_output: tensor([[3.5949],
            [4.0575],
            [3.5488],
            [2.9726]]), target_rating: tensor([3, 5, 4, 3])
    model_output: tensor([[3.2269],
            [4.0787],
            [3.5334],
            [3.2923]]), target_rating: tensor([3, 3, 5, 4])
    model_output: tensor([[3.2991],
            [4.0584],
            [2.6058],
            [3.9481]]), target_rating: tensor([2, 5, 3, 4])
    model_output: tensor([[3.5992],
            [2.9814],
            [3.4489],
            [2.8686]]), target_rating: tensor([4, 3, 3, 2])
    model_output: tensor([[3.5108],
            [4.1743],
            [4.0031],
            [4.0061]]), target_rating: tensor([5, 4, 4, 3])
    model_output: tensor([[4.1249],
            [2.6762],
            [2.2151],
            [1.8663]]), target_rating: tensor([4, 2, 1, 2])
    model_output: tensor([[3.2256],
            [2.6036],
            [3.3670],
            [2.7543]]), target_rating: tensor([3, 2, 3, 2])
    model_output: tensor([[3.2765],
            [1.8451],
            [3.5717],
            [3.3055]]), target_rating: tensor([2, 5, 5, 3])
    model_output: tensor([[3.7221],
            [2.4798],
            [3.1309],
            [3.7381]]), target_rating: tensor([3, 4, 4, 4])
    model_output: tensor([[3.6930],
            [2.3392],
            [3.5827],
            [3.5022]]), target_rating: tensor([3, 3, 5, 3])
    model_output: tensor([[4.3153],
            [3.0329],
            [1.7292],
            [2.4593]]), target_rating: tensor([5, 2, 1, 1])
    model_output: tensor([[3.3660],
            [3.0537],
            [3.0700],
            [2.7170]]), target_rating: tensor([3, 3, 3, 2])
    model_output: tensor([[3.6087],
            [3.3438],
            [3.0066],
            [3.2971]]), target_rating: tensor([4, 4, 4, 1])
    model_output: tensor([[3.0305],
            [2.7730],
            [3.2224],
            [2.8748]]), target_rating: tensor([3, 3, 4, 4])
    model_output: tensor([[4.0643],
            [3.6080],
            [2.7369],
            [3.0942]]), target_rating: tensor([4, 5, 2, 2])
    model_output: tensor([[2.8464],
            [2.9587],
            [2.9225],
            [3.4125]]), target_rating: tensor([3, 3, 2, 3])
    model_output: tensor([[3.1345],
            [2.7440],
            [3.4364],
            [2.9232]]), target_rating: tensor([5, 3, 3, 2])
    model_output: tensor([[3.2151],
            [2.3231],
            [3.2939],
            [3.5604]]), target_rating: tensor([3, 1, 3, 3])
    model_output: tensor([[3.1370],
            [4.2056],
            [3.9346],
            [3.2647]]), target_rating: tensor([3, 3, 4, 4])
    model_output: tensor([[2.9735],
            [3.6957],
            [3.0106],
            [3.5708]]), target_rating: tensor([3, 4, 3, 4])
    model_output: tensor([[3.2058],
            [3.3465],
            [2.8662],
            [3.0792]]), target_rating: tensor([3, 3, 4, 4])
    model_output: tensor([[3.5920],
            [3.2718],
            [3.0226],
            [3.5303]]), target_rating: tensor([3, 4, 3, 4])
    model_output: tensor([[3.2769],
            [3.4379],
            [4.2907],
            [2.7737]]), target_rating: tensor([4, 4, 4, 3])
    model_output: tensor([[2.7772],
            [3.2204],
            [3.8095],
            [2.6443]]), target_rating: tensor([3, 3, 4, 3])
    model_output: tensor([[3.2974],
            [2.3829],
            [3.3983],
            [2.3503]]), target_rating: tensor([4, 1, 3, 4])
    model_output: tensor([[2.4971],
            [3.1004],
            [3.6723],
            [2.8165]]), target_rating: tensor([3, 2, 4, 3])
    model_output: tensor([[3.3212],
            [2.1960],
            [4.2152],
            [3.4435]]), target_rating: tensor([3, 2, 5, 2])
    model_output: tensor([[3.1745],
            [4.1787],
            [4.1012],
            [2.9152]]), target_rating: tensor([3, 4, 5, 4])
    model_output: tensor([[2.6901],
            [3.5353],
            [3.0078],
            [2.5479]]), target_rating: tensor([4, 3, 3, 2])
    model_output: tensor([[3.3342],
            [3.2131],
            [3.8148],
            [4.1263]]), target_rating: tensor([2, 3, 3, 4])
    model_output: tensor([[2.8259],
            [3.1462],
            [3.9379],
            [3.8138]]), target_rating: tensor([3, 5, 4, 4])
    model_output: tensor([[3.6508],
            [3.2431],
            [3.7020],
            [3.2662]]), target_rating: tensor([4, 4, 3, 3])
    model_output: tensor([[3.1725],
            [3.6323],
            [3.1336],
            [3.3826]]), target_rating: tensor([5, 3, 3, 3])
    model_output: tensor([[4.2524],
            [2.9706],
            [3.5508],
            [3.2525]]), target_rating: tensor([5, 2, 3, 3])
    model_output: tensor([[3.3200],
            [3.3441],
            [3.4575],
            [3.0441]]), target_rating: tensor([4, 3, 4, 3])
    model_output: tensor([[3.6880],
            [3.1527],
            [3.1813],
            [3.5426]]), target_rating: tensor([0, 2, 3, 3])
    model_output: tensor([[3.6984],
            [3.0868],
            [3.9863],
            [3.0911]]), target_rating: tensor([3, 2, 4, 3])
    model_output: tensor([[3.3093],
            [2.5079],
            [3.7666],
            [3.2343]]), target_rating: tensor([3, 2, 4, 3])
    model_output: tensor([[3.6282],
            [3.5437],
            [2.9043],
            [3.6386]]), target_rating: tensor([5, 4, 3, 4])
    model_output: tensor([[3.0020],
            [3.6641],
            [3.0885],
            [3.1500]]), target_rating: tensor([3, 5, 4, 4])
    model_output: tensor([[2.8737],
            [2.6699],
            [3.5948],
            [2.9536]]), target_rating: tensor([1, 2, 4, 3])
    model_output: tensor([[4.0327],
            [3.5476],
            [2.7825],
            [3.8728]]), target_rating: tensor([3, 3, 3, 4])
    model_output: tensor([[2.4556],
            [3.7964],
            [2.9349],
            [3.6857]]), target_rating: tensor([3, 4, 3, 5])
    model_output: tensor([[3.8403],
            [3.6452],
            [3.2995],
            [3.3584]]), target_rating: tensor([3, 4, 4, 5])
    model_output: tensor([[2.9241],
            [3.5529],
            [2.8054],
            [2.8033]]), target_rating: tensor([4, 5, 2, 4])
    model_output: tensor([[3.9773],
            [2.9110],
            [3.2319],
            [2.7525]]), target_rating: tensor([4, 3, 4, 1])
    model_output: tensor([[3.5644],
            [2.5877],
            [3.6852],
            [3.5813]]), target_rating: tensor([1, 3, 4, 3])
    model_output: tensor([[2.9412],
            [3.9578],
            [2.8139],
            [3.0293]]), target_rating: tensor([3, 4, 4, 3])
    model_output: tensor([[3.8437],
            [3.2701],
            [2.6757],
            [3.0047]]), target_rating: tensor([4, 3, 4, 4])
    model_output: tensor([[3.4919],
            [3.0383],
            [2.8481],
            [3.0068]]), target_rating: tensor([4, 5, 4, 2])
    model_output: tensor([[3.5068],
            [2.2274],
            [3.8738],
            [2.8498]]), target_rating: tensor([4, 3, 3, 5])
    model_output: tensor([[2.7846],
            [3.5973],
            [2.7844],
            [3.6470]]), target_rating: tensor([1, 2, 2, 5])
    model_output: tensor([[3.6832],
            [3.4302],
            [3.8112],
            [3.0015]]), target_rating: tensor([3, 4, 3, 2])
    model_output: tensor([[3.0259],
            [4.1948],
            [2.6930],
            [3.4228]]), target_rating: tensor([3, 5, 4, 1])
    model_output: tensor([[3.5315],
            [3.0200],
            [2.8478],
            [3.9225]]), target_rating: tensor([3, 3, 4, 5])
    model_output: tensor([[1.8722],
            [3.7744],
            [2.9798],
            [3.4722]]), target_rating: tensor([3, 2, 3, 3])
    model_output: tensor([[3.6154],
            [2.9234],
            [3.9625],
            [3.4217]]), target_rating: tensor([1, 3, 3, 4])
    model_output: tensor([[3.9556],
            [4.1659],
            [4.3668],
            [3.9332]]), target_rating: tensor([3, 5, 4, 4])
    model_output: tensor([[4.1672],
            [3.4360],
            [3.1849],
            [4.2284]]), target_rating: tensor([5, 4, 4, 4])
    model_output: tensor([[3.2093],
            [3.9662],
            [3.6882],
            [4.0879]]), target_rating: tensor([3, 4, 4, 5])
    model_output: tensor([[3.7387],
            [3.3623],
            [2.6160],
            [3.4499]]), target_rating: tensor([3, 3, 2, 1])
    model_output: tensor([[3.7324],
            [2.8750],
            [2.7938],
            [4.0323]]), target_rating: tensor([4, 4, 2, 4])
    model_output: tensor([[3.2208],
            [3.7286],
            [3.1109],
            [4.1780]]), target_rating: tensor([4, 4, 3, 5])
    model_output: tensor([[4.3760],
            [4.1650],
            [3.6728],
            [3.3801]]), target_rating: tensor([5, 4, 4, 2])
    model_output: tensor([[2.2337],
            [2.9141],
            [3.3766],
            [3.4003]]), target_rating: tensor([3, 2, 3, 3])
    model_output: tensor([[3.4283],
            [3.6680],
            [2.8772],
            [3.3050]]), target_rating: tensor([3, 4, 3, 2])
    model_output: tensor([[3.7247],
            [4.1531],
            [3.6876],
            [4.2804]]), target_rating: tensor([5, 3, 3, 4])
    model_output: tensor([[3.8983],
            [3.2520],
            [3.0933],
            [3.4562]]), target_rating: tensor([2, 3, 2, 4])
    model_output: tensor([[2.3324],
            [3.5410],
            [3.5109],
            [2.8727]]), target_rating: tensor([3, 4, 2, 4])
    model_output: tensor([[3.9313],
            [4.2281],
            [3.2956],
            [3.6764]]), target_rating: tensor([4, 4, 5, 5])
    model_output: tensor([[2.9349],
            [3.9941],
            [4.1524],
            [3.9461]]), target_rating: tensor([3, 4, 5, 3])
    model_output: tensor([[3.7516],
            [3.2113],
            [3.2575],
            [3.3986]]), target_rating: tensor([3, 3, 3, 2])
    model_output: tensor([[3.3865],
            [1.8918],
            [4.0005],
            [2.5259]]), target_rating: tensor([4, 2, 5, 4])
    model_output: tensor([[3.0273],
            [2.4402],
            [3.4549],
            [3.5834]]), target_rating: tensor([4, 3, 1, 5])
    model_output: tensor([[3.0925],
            [3.4410],
            [4.1510],
            [3.0257]]), target_rating: tensor([4, 3, 5, 4])
    model_output: tensor([[2.6392],
            [2.9927],
            [3.8443],
            [3.3600]]), target_rating: tensor([4, 3, 5, 2])
    model_output: tensor([[2.6562],
            [3.4552],
            [3.9824],
            [3.0576]]), target_rating: tensor([3, 1, 5, 3])
    model_output: tensor([[3.2612],
            [3.6408],
            [2.8122],
            [3.2545]]), target_rating: tensor([3, 5, 4, 4])
    model_output: tensor([[4.2247],
            [2.9643],
            [4.3014],
            [3.4162]]), target_rating: tensor([5, 3, 4, 3])
    model_output: tensor([[3.1698],
            [2.4617],
            [3.5963],
            [4.8177]]), target_rating: tensor([3, 3, 4, 5])
    model_output: tensor([[3.4833],
            [3.7433],
            [3.1249],
            [3.8282]]), target_rating: tensor([4, 4, 4, 4])
    model_output: tensor([[3.5754],
            [3.6599],
            [3.9284],
            [3.4784]]), target_rating: tensor([3, 4, 3, 5])
    model_output: tensor([[3.9347],
            [3.5301],
            [3.0123],
            [3.3027]]), target_rating: tensor([5, 4, 4, 4])
    model_output: tensor([[3.6220],
            [3.6638],
            [3.8187],
            [3.0779]]), target_rating: tensor([4, 2, 5, 3])
    model_output: tensor([[2.6993],
            [2.6968],
            [3.1523],
            [3.6516]]), target_rating: tensor([3, 2, 5, 4])
    model_output: tensor([[3.5214],
            [3.2409],
            [3.1874],
            [3.1705]]), target_rating: tensor([4, 4, 3, 2])
    model_output: tensor([[2.1214],
            [2.3558],
            [3.6964],
            [2.8546]]), target_rating: tensor([2, 4, 4, 4])
    model_output: tensor([[3.9512],
            [2.9694],
            [4.1824],
            [3.7434]]), target_rating: tensor([4, 3, 4, 3])
    model_output: tensor([[2.5986],
            [2.4588],
            [4.3582],
            [3.1293]]), target_rating: tensor([3, 2, 4, 4])
    model_output: tensor([[3.0561],
            [3.9041],
            [3.9325],
            [3.5586]]), target_rating: tensor([1, 2, 5, 3])
    model_output: tensor([[3.1712],
            [3.6980],
            [3.6945],
            [1.9532]]), target_rating: tensor([3, 3, 5, 2])
    model_output: tensor([[3.3135],
            [3.8554],
            [3.6895],
            [3.5163]]), target_rating: tensor([2, 4, 4, 2])
    model_output: tensor([[3.6342],
            [2.7690],
            [3.6120],
            [2.8683]]), target_rating: tensor([4, 3, 3, 3])
    model_output: tensor([[2.0049],
            [3.0212],
            [3.5360],
            [3.5650]]), target_rating: tensor([2, 3, 1, 3])
    model_output: tensor([[3.2758],
            [2.9570],
            [3.0448],
            [2.9580]]), target_rating: tensor([2, 3, 4, 4])
    model_output: tensor([[3.9706],
            [2.9370],
            [3.6676],
            [3.0134]]), target_rating: tensor([4, 3, 4, 5])
    model_output: tensor([[3.3735],
            [2.6909],
            [3.1035],
            [4.6162]]), target_rating: tensor([3, 3, 3, 5])
    model_output: tensor([[2.6018],
            [3.2758],
            [3.5321],
            [3.1121]]), target_rating: tensor([3, 3, 5, 2])
    model_output: tensor([[3.2015],
            [3.2380],
            [3.5721],
            [2.6612]]), target_rating: tensor([3, 3, 3, 3])
    model_output: tensor([[1.7553],
            [3.1305],
            [2.8528],
            [3.6870]]), target_rating: tensor([0, 2, 3, 5])
    model_output: tensor([[3.8766],
            [3.8644],
            [2.4431],
            [3.7481]]), target_rating: tensor([4, 4, 3, 4])
    model_output: tensor([[3.6608],
            [2.8762],
            [2.9648],
            [3.6178]]), target_rating: tensor([2, 3, 4, 4])
    model_output: tensor([[3.3269],
            [3.6169],
            [3.0777],
            [3.4113]]), target_rating: tensor([2, 5, 4, 3])
    model_output: tensor([[4.1200],
            [3.1292],
            [2.5821],
            [3.4541]]), target_rating: tensor([4, 2, 2, 4])
    model_output: tensor([[2.9415],
            [2.5432],
            [3.0825],
            [3.0155]]), target_rating: tensor([2, 2, 2, 4])
    model_output: tensor([[3.0668],
            [3.4016],
            [2.2087],
            [3.7663]]), target_rating: tensor([4, 3, 2, 3])
    model_output: tensor([[3.8187],
            [3.3257],
            [3.1478],
            [3.5191]]), target_rating: tensor([3, 3, 3, 3])
    model_output: tensor([[4.3994],
            [4.0049],
            [3.4449],
            [3.8824]]), target_rating: tensor([4, 4, 4, 3])
    model_output: tensor([[3.3885],
            [3.4061],
            [3.1249],
            [3.7165]]), target_rating: tensor([2, 4, 3, 4])
    model_output: tensor([[3.3671],
            [2.8216],
            [2.5304],
            [3.1292]]), target_rating: tensor([3, 2, 1, 4])
    model_output: tensor([[4.9307],
            [2.9982],
            [3.5739],
            [3.3838]]), target_rating: tensor([5, 4, 3, 4])
    model_output: tensor([[3.6234],
            [4.2999],
            [2.8358],
            [3.5921]]), target_rating: tensor([5, 5, 5, 3])
    model_output: tensor([[3.4272],
            [3.1072],
            [2.9998],
            [2.5433]]), target_rating: tensor([3, 0, 2, 3])
    model_output: tensor([[3.1285],
            [3.4361],
            [2.6475],
            [2.5464]]), target_rating: tensor([4, 2, 2, 3])
    model_output: tensor([[3.6330],
            [3.4832],
            [2.9827],
            [3.8129]]), target_rating: tensor([5, 2, 3, 3])
    model_output: tensor([[2.8000],
            [3.5988],
            [3.2900],
            [3.2190]]), target_rating: tensor([4, 2, 2, 5])
    model_output: tensor([[4.0841],
            [2.9242],
            [3.3465],
            [3.5030]]), target_rating: tensor([5, 2, 4, 3])
    model_output: tensor([[3.7437],
            [3.1857],
            [3.7745],
            [2.7897]]), target_rating: tensor([4, 2, 5, 4])
    model_output: tensor([[3.9601],
            [3.4426],
            [2.1314],
            [2.3765]]), target_rating: tensor([5, 3, 4, 2])
    model_output: tensor([[2.5907],
            [3.2909],
            [3.2252],
            [3.3291]]), target_rating: tensor([3, 2, 2, 4])
    model_output: tensor([[3.5297],
            [3.3131],
            [3.5611],
            [4.7258]]), target_rating: tensor([4, 4, 3, 5])
    model_output: tensor([[4.3934],
            [3.9626],
            [2.9443],
            [3.4969]]), target_rating: tensor([5, 5, 4, 3])
    model_output: tensor([[3.3813],
            [4.1288],
            [2.8265],
            [2.7353]]), target_rating: tensor([4, 1, 4, 1])
    model_output: tensor([[2.3686],
            [3.4419],
            [3.6759],
            [3.6722]]), target_rating: tensor([3, 4, 2, 4])
    model_output: tensor([[3.8054],
            [2.9499],
            [4.1165],
            [3.5486]]), target_rating: tensor([5, 4, 4, 4])
    model_output: tensor([[3.3188],
            [3.8395],
            [3.0659],
            [3.8193]]), target_rating: tensor([3, 4, 4, 4])
    model_output: tensor([[4.3474],
            [3.5041],
            [3.2883],
            [2.9765]]), target_rating: tensor([4, 3, 4, 2])
    model_output: tensor([[4.2724],
            [3.3678],
            [3.6698],
            [3.2281]]), target_rating: tensor([4, 4, 4, 5])
    model_output: tensor([[3.0938],
            [4.3752],
            [3.1158],
            [2.7551]]), target_rating: tensor([2, 5, 2, 2])
    model_output: tensor([[2.6589],
            [3.5407],
            [3.2457],
            [3.2893]]), target_rating: tensor([2, 5, 4, 5])
    model_output: tensor([[3.3973],
            [3.6111],
            [3.0537],
            [3.3266]]), target_rating: tensor([2, 3, 4, 4])
    model_output: tensor([[2.2576],
            [2.9027],
            [2.3274],
            [2.9669]]), target_rating: tensor([2, 4, 2, 3])
    model_output: tensor([[4.0020],
            [2.5708],
            [4.1605],
            [3.0166]]), target_rating: tensor([5, 1, 4, 4])
    model_output: tensor([[2.9076],
            [3.4234],
            [2.8358],
            [3.0023]]), target_rating: tensor([3, 4, 3, 2])
    model_output: tensor([[3.1977],
            [3.1237],
            [1.6393],
            [2.9795]]), target_rating: tensor([4, 0, 2, 4])
    model_output: tensor([[3.1214],
            [3.2245],
            [2.7284],
            [2.5479]]), target_rating: tensor([3, 2, 2, 3])
    model_output: tensor([[3.6797],
            [3.2239],
            [3.1060],
            [2.8558]]), target_rating: tensor([3, 3, 2, 3])
    model_output: tensor([[3.3760],
            [3.1961],
            [3.8090],
            [2.3528]]), target_rating: tensor([3, 4, 4, 3])
    model_output: tensor([[3.3555],
            [3.6558],
            [3.7088],
            [3.1586]]), target_rating: tensor([3, 4, 4, 1])
    model_output: tensor([[3.8429],
            [3.0472],
            [3.3170],
            [2.3248]]), target_rating: tensor([4, 3, 3, 3])
    model_output: tensor([[3.8346],
            [4.2177],
            [3.4470],
            [2.4721]]), target_rating: tensor([3, 5, 4, 3])
    model_output: tensor([[2.7533],
            [2.6280],
            [2.4911],
            [3.5163]]), target_rating: tensor([4, 3, 2, 3])
    model_output: tensor([[1.8816],
            [3.9789],
            [3.1744],
            [4.0143]]), target_rating: tensor([3, 5, 4, 4])
    model_output: tensor([[3.6861],
            [3.2517],
            [3.2987],
            [2.7602]]), target_rating: tensor([4, 4, 3, 3])
    model_output: tensor([[3.4624],
            [3.5759],
            [3.0776],
            [4.3061]]), target_rating: tensor([3, 5, 4, 5])
    model_output: tensor([[3.4853],
            [3.3035],
            [2.9527],
            [3.6118]]), target_rating: tensor([4, 4, 2, 2])
    model_output: tensor([[3.0319],
            [3.5199],
            [3.5355],
            [2.8855]]), target_rating: tensor([3, 4, 3, 3])
    model_output: tensor([[4.0812],
            [2.7936],
            [2.2462],
            [4.1035]]), target_rating: tensor([2, 4, 3, 4])
    model_output: tensor([[2.6832],
            [2.1670],
            [3.7684],
            [3.7766]]), target_rating: tensor([2, 2, 3, 4])
    model_output: tensor([[2.1534],
            [2.2692],
            [3.5965],
            [2.7905]]), target_rating: tensor([1, 3, 3, 4])
    model_output: tensor([[3.5994],
            [2.0293],
            [4.3340],
            [2.3641]]), target_rating: tensor([4, 4, 5, 2])
    model_output: tensor([[3.8814],
            [4.3272],
            [2.9920],
            [4.0475]]), target_rating: tensor([4, 3, 2, 4])
    model_output: tensor([[3.4109],
            [4.6971],
            [2.4947],
            [2.4778]]), target_rating: tensor([4, 4, 2, 2])
    model_output: tensor([[3.5416],
            [2.5619],
            [3.6702],
            [3.6015]]), target_rating: tensor([4, 3, 5, 4])
    model_output: tensor([[3.5428],
            [4.4599],
            [3.5312],
            [3.7514]]), target_rating: tensor([3, 5, 4, 4])
    model_output: tensor([[3.4603],
            [2.6981],
            [3.4111],
            [3.2107]]), target_rating: tensor([3, 3, 4, 3])
    model_output: tensor([[3.1630],
            [3.5902],
            [4.3025],
            [3.2194]]), target_rating: tensor([4, 3, 5, 2])
    model_output: tensor([[3.2520],
            [3.1501],
            [3.4026],
            [2.6265]]), target_rating: tensor([3, 3, 3, 2])
    model_output: tensor([[3.5172],
            [3.4720],
            [3.8350],
            [4.5951]]), target_rating: tensor([4, 3, 5, 5])
    model_output: tensor([[2.6570],
            [4.2180],
            [2.7836],
            [3.7966]]), target_rating: tensor([3, 5, 1, 4])
    model_output: tensor([[3.1945],
            [3.5340],
            [2.8585],
            [3.4579]]), target_rating: tensor([3, 4, 4, 4])
    model_output: tensor([[3.4200],
            [3.6717],
            [3.0588],
            [3.0472]]), target_rating: tensor([5, 4, 3, 4])
    model_output: tensor([[3.2410],
            [2.8882],
            [4.2466],
            [3.6041]]), target_rating: tensor([3, 3, 4, 4])
    model_output: tensor([[3.3591],
            [3.0348],
            [3.1769],
            [3.3506]]), target_rating: tensor([5, 5, 3, 4])
    model_output: tensor([[2.8474],
            [3.5970],
            [3.1838],
            [2.7314]]), target_rating: tensor([4, 3, 3, 4])
    model_output: tensor([[3.2442],
            [2.9718],
            [1.9408],
            [3.4068]]), target_rating: tensor([2, 3, 2, 4])
    model_output: tensor([[2.9645],
            [2.6928],
            [3.7984],
            [2.6830]]), target_rating: tensor([4, 1, 3, 3])
    model_output: tensor([[3.0488],
            [3.3632],
            [2.2079],
            [4.0057]]), target_rating: tensor([4, 3, 4, 4])
    model_output: tensor([[3.0569],
            [3.7737],
            [3.5912],
            [3.5878]]), target_rating: tensor([2, 2, 3, 4])
    model_output: tensor([[2.6053],
            [3.6449],
            [2.9007],
            [3.4226]]), target_rating: tensor([4, 4, 4, 2])
    model_output: tensor([[3.8120],
            [3.1787],
            [3.5494],
            [3.4789]]), target_rating: tensor([4, 2, 3, 3])
    model_output: tensor([[3.0686],
            [3.8772],
            [2.9321],
            [3.4323]]), target_rating: tensor([3, 4, 3, 3])
    model_output: tensor([[3.2488],
            [3.5344],
            [3.0978],
            [2.9713]]), target_rating: tensor([4, 4, 3, 3])
    model_output: tensor([[1.8978],
            [2.6505],
            [3.7776],
            [3.5105]]), target_rating: tensor([1, 3, 3, 4])
    model_output: tensor([[3.2898],
            [4.1124],
            [2.8891],
            [3.9791]]), target_rating: tensor([4, 4, 3, 4])
    model_output: tensor([[3.6712],
            [3.9931],
            [3.3006],
            [3.6332]]), target_rating: tensor([3, 4, 3, 3])
    model_output: tensor([[3.7932],
            [3.4542],
            [4.1695],
            [3.6322]]), target_rating: tensor([4, 4, 5, 4])
    model_output: tensor([[3.9918],
            [3.5139],
            [3.2393],
            [3.5105]]), target_rating: tensor([5, 4, 4, 3])
    model_output: tensor([[4.0532],
            [3.2701],
            [3.7016],
            [2.7934]]), target_rating: tensor([4, 2, 3, 3])
    model_output: tensor([[3.7668],
            [2.5268],
            [3.3768],
            [3.2861]]), target_rating: tensor([3, 0, 2, 3])
    model_output: tensor([[2.5040],
            [3.5198],
            [4.1196],
            [3.1196]]), target_rating: tensor([3, 4, 5, 2])
    model_output: tensor([[3.2270],
            [4.0575],
            [3.0697],
            [2.6600]]), target_rating: tensor([3, 5, 4, 0])
    model_output: tensor([[3.1689],
            [4.0721],
            [3.4404],
            [2.3527]]), target_rating: tensor([1, 4, 4, 3])
    model_output: tensor([[3.6553],
            [2.4915],
            [1.7124],
            [2.6358]]), target_rating: tensor([5, 2, 3, 1])
    model_output: tensor([[3.2808],
            [3.2476],
            [3.5645],
            [2.5929]]), target_rating: tensor([5, 1, 3, 3])
    model_output: tensor([[4.3570],
            [3.0747],
            [3.1346],
            [3.7514]]), target_rating: tensor([5, 3, 4, 4])
    model_output: tensor([[3.4372],
            [3.1077],
            [3.5511],
            [4.4016]]), target_rating: tensor([5, 3, 4, 5])
    model_output: tensor([[3.8381],
            [2.9262],
            [3.4660],
            [3.4787]]), target_rating: tensor([3, 3, 2, 4])
    model_output: tensor([[3.7568],
            [3.2031],
            [3.1399],
            [3.2049]]), target_rating: tensor([4, 4, 3, 3])
    model_output: tensor([[3.1049],
            [4.0474],
            [3.7157],
            [3.7176]]), target_rating: tensor([3, 5, 4, 3])
    model_output: tensor([[4.0684],
            [3.3593],
            [4.3012],
            [2.7951]]), target_rating: tensor([1, 4, 5, 3])
    model_output: tensor([[3.3684],
            [3.3713],
            [3.5729],
            [2.9757]]), target_rating: tensor([4, 4, 4, 4])
    model_output: tensor([[2.6309],
            [3.5213],
            [3.2384],
            [3.2075]]), target_rating: tensor([2, 3, 3, 3])
    model_output: tensor([[3.2115],
            [3.7721],
            [3.2350],
            [2.3706]]), target_rating: tensor([2, 3, 2, 1])
    model_output: tensor([[3.2578],
            [3.2588],
            [3.1111],
            [3.5989]]), target_rating: tensor([4, 4, 3, 3])
    model_output: tensor([[3.9033],
            [3.6577],
            [2.6763],
            [3.3528]]), target_rating: tensor([4, 4, 4, 4])
    model_output: tensor([[3.0392],
            [3.1229],
            [3.6772],
            [3.0831]]), target_rating: tensor([3, 4, 3, 3])
    model_output: tensor([[3.6597],
            [2.8912],
            [2.5977],
            [3.6638]]), target_rating: tensor([2, 4, 2, 3])
    model_output: tensor([[3.0778],
            [3.7500],
            [2.9549],
            [3.3125]]), target_rating: tensor([2, 3, 3, 3])
    model_output: tensor([[2.9621],
            [3.6212],
            [3.7844],
            [3.0315]]), target_rating: tensor([5, 3, 4, 3])
    model_output: tensor([[4.7009],
            [3.3884],
            [3.4814],
            [3.7295]]), target_rating: tensor([4, 3, 4, 3])
    model_output: tensor([[3.9517],
            [3.7848],
            [3.9433],
            [3.7820]]), target_rating: tensor([2, 4, 3, 5])
    model_output: tensor([[4.1552],
            [3.2267],
            [2.7725],
            [3.4342]]), target_rating: tensor([5, 2, 3, 5])
    model_output: tensor([[4.1107],
            [4.2336],
            [3.3526],
            [2.7345]]), target_rating: tensor([2, 3, 5, 5])
    model_output: tensor([[2.7608],
            [2.5276],
            [3.3155],
            [3.4143]]), target_rating: tensor([3, 1, 4, 4])
    model_output: tensor([[3.3871],
            [3.1516],
            [3.7666],
            [3.7043]]), target_rating: tensor([3, 5, 4, 5])
    model_output: tensor([[3.8073],
            [3.7862],
            [3.1923],
            [4.0015]]), target_rating: tensor([5, 5, 4, 4])
    model_output: tensor([[3.3192],
            [2.8117],
            [4.1044],
            [2.6681]]), target_rating: tensor([5, 4, 4, 2])
    model_output: tensor([[3.3227],
            [3.5609],
            [3.1586],
            [3.7776]]), target_rating: tensor([4, 4, 3, 3])
    model_output: tensor([[2.3971],
            [2.5717],
            [3.0804],
            [2.9874]]), target_rating: tensor([1, 2, 4, 3])
    model_output: tensor([[3.2689],
            [3.2074],
            [4.1847],
            [3.8700]]), target_rating: tensor([4, 5, 5, 4])
    model_output: tensor([[3.4337],
            [2.7532],
            [2.3931],
            [3.7207]]), target_rating: tensor([3, 3, 2, 4])
    model_output: tensor([[2.8169],
            [3.0352],
            [2.8864],
            [4.0102]]), target_rating: tensor([3, 4, 1, 3])
    model_output: tensor([[2.7296],
            [3.8315],
            [2.8455],
            [3.1347]]), target_rating: tensor([4, 2, 3, 4])
    model_output: tensor([[3.6254],
            [3.4804],
            [3.5558],
            [3.7522]]), target_rating: tensor([2, 2, 4, 5])
    model_output: tensor([[4.4136],
            [2.9110],
            [3.4383],
            [2.7503]]), target_rating: tensor([2, 4, 2, 3])
    model_output: tensor([[4.2320],
            [3.4147],
            [2.6160],
            [2.0168]]), target_rating: tensor([5, 3, 3, 0])
    model_output: tensor([[3.7430],
            [3.1362],
            [3.5069],
            [3.7675]]), target_rating: tensor([3, 3, 3, 3])
    model_output: tensor([[2.9109],
            [3.8388],
            [3.6558],
            [2.3504]]), target_rating: tensor([3, 5, 3, 0])
    model_output: tensor([[3.0977],
            [3.3570],
            [3.6543],
            [3.1247]]), target_rating: tensor([3, 3, 3, 4])
    model_output: tensor([[4.0099],
            [3.0305],
            [4.0445],
            [3.4173]]), target_rating: tensor([5, 3, 5, 4])
    model_output: tensor([[2.8180],
            [4.1124],
            [3.8900],
            [3.2766]]), target_rating: tensor([4, 3, 4, 3])
    model_output: tensor([[2.6268],
            [3.2567],
            [3.7951],
            [3.4879]]), target_rating: tensor([2, 4, 3, 3])
    model_output: tensor([[3.1316],
            [3.0709],
            [4.3110],
            [2.4762]]), target_rating: tensor([3, 2, 5, 3])
    model_output: tensor([[3.5652],
            [4.2087],
            [3.7522],
            [2.8574]]), target_rating: tensor([4, 4, 3, 2])
    model_output: tensor([[2.6875],
            [3.6215],
            [3.5726],
            [4.0859]]), target_rating: tensor([3, 4, 3, 3])
    model_output: tensor([[2.4999],
            [4.2001],
            [3.0188],
            [2.9028]]), target_rating: tensor([2, 5, 4, 3])
    model_output: tensor([[2.3972],
            [3.4228],
            [3.3406],
            [4.4041]]), target_rating: tensor([3, 4, 5, 3])
    model_output: tensor([[3.3242],
            [3.0577],
            [3.9169],
            [3.6046]]), target_rating: tensor([5, 2, 3, 4])
    model_output: tensor([[3.9097],
            [3.2403],
            [3.2065],
            [2.8247]]), target_rating: tensor([5, 3, 3, 4])
    model_output: tensor([[2.9934],
            [4.4289],
            [3.4841],
            [3.6824]]), target_rating: tensor([3, 5, 2, 2])
    model_output: tensor([[3.5418],
            [4.0274],
            [3.1693],
            [3.1488]]), target_rating: tensor([3, 4, 3, 0])
    model_output: tensor([[3.7646],
            [2.8640],
            [3.9383],
            [3.3493]]), target_rating: tensor([5, 2, 4, 4])
    model_output: tensor([[3.2789],
            [3.4520],
            [3.3365],
            [3.5393]]), target_rating: tensor([1, 4, 4, 4])
    model_output: tensor([[4.3662],
            [2.7894],
            [3.2186],
            [2.9967]]), target_rating: tensor([4, 4, 4, 4])
    model_output: tensor([[2.9317],
            [2.7201],
            [3.5395],
            [3.8223]]), target_rating: tensor([3, 3, 4, 5])
    model_output: tensor([[2.5924],
            [2.7373],
            [3.9008],
            [3.3598]]), target_rating: tensor([3, 1, 4, 3])
    model_output: tensor([[2.6072],
            [3.9070],
            [3.3830],
            [3.4990]]), target_rating: tensor([1, 3, 4, 4])
    model_output: tensor([[3.3436],
            [3.7059],
            [2.8358],
            [3.5159]]), target_rating: tensor([3, 4, 1, 4])
    model_output: tensor([[3.6574],
            [2.2620],
            [2.9194],
            [4.5796]]), target_rating: tensor([3, 2, 2, 4])
    model_output: tensor([[3.3905],
            [3.2750],
            [3.0761],
            [3.2053]]), target_rating: tensor([2, 3, 4, 4])
    model_output: tensor([[4.2883],
            [3.2796],
            [2.9583],
            [3.6964]]), target_rating: tensor([4, 3, 4, 5])
    model_output: tensor([[3.3493],
            [2.9395],
            [3.2806],
            [3.0085]]), target_rating: tensor([3, 4, 3, 1])
    model_output: tensor([[3.3817],
            [2.9544],
            [2.0072],
            [4.2092]]), target_rating: tensor([4, 3, 0, 5])
    model_output: tensor([[3.9827],
            [3.0471],
            [3.3842],
            [3.0789]]), target_rating: tensor([4, 3, 3, 4])
    model_output: tensor([[1.8699],
            [3.4111],
            [2.8779],
            [3.6078]]), target_rating: tensor([3, 3, 3, 4])
    model_output: tensor([[3.9900],
            [3.6842],
            [3.8996],
            [4.1809]]), target_rating: tensor([5, 4, 3, 5])
    model_output: tensor([[3.3209],
            [3.1768],
            [2.9356],
            [3.7499]]), target_rating: tensor([3, 3, 3, 5])
    model_output: tensor([[3.2541],
            [2.1646],
            [2.5216],
            [3.3596]]), target_rating: tensor([3, 2, 3, 4])
    model_output: tensor([[2.8146],
            [3.2870],
            [3.5341],
            [3.7355]]), target_rating: tensor([1, 2, 4, 5])
    model_output: tensor([[3.3698],
            [3.5717],
            [3.3829],
            [2.8581]]), target_rating: tensor([4, 4, 3, 3])
    model_output: tensor([[2.7278],
            [3.2644],
            [2.9494],
            [4.3919]]), target_rating: tensor([4, 4, 3, 4])
    model_output: tensor([[3.0044],
            [3.3470],
            [2.5319],
            [3.5553]]), target_rating: tensor([3, 3, 4, 4])
    model_output: tensor([[3.7624],
            [2.8500],
            [2.9631],
            [3.9673]]), target_rating: tensor([4, 3, 2, 5])
    model_output: tensor([[3.5017],
            [3.5390],
            [3.5086],
            [3.4381]]), target_rating: tensor([4, 3, 4, 3])
    model_output: tensor([[3.2695],
            [3.9307],
            [4.0447],
            [3.4499]]), target_rating: tensor([3, 5, 4, 3])
    model_output: tensor([[3.6515],
            [2.9381],
            [3.7736],
            [3.0324]]), target_rating: tensor([4, 1, 3, 4])
    model_output: tensor([[3.8938],
            [3.2793],
            [3.6462],
            [2.8944]]), target_rating: tensor([5, 2, 4, 3])
    model_output: tensor([[2.5126],
            [3.8466],
            [3.6606],
            [3.1551]]), target_rating: tensor([3, 4, 4, 3])
    model_output: tensor([[3.6636],
            [2.2597],
            [3.0548],
            [2.5914]]), target_rating: tensor([2, 1, 4, 3])
    model_output: tensor([[2.9925],
            [3.0006],
            [3.5808],
            [3.1373]]), target_rating: tensor([2, 5, 3, 2])
    model_output: tensor([[3.5720],
            [3.4464],
            [3.3716],
            [2.1528]]), target_rating: tensor([4, 4, 4, 2])
    model_output: tensor([[3.6325],
            [2.3826],
            [3.5757],
            [3.9636]]), target_rating: tensor([3, 3, 5, 3])
    model_output: tensor([[2.5447],
            [4.0327],
            [3.0727],
            [2.4392]]), target_rating: tensor([1, 5, 3, 2])
    model_output: tensor([[3.7553],
            [3.2337],
            [3.6174],
            [3.6278]]), target_rating: tensor([4, 4, 4, 5])
    model_output: tensor([[3.4377],
            [3.1654],
            [3.4804],
            [4.0927]]), target_rating: tensor([3, 4, 4, 5])
    model_output: tensor([[3.2624],
            [3.0609],
            [2.9710],
            [3.4503]]), target_rating: tensor([1, 3, 3, 3])
    model_output: tensor([[2.7082],
            [3.4447],
            [3.1901],
            [2.6827]]), target_rating: tensor([2, 4, 3, 4])
    model_output: tensor([[3.7047],
            [3.7111],
            [3.5146],
            [3.3605]]), target_rating: tensor([4, 5, 4, 2])
    model_output: tensor([[3.3966],
            [2.8121],
            [3.3499],
            [2.8909]]), target_rating: tensor([3, 3, 3, 3])
    model_output: tensor([[4.0186],
            [3.2411],
            [3.0523],
            [2.5461]]), target_rating: tensor([3, 4, 3, 4])
    model_output: tensor([[4.2184],
            [3.4889],
            [3.4939],
            [3.6938]]), target_rating: tensor([5, 4, 4, 5])
    model_output: tensor([[3.9084],
            [3.1341],
            [2.6104],
            [3.7514]]), target_rating: tensor([5, 4, 3, 4])
    model_output: tensor([[3.3860],
            [3.4779],
            [3.3818],
            [3.1700]]), target_rating: tensor([3, 2, 4, 3])
    model_output: tensor([[3.9441],
            [2.1991],
            [3.4753],
            [4.0967]]), target_rating: tensor([5, 1, 4, 3])
    model_output: tensor([[3.2532],
            [2.8660],
            [3.8085],
            [3.9141]]), target_rating: tensor([4, 2, 4, 4])
    model_output: tensor([[2.6942],
            [3.0538],
            [3.7779],
            [2.8708]]), target_rating: tensor([3, 3, 4, 4])
    model_output: tensor([[3.6366],
            [3.0363],
            [3.9965],
            [3.2739]]), target_rating: tensor([4, 0, 5, 3])
    model_output: tensor([[3.7167],
            [3.1037],
            [3.5371],
            [3.8940]]), target_rating: tensor([4, 4, 4, 4])
    model_output: tensor([[4.1033],
            [2.9864],
            [2.8020],
            [3.9491]]), target_rating: tensor([4, 2, 3, 5])
    model_output: tensor([[4.4223],
            [3.6116],
            [2.4946],
            [3.1302]]), target_rating: tensor([4, 4, 3, 4])
    model_output: tensor([[3.2369],
            [3.7372],
            [3.2735],
            [3.7526]]), target_rating: tensor([5, 4, 3, 4])
    model_output: tensor([[3.8720],
            [3.3194],
            [4.8031],
            [3.7578]]), target_rating: tensor([3, 4, 5, 3])
    model_output: tensor([[3.0998],
            [3.9652],
            [2.3295],
            [3.8239]]), target_rating: tensor([3, 4, 3, 5])
    rms: 0.4671849638701718
    

## Evaluation with Recall@K


```python
from collections import defaultdict

# a dict that stores a list of predicted rating and actual rating pair for each user
user_est_true = defaultdict(list)

# iterate through the validation data to build the user -> [(y1, y1_hat), (y2, y2_hat), ...]
with torch.no_grad():
  for i, batched_data in enumerate(validation_loader):
    users = batched_data['users']
    movies = batched_data['movies']
    ratings = batched_data['ratings']

    model_output = model(batched_data['users'], batched_data['movies'])

    for i in range(len(users)):
      user_id = users[i].item()
      movie_id = movies[i].item()
      pred_rating = model_output[i][0].item()
      true_rating = ratings[i].item()

      print(f'{user_id}, {movie_id}, {pred_rating}, {true_rating}')
      user_est_true[user_id].append((pred_rating, true_rating))
```

    [1;30;43m스트리밍 출력 내용이 길어서 마지막 5000줄이 삭제되었습니다.[0m
    218, 6572, 2.8117287158966064, 4
    231, 3189, 3.6108787059783936, 4
    483, 5922, 3.5660274028778076, 4
    216, 938, 3.2754721641540527, 2
    589, 509, 3.3057427406311035, 3
    513, 1605, 3.0972518920898438, 2
    306, 4875, 2.2784624099731445, 1
    131, 1283, 3.2318954467773438, 3
    579, 4345, 2.9619386196136475, 5
    293, 306, 2.9402098655700684, 2
    17, 8237, 3.1363441944122314, 3
    152, 1430, 2.4374616146087646, 0
    17, 7104, 3.0043256282806396, 4
    72, 8988, 3.3817813396453857, 4
    437, 656, 2.9531631469726562, 3
    63, 62, 3.371731758117676, 4
    358, 5500, 3.401132822036743, 3
    279, 913, 4.278432369232178, 5
    554, 1807, 3.61761474609375, 3
    304, 7581, 3.6093316078186035, 4
    544, 6885, 3.0797879695892334, 2
    384, 84, 3.210280656814575, 3
    248, 2257, 3.86155104637146, 4
    605, 5156, 3.8351073265075684, 4
    214, 510, 4.098407745361328, 3
    427, 4239, 2.696013927459717, 4
    181, 888, 3.447542190551758, 4
    203, 6154, 3.379505157470703, 3
    468, 946, 3.8073222637176514, 5
    139, 3208, 3.2154910564422607, 4
    417, 4791, 3.922544479370117, 5
    566, 6737, 1.7291812896728516, 1
    379, 7394, 3.4840643405914307, 2
    67, 6399, 2.3965752124786377, 3
    596, 1216, 3.713890552520752, 4
    389, 6911, 3.261918067932129, 4
    356, 6579, 3.6803321838378906, 5
    304, 8217, 3.575894832611084, 5
    243, 118, 3.097738265991211, 3
    413, 5989, 3.0280957221984863, 3
    447, 2692, 2.5353875160217285, 4
    508, 8271, 2.5908496379852295, 3
    333, 2911, 3.3346948623657227, 1
    49, 5608, 2.6018521785736084, 4
    306, 5369, 2.029273509979248, 4
    437, 5729, 2.888432264328003, 3
    134, 152, 3.2898306846618652, 4
    306, 126, 1.9956152439117432, 2
    122, 2144, 4.26341438293457, 3
    138, 5259, 2.1196818351745605, 2
    201, 520, 4.2339982986450195, 4
    598, 1591, 2.1712429523468018, 1
    273, 1551, 2.496629238128662, 3
    198, 5212, 3.214122772216797, 3
    96, 2982, 3.576423168182373, 3
    599, 2191, 3.029188871383667, 3
    476, 4223, 2.847526788711548, 3
    437, 5227, 3.3504066467285156, 3
    231, 551, 3.1152195930480957, 3
    469, 533, 3.481677532196045, 3
    67, 4900, 3.4447622299194336, 4
    75, 3609, 3.1392955780029297, 3
    216, 1067, 2.849473714828491, 3
    512, 201, 3.212935447692871, 4
    473, 914, 3.459085464477539, 4
    413, 3849, 3.4257702827453613, 4
    413, 1660, 3.477782964706421, 3
    0, 913, 4.809704780578613, 5
    124, 7629, 3.6839756965637207, 4
    18, 2638, 2.8331170082092285, 2
    12, 1290, 3.3035433292388916, 4
    405, 980, 3.899655342102051, 1
    297, 2620, 2.0642387866973877, 0
    321, 1729, 3.3288702964782715, 4
    155, 986, 3.758057117462158, 4
    468, 97, 4.009463310241699, 5
    413, 5377, 3.3337574005126953, 4
    384, 228, 3.962888240814209, 3
    317, 6244, 3.0405640602111816, 3
    392, 5834, 3.562786102294922, 0
    159, 1291, 2.536069869995117, 1
    444, 5761, 3.408339500427246, 2
    17, 6592, 3.1048879623413086, 4
    530, 224, 4.066861152648926, 5
    269, 47, 2.9358432292938232, 4
    508, 7109, 2.8962483406066895, 3
    381, 6784, 3.2070796489715576, 1
    437, 2713, 2.7554707527160645, 2
    559, 6032, 2.9745657444000244, 3
    343, 7752, 3.5761358737945557, 4
    413, 2021, 3.1108779907226562, 3
    151, 7378, 4.060086250305176, 3
    589, 970, 3.3977701663970947, 4
    379, 3279, 3.71026611328125, 4
    533, 1157, 3.6863222122192383, 4
    447, 5319, 2.7135539054870605, 3
    291, 3897, 3.1269583702087402, 3
    220, 906, 4.32285737991333, 4
    134, 620, 3.4403882026672363, 4
    175, 197, 3.320091724395752, 4
    571, 1083, 4.125436782836914, 3
    94, 1444, 3.931318759918213, 4
    275, 609, 3.9613864421844482, 4
    101, 172, 3.1158361434936523, 2
    447, 1511, 2.8179843425750732, 4
    189, 5370, 3.6014938354492188, 3
    447, 4640, 2.8992128372192383, 3
    447, 565, 2.5677554607391357, 2
    183, 9717, 3.1346921920776367, 4
    233, 2802, 3.5844674110412598, 1
    598, 887, 2.6016032695770264, 3
    248, 325, 3.6013007164001465, 4
    88, 9484, 2.8632853031158447, 4
    468, 337, 3.4813766479492188, 4
    293, 2552, 2.8578450679779053, 5
    102, 2729, 3.7476015090942383, 4
    524, 1266, 3.485327959060669, 4
    131, 513, 2.951751947402954, 2
    473, 711, 3.4813971519470215, 4
    444, 510, 4.162242889404297, 4
    479, 827, 3.5828709602355957, 4
    128, 224, 4.1828765869140625, 4
    390, 2014, 3.799464702606201, 5
    476, 1708, 3.3577322959899902, 4
    349, 594, 3.2770192623138428, 3
    107, 3006, 4.2196173667907715, 5
    198, 2391, 3.1500067710876465, 4
    447, 3689, 2.783554792404175, 1
    598, 941, 2.155235767364502, 3
    3, 20, 3.488929271697998, 3
    425, 506, 3.568136215209961, 4
    431, 7338, 3.4814181327819824, 4
    215, 2208, 3.98423171043396, 3
    468, 936, 3.777782917022705, 4
    351, 6314, 4.023944854736328, 3
    67, 6525, 2.3826074600219727, 3
    317, 527, 3.4334001541137695, 3
    262, 3002, 3.5344276428222656, 3
    216, 1161, 2.707617998123169, 3
    424, 1370, 3.085937976837158, 3
    343, 6726, 3.564124345779419, 4
    67, 622, 2.2597458362579346, 1
    413, 801, 3.337148904800415, 3
    392, 1242, 3.9126405715942383, 4
    516, 3189, 2.7865004539489746, 4
    110, 2962, 2.707014560699463, 3
    152, 4939, 1.8986859321594238, 0
    541, 116, 3.3126108646392822, 4
    378, 217, 3.318185567855835, 3
    390, 2501, 3.8842077255249023, 4
    576, 742, 3.3715710639953613, 4
    473, 4606, 2.719533920288086, 3
    589, 5025, 3.460315704345703, 4
    598, 9510, 2.390331983566284, 2
    103, 224, 4.129449844360352, 3
    112, 2136, 3.3707995414733887, 1
    334, 1938, 3.704932928085327, 5
    413, 2027, 3.434624671936035, 4
    368, 908, 3.6375763416290283, 3
    473, 244, 3.1970136165618896, 3
    209, 3569, 3.725243091583252, 4
    297, 2632, 2.4777820110321045, 3
    16, 7396, 3.7437045574188232, 4
    5, 153, 3.2248668670654297, 3
    565, 303, 3.498434066772461, 4
    380, 4600, 3.4200472831726074, 5
    598, 3953, 2.0784201622009277, 3
    609, 2556, 2.929947853088379, 4
    218, 3868, 3.071021795272827, 3
    20, 8183, 2.748055934906006, 3
    231, 3456, 2.9259657859802246, 4
    447, 9115, 3.3023674488067627, 2
    220, 4739, 3.6732468605041504, 4
    413, 1958, 3.14338755607605, 2
    352, 145, 2.3526570796966553, 3
    473, 1705, 2.9831395149230957, 3
    67, 4474, 2.8191683292388916, 3
    521, 7306, 3.6847476959228516, 4
    325, 4607, 3.9319002628326416, 5
    468, 1796, 3.7918429374694824, 4
    602, 1825, 3.238041639328003, 3
    181, 1622, 3.3094050884246826, 4
    478, 1486, 3.1132383346557617, 3
    88, 5858, 3.3727221488952637, 5
    413, 6612, 3.091660499572754, 3
    468, 1217, 3.9301671981811523, 5
    233, 1182, 3.7088377475738525, 4
    350, 7652, 3.448042869567871, 2
    482, 627, 3.720676898956299, 4
    216, 2290, 3.012251615524292, 4
    78, 910, 4.532846450805664, 5
    73, 2979, 4.279061317443848, 5
    22, 1335, 3.230372905731201, 3
    572, 2144, 4.005255699157715, 2
    563, 8294, 3.282243013381958, 3
    605, 4246, 3.5318095684051514, 3
    559, 4419, 2.766103744506836, 3
    598, 761, 2.0128684043884277, 2
    158, 2437, 3.384582042694092, 1
    56, 2559, 3.477900505065918, 4
    407, 7511, 3.686246871948242, 4
    222, 4791, 3.3776416778564453, 3
    562, 8372, 2.6242787837982178, 2
    90, 1027, 3.2866616249084473, 3
    312, 1450, 2.9025208950042725, 3
    29, 4791, 4.136124134063721, 5
    470, 7784, 3.6842031478881836, 4
    20, 8293, 2.810737371444702, 3
    231, 6164, 2.6084771156311035, 3
    419, 1182, 3.5292129516601562, 3
    110, 8743, 3.4575889110565186, 5
    216, 713, 3.0655734539031982, 3
    181, 2602, 2.920958995819092, 2
    598, 5898, 2.2273778915405273, 3
    103, 78, 3.3993005752563477, 3
    488, 1144, 2.6774935722351074, 0
    413, 6209, 3.268282413482666, 3
    73, 906, 4.73189640045166, 5
    566, 1729, 2.499835729598999, 3
    190, 275, 3.689603805541992, 1
    327, 980, 3.31923770904541, 5
    400, 5988, 3.069814443588257, 4
    296, 794, 2.464310646057129, 2
    571, 398, 4.549517631530762, 5
    255, 7001, 3.8374431133270264, 4
    190, 52, 3.706789493560791, 4
    358, 6227, 3.312849998474121, 3
    83, 705, 3.6933319568634033, 4
    327, 785, 3.58052134513855, 5
    325, 6681, 3.874807119369507, 4
    82, 828, 3.1558151245117188, 1
    413, 304, 3.13260817527771, 3
    598, 6138, 2.676243305206299, 2
    479, 2957, 2.801941394805908, 4
    304, 8481, 3.3483080863952637, 4
    476, 1392, 3.048746109008789, 3
    426, 3657, 3.0531044006347656, 3
    367, 838, 2.492349147796631, 4
    123, 2448, 3.2386951446533203, 4
    598, 197, 2.112877368927002, 3
    32, 314, 4.438712120056152, 5
    19, 2659, 3.344088077545166, 3
    488, 1318, 2.5432918071746826, 3
    286, 3094, 1.9518760442733765, 0
    50, 196, 3.5022733211517334, 2
    248, 1370, 3.1877663135528564, 3
    198, 4931, 2.8779489994049072, 3
    431, 4786, 3.190971851348877, 4
    528, 622, 2.7056610584259033, 3
    317, 5380, 3.5585789680480957, 3
    508, 3826, 3.0431113243103027, 3
    308, 905, 3.728433847427368, 4
    221, 1264, 2.95532488822937, 3
    485, 506, 3.916896343231201, 3
    465, 6471, 3.4770126342773438, 4
    159, 2679, 2.002338409423828, 4
    248, 8248, 3.1092145442962646, 4
    233, 1070, 3.481092929840088, 4
    104, 7784, 4.098446846008301, 5
    321, 234, 3.2650961875915527, 3
    155, 2464, 3.6108264923095703, 2
    390, 2218, 3.7157130241394043, 3
    67, 6753, 2.6058151721954346, 3
    63, 2019, 3.454944133758545, 5
    531, 2144, 4.366244316101074, 4
    447, 1566, 2.7584896087646484, 4
    390, 678, 3.890878915786743, 2
    483, 3404, 3.136488437652588, 4
    62, 7449, 3.6105196475982666, 4
    306, 6136, 2.4610376358032227, 3
    0, 862, 4.930717468261719, 5
    183, 9643, 3.369040012359619, 4
    209, 8457, 3.734416961669922, 5
    447, 9253, 3.0013198852539062, 0
    260, 6314, 3.9760589599609375, 4
    488, 511, 2.922819137573242, 3
    473, 2623, 2.8641254901885986, 3
    533, 618, 3.480039119720459, 4
    201, 878, 4.064828872680664, 4
    219, 6195, 3.6337172985076904, 5
    124, 9212, 3.6481688022613525, 3
    451, 142, 4.182775497436523, 4
    67, 2040, 2.248399257659912, 3
    248, 3629, 3.1065056324005127, 3
    126, 2290, 2.9425692558288574, 4
    379, 3136, 4.491684913635254, 5
    102, 7261, 3.1134281158447266, 3
    238, 921, 4.247687816619873, 5
    550, 4931, 2.9444570541381836, 4
    31, 823, 3.811281204223633, 4
    473, 1063, 2.9542934894561768, 2
    473, 1662, 2.958207845687866, 3
    18, 0, 3.0202372074127197, 4
    533, 7615, 3.110931396484375, 3
    386, 4161, 2.8557686805725098, 3
    61, 7407, 3.644801616668701, 4
    380, 4419, 2.999833822250366, 2
    559, 7102, 3.149888038635254, 2
    216, 1861, 2.8822736740112305, 2
    291, 2377, 3.176896095275879, 3
    259, 4696, 3.417638063430786, 4
    606, 1559, 3.692145824432373, 5
    49, 8100, 2.7174949645996094, 2
    248, 6671, 3.4264235496520996, 3
    121, 2353, 4.427562713623047, 5
    423, 7355, 3.5769906044006348, 2
    231, 770, 3.270695209503174, 3
    489, 183, 2.807466506958008, 4
    279, 2021, 3.554307460784912, 4
    338, 8739, 3.6641898155212402, 4
    281, 1938, 3.9080777168273926, 5
    269, 24, 2.806666851043701, 4
    190, 267, 4.0423736572265625, 5
    447, 2765, 2.4605817794799805, 2
    598, 1283, 2.877984046936035, 4
    451, 1906, 4.356415748596191, 4
    233, 1821, 3.2154934406280518, 2
    442, 895, 4.280440330505371, 4
    379, 1692, 3.7246174812316895, 3
    168, 459, 3.560811996459961, 5
    572, 6501, 3.691894292831421, 4
    274, 2391, 4.114770889282227, 5
    531, 1429, 4.263758659362793, 4
    286, 2102, 2.2439284324645996, 3
    291, 815, 3.4218716621398926, 4
    219, 138, 3.924448251724243, 4
    527, 6042, 3.1072213649749756, 0
    220, 3297, 4.06317138671875, 4
    473, 1, 2.9791059494018555, 3
    231, 4422, 3.0318868160247803, 3
    122, 3563, 4.033083915710449, 4
    476, 1507, 3.0421814918518066, 3
    369, 3827, 2.874523162841797, 1
    120, 43, 3.520270824432373, 3
    413, 9351, 2.892592191696167, 4
    591, 138, 3.640383720397949, 3
    22, 4733, 3.2847447395324707, 3
    447, 302, 2.5393362045288086, 3
    88, 968, 3.505594253540039, 1
    199, 6877, 3.707624912261963, 4
    297, 1643, 1.9656164646148682, 3
    483, 3524, 3.221776247024536, 4
    15, 866, 3.219407558441162, 4
    287, 1170, 2.8904759883880615, 4
    313, 333, 2.538470983505249, 4
    447, 767, 2.7111830711364746, 3
    67, 6693, 3.1718709468841553, 5
    609, 6817, 3.363206624984741, 3
    386, 4636, 3.051076650619507, 3
    291, 302, 2.919745922088623, 2
    166, 4070, 3.7439606189727783, 4
    513, 4561, 3.4449734687805176, 4
    400, 6593, 2.942655563354492, 2
    297, 862, 2.8633174896240234, 2
    565, 277, 3.95106840133667, 5
    464, 714, 3.8144071102142334, 5
    434, 4607, 3.8762381076812744, 4
    219, 1486, 3.586418628692627, 4
    331, 0, 3.4432990550994873, 4
    585, 506, 4.378699779510498, 4
    273, 5861, 2.8993120193481445, 1
    392, 6405, 3.9446277618408203, 4
    605, 1473, 3.3573391437530518, 1
    513, 2144, 3.4962031841278076, 4
    248, 7776, 3.220435857772827, 3
    180, 141, 3.02250075340271, 3
    456, 2832, 3.962496280670166, 3
    55, 138, 3.899844169616699, 3
    71, 831, 3.3815746307373047, 3
    358, 3635, 3.5563931465148926, 5
    572, 4631, 3.576550245285034, 3
    413, 5998, 3.5618815422058105, 3
    352, 202, 3.082481622695923, 3
    318, 8528, 4.310175895690918, 4
    18, 1059, 2.4402382373809814, 3
    67, 3794, 2.4961295127868652, 4
    73, 705, 4.393424987792969, 5
    609, 8187, 2.73742413520813, 3
    425, 7739, 2.896256685256958, 3
    17, 7571, 2.882310628890991, 3
    481, 4421, 3.5380282402038574, 4
    336, 613, 4.057454586029053, 4
    128, 5885, 3.6271066665649414, 3
    108, 47, 3.142604351043701, 3
    333, 5682, 3.562988758087158, 3
    56, 2592, 3.5834391117095947, 5
    410, 340, 3.1369123458862305, 2
    270, 2759, 3.2371606826782227, 3
    473, 2383, 2.733628273010254, 3
    536, 6602, 3.7192375659942627, 5
    281, 2217, 3.660722255706787, 4
    607, 3662, 2.8557522296905518, 3
    366, 2913, 4.091195583343506, 4
    181, 1231, 3.214749813079834, 4
    390, 3952, 4.194945335388184, 2
    248, 7829, 3.524498462677002, 3
    101, 473, 3.011445999145508, 1
    413, 3415, 3.202251672744751, 4
    576, 991, 3.2682716846466064, 4
    0, 910, 4.763266563415527, 5
    447, 1080, 2.857545852661133, 4
    338, 7061, 3.8933963775634766, 2
    609, 6629, 3.2578680515289307, 3
    181, 3558, 3.3774642944335938, 3
    41, 77, 3.554561138153076, 4
    360, 5901, 3.354595184326172, 4
    289, 111, 4.318558692932129, 2
    167, 1294, 4.067120552062988, 4
    437, 2455, 2.6594390869140625, 3
    434, 2620, 3.4217090606689453, 4
    598, 8559, 2.7387301921844482, 1
    598, 2019, 2.531851291656494, 4
    473, 4828, 3.307335376739502, 4
    67, 5100, 2.9052748680114746, 4
    407, 2043, 3.200330972671509, 3
    197, 2744, 3.077772378921509, 3
    67, 4604, 2.75877046585083, 2
    572, 968, 3.879657030105591, 4
    451, 2078, 4.731656074523926, 5
    248, 6241, 3.7182555198669434, 4
    488, 3147, 3.0292177200317383, 1
    238, 505, 3.7648677825927734, 3
    195, 2670, 3.212907075881958, 5
    181, 1343, 2.981515645980835, 2
    304, 6631, 3.9525368213653564, 4
    413, 4791, 3.750438690185547, 4
    176, 2980, 3.1236512660980225, 0
    190, 203, 3.615431785583496, 1
    223, 2557, 3.5748493671417236, 2
    44, 1796, 3.951698064804077, 4
    461, 1666, 3.225020408630371, 4
    248, 6436, 3.5024476051330566, 4
    598, 1715, 2.38193678855896, 1
    50, 1801, 3.2161574363708496, 1
    78, 1032, 3.963566541671753, 3
    559, 6621, 3.349453926086426, 3
    271, 1266, 3.527564525604248, 4
    304, 828, 3.8694472312927246, 5
    605, 5873, 3.283473491668701, 4
    566, 9071, 1.8490041494369507, 4
    297, 2832, 2.4333925247192383, 2
    355, 2552, 3.8186254501342773, 5
    218, 5837, 2.819216012954712, 0
    576, 615, 3.0132479667663574, 3
    412, 3633, 4.281593322753906, 5
    297, 4793, 1.8764302730560303, 1
    379, 915, 4.215224266052246, 5
    217, 1223, 2.519754409790039, 2
    533, 2670, 3.7450950145721436, 4
    286, 1544, 2.187591314315796, 0
    508, 7784, 3.1283116340637207, 3
    2, 1700, 3.757814884185791, 4
    304, 5938, 3.8181610107421875, 2
    427, 468, 2.690068006515503, 1
    27, 2610, 2.982656240463257, 3
    425, 6968, 3.1846261024475098, 4
    18, 1872, 2.7526001930236816, 4
    599, 5736, 2.5040271282196045, 3
    293, 383, 2.6992735862731934, 3
    327, 3740, 3.161210536956787, 2
    219, 6849, 3.454503059387207, 2
    386, 5517, 2.9518134593963623, 4
    121, 862, 4.971631050109863, 5
    94, 4355, 3.9072694778442383, 3
    290, 7022, 4.061336517333984, 4
    508, 1176, 2.874965190887451, 4
    233, 131, 3.367290496826172, 5
    413, 263, 2.9307942390441895, 3
    456, 2903, 4.002574920654297, 4
    308, 933, 3.440906047821045, 4
    186, 62, 3.514237403869629, 4
    226, 659, 4.494078159332275, 4
    352, 510, 3.9101057052612305, 5
    473, 5769, 3.1697258949279785, 4
    598, 1492, 2.246239185333252, 3
    459, 1186, 3.7439799308776855, 5
    18, 2309, 2.5136711597442627, 2
    247, 8269, 3.3544416427612305, 3
    8, 3352, 2.911957263946533, 1
    482, 4422, 3.2077033519744873, 3
    404, 6659, 4.096656322479248, 3
    183, 9493, 3.3616514205932617, 5
    559, 7627, 2.92960262298584, 4
    297, 6868, 2.5065248012542725, 1
    248, 9297, 3.2311201095581055, 4
    413, 6244, 2.8870608806610107, 3
    513, 8681, 3.247683525085449, 4
    209, 6638, 3.537099599838257, 4
    78, 1997, 4.110723972320557, 2
    225, 956, 3.495701313018799, 4
    386, 1266, 3.338803291320801, 3
    602, 2576, 3.407474994659424, 2
    14, 3635, 3.671973705291748, 5
    169, 175, 3.072721242904663, 3
    306, 527, 2.3200137615203857, 2
    476, 506, 3.5918564796447754, 3
    473, 1567, 2.8025503158569336, 3
    604, 783, 2.8934223651885986, 3
    473, 2441, 2.7867846488952637, 3
    123, 224, 4.021080493927002, 4
    63, 2392, 3.3601162433624268, 3
    19, 4021, 3.5698299407958984, 4
    313, 145, 2.0835540294647217, 3
    585, 3633, 4.748897552490234, 5
    110, 8560, 3.3251395225524902, 2
    427, 8, 2.5432331562042236, 2
    76, 3827, 3.2733266353607178, 5
    159, 97, 3.2471070289611816, 4
    85, 4791, 4.065042495727539, 4
    413, 145, 2.4902760982513428, 3
    63, 3283, 3.3092715740203857, 3
    248, 7710, 3.5554986000061035, 4
    280, 618, 2.849978446960449, 2
    521, 7961, 3.3384618759155273, 3
    552, 2511, 4.131510257720947, 3
    479, 2775, 2.8348331451416016, 3
    306, 297, 2.4234330654144287, 1
    98, 138, 3.952016830444336, 5
    311, 474, 4.032705307006836, 5
    596, 3263, 3.728118419647217, 3
    90, 5135, 2.900695323944092, 1
    413, 3374, 2.8985068798065186, 3
    488, 1330, 2.656348466873169, 3
    10, 510, 4.214905738830566, 5
    239, 1, 3.7175655364990234, 5
    606, 138, 3.9661574363708496, 4
    311, 2026, 3.2582015991210938, 3
    6, 6071, 2.4909322261810303, 4
    265, 1210, 3.8099255561828613, 5
    451, 1324, 4.273224353790283, 5
    22, 5124, 3.4794960021972656, 4
    482, 4939, 3.149725914001465, 2
    598, 524, 2.21510648727417, 1
    562, 7171, 2.4391958713531494, 2
    79, 5301, 3.7054145336151123, 4
    273, 7263, 2.7975118160247803, 3
    16, 2760, 3.674964666366577, 3
    47, 1754, 3.7174010276794434, 3
    479, 3094, 2.937000274658203, 3
    102, 1035, 3.5214333534240723, 4
    482, 7769, 3.4840829372406006, 3
    386, 4383, 2.9494147300720215, 3
    21, 2601, 2.4353322982788086, 0
    440, 6693, 4.514360427856445, 5
    413, 1913, 2.9470460414886475, 3
    598, 8402, 2.328238010406494, 3
    219, 6897, 3.55792236328125, 3
    413, 3972, 2.9376819133758545, 2
    265, 167, 3.3416619300842285, 3
    413, 7047, 3.1569836139678955, 2
    596, 148, 3.89864444732666, 4
    595, 914, 3.4219679832458496, 4
    501, 1008, 2.6391561031341553, 4
    50, 4956, 3.7489542961120605, 5
    482, 7195, 3.3906893730163574, 4
    306, 3050, 1.9895315170288086, 2
    28, 8038, 3.5426974296569824, 4
    607, 4416, 2.4569852352142334, 2
    496, 7971, 3.2682864665985107, 5
    20, 2353, 3.0646345615386963, 3
    413, 4432, 3.090453863143921, 2
    252, 2326, 3.4663596153259277, 5
    566, 3827, 1.730971097946167, 0
    413, 3362, 2.8714230060577393, 3
    30, 1040, 3.502732992172241, 4
    452, 2692, 3.341517448425293, 4
    423, 907, 3.726778030395508, 3
    474, 2803, 4.008031368255615, 4
    176, 1497, 3.028481960296631, 2
    181, 2030, 3.170478105545044, 2
    551, 3192, 2.6643221378326416, 0
    304, 8406, 3.248849630355835, 2
    526, 1700, 4.217959880828857, 5
    463, 229, 3.3797903060913086, 4
    503, 3644, 3.7592315673828125, 4
    482, 6237, 3.285745620727539, 3
    581, 3136, 4.025602340698242, 4
    329, 395, 3.3796520233154297, 3
    554, 1540, 3.7118701934814453, 3
    554, 514, 3.629868984222412, 4
    523, 930, 3.4757790565490723, 4
    61, 9415, 3.794175148010254, 4
    605, 1476, 3.3960187435150146, 0
    421, 2335, 3.294388771057129, 4
    330, 6464, 3.5390381813049316, 4
    596, 2254, 3.603459119796753, 4
    345, 1444, 3.6197075843811035, 4
    382, 1796, 3.6746106147766113, 5
    572, 2992, 3.952159881591797, 4
    598, 7131, 2.210526466369629, 2
    379, 5253, 3.766822338104248, 3
    176, 7154, 3.3160929679870605, 3
    56, 832, 3.161046028137207, 3
    259, 915, 3.8719396591186523, 4
    131, 509, 2.8128557205200195, 2
    371, 1000, 3.395514965057373, 3
    253, 6329, 3.6587367057800293, 4
    304, 959, 3.914142608642578, 4
    365, 7355, 3.670933961868286, 4
    17, 8551, 3.451277017593384, 4
    231, 2224, 3.8855879306793213, 4
    379, 4013, 3.834014654159546, 3
    349, 551, 3.2849137783050537, 3
    598, 3065, 2.4244022369384766, 2
    67, 2653, 2.6154425144195557, 3
    201, 134, 3.881331443786621, 4
    273, 3563, 3.053770065307617, 3
    346, 507, 3.711430549621582, 3
    168, 2097, 4.070782661437988, 5
    90, 2153, 3.19469952583313, 3
    281, 4600, 3.427342414855957, 4
    524, 8001, 3.241086006164551, 4
    88, 6388, 3.6621694564819336, 5
    39, 311, 3.790374994277954, 3
    60, 254, 3.8440370559692383, 4
    437, 1986, 3.165602207183838, 4
    521, 8133, 3.257535696029663, 3
    257, 8528, 3.966817855834961, 5
    304, 4519, 3.728841781616211, 4
    192, 5441, 3.134922504425049, 5
    203, 7073, 3.6697983741760254, 4
    455, 577, 3.3911211490631104, 3
    26, 1978, 3.3590731620788574, 5
    201, 1806, 3.6675779819488525, 4
    413, 6227, 3.4253928661346436, 4
    225, 3152, 3.344481945037842, 2
    67, 1915, 2.601074695587158, 4
    238, 6293, 3.6358838081359863, 3
    473, 2190, 3.4190025329589844, 3
    413, 1988, 2.6965627670288086, 2
    116, 506, 3.6452596187591553, 4
    442, 1938, 4.148639678955078, 3
    379, 1190, 3.6126959323883057, 3
    605, 1110, 3.6476643085479736, 4
    479, 4476, 3.2985053062438965, 4
    273, 277, 3.5928289890289307, 4
    479, 3543, 3.2134246826171875, 3
    598, 3963, 2.1683928966522217, 1
    181, 2275, 3.5109400749206543, 2
    437, 1259, 3.2007293701171875, 3
    99, 508, 3.845993995666504, 4
    198, 2117, 3.2713236808776855, 1
    110, 5212, 3.3576500415802, 4
    364, 5298, 2.868617534637451, 2
    139, 1083, 3.5723133087158203, 4
    229, 2034, 2.3555290699005127, 3
    478, 527, 3.264871120452881, 4
    308, 4345, 3.1782569885253906, 2
    474, 7571, 3.837900161743164, 5
    447, 2903, 3.180001974105835, 4
    533, 8817, 3.112731456756592, 4
    239, 507, 4.301444053649902, 4
    609, 6959, 3.076286792755127, 3
    304, 938, 3.9041779041290283, 3
    248, 6269, 3.421799659729004, 1
    609, 4925, 3.131620168685913, 4
    326, 6693, 4.205561637878418, 3
    598, 3136, 3.1334705352783203, 4
    386, 4978, 2.989619731903076, 4
    599, 3978, 2.8912220001220703, 4
    22, 4926, 3.6664061546325684, 3
    540, 97, 3.8731508255004883, 5
    598, 7640, 2.153988838195801, 2
    413, 6039, 3.258606433868408, 4
    248, 9350, 3.1757473945617676, 3
    361, 7750, 4.154757499694824, 4
    44, 1294, 3.7052836418151855, 4
    473, 1530, 2.8727455139160156, 4
    88, 867, 3.2711398601531982, 2
    176, 1712, 3.228943347930908, 3
    386, 5068, 3.270235300064087, 2
    22, 31, 3.7272820472717285, 3
    18, 598, 2.198613166809082, 2
    482, 3024, 3.1630699634552, 5
    152, 615, 2.00722599029541, 0
    306, 862, 2.9961323738098145, 3
    339, 508, 4.194774627685547, 5
    598, 5780, 2.0946459770202637, 2
    473, 3892, 3.176344156265259, 3
    502, 2335, 3.2945194244384766, 3
    5, 315, 3.4985592365264893, 3
    273, 7228, 2.5524394512176514, 1
    605, 3404, 2.9586329460144043, 2
    231, 5317, 3.0172715187072754, 3
    304, 2988, 3.4555139541625977, 4
    609, 183, 3.223529577255249, 5
    488, 133, 2.414844036102295, 1
    516, 4090, 1.9715955257415771, 0
    316, 7198, 3.764519453048706, 5
    329, 1882, 3.6408071517944336, 5
    231, 1480, 2.80776309967041, 3
    390, 1260, 3.354614019393921, 1
    201, 2708, 3.983139753341675, 3
    131, 4135, 2.7025868892669678, 1
    490, 6886, 3.8316807746887207, 4
    65, 973, 3.696375608444214, 4
    164, 2441, 3.0151071548461914, 4
    342, 6298, 4.006393909454346, 4
    176, 2696, 2.982375144958496, 3
    18, 283, 2.675537347793579, 2
    67, 2191, 2.880185604095459, 3
    513, 9633, 2.791977643966675, 2
    201, 474, 4.302478313446045, 5
    379, 5834, 3.70267915725708, 4
    598, 322, 3.04409122467041, 3
    442, 8277, 3.506436824798584, 5
    231, 6497, 3.5393080711364746, 4
    63, 4112, 3.590240001678467, 3
    245, 2911, 4.139246940612793, 2
    413, 5155, 3.411134719848633, 3
    554, 2126, 3.0854034423828125, 1
    303, 1480, 3.27064847946167, 4
    14, 1485, 3.1849451065063477, 5
    381, 224, 3.9404237270355225, 5
    114, 2016, 3.3027279376983643, 4
    63, 291, 3.5402863025665283, 4
    351, 7207, 3.2080235481262207, 3
    468, 959, 3.891592264175415, 4
    598, 1865, 2.2360072135925293, 2
    413, 509, 3.3438334465026855, 4
    152, 8961, 2.2951183319091797, 3
    52, 853, 4.317069053649902, 5
    413, 2348, 3.2889020442962646, 3
    364, 6217, 1.8823908567428589, 2
    488, 3712, 3.211296796798706, 4
    274, 3762, 4.034208297729492, 2
    413, 504, 3.2140612602233887, 3
    233, 585, 3.693021774291992, 2
    390, 705, 4.023350238800049, 5
    465, 6341, 3.643442392349243, 4
    278, 914, 3.60552978515625, 5
    61, 8663, 3.8714170455932617, 4
    570, 1968, 2.6519103050231934, 1
    566, 7348, 1.972179651260376, 1
    447, 9251, 2.5142221450805664, 3
    386, 3531, 2.899085521697998, 2
    159, 4320, 2.735291004180908, 1
    541, 1321, 3.4380948543548584, 3
    486, 4421, 3.2751405239105225, 2
    418, 6497, 3.928389072418213, 3
    140, 43, 3.4322926998138428, 3
    533, 915, 3.95024037361145, 4
    6, 3192, 2.704014301300049, 0
    158, 2353, 3.1236579418182373, 4
    386, 2592, 3.3145318031311035, 3
    56, 560, 3.3487865924835205, 4
    386, 1408, 2.9329583644866943, 4
    579, 2014, 3.104915142059326, 3
    140, 1157, 3.2616419792175293, 3
    413, 3217, 3.7823970317840576, 4
    408, 520, 4.144129276275635, 5
    65, 1475, 3.6441211700439453, 3
    413, 4514, 3.372305393218994, 3
    75, 705, 3.1088342666625977, 1
    566, 9586, 1.9490327835083008, 5
    67, 1282, 2.571979522705078, 3
    17, 8747, 3.2775280475616455, 3
    231, 2041, 2.4979135990142822, 2
    524, 906, 3.860360860824585, 3
    553, 707, 3.6341986656188965, 4
    90, 4010, 3.1367204189300537, 3
    287, 4799, 2.8184640407562256, 3
    181, 2080, 2.8923914432525635, 3
    476, 900, 3.916823148727417, 4
    262, 190, 4.01031494140625, 1
    124, 8531, 3.4377293586730957, 4
    139, 4643, 3.345036029815674, 3
    483, 2586, 3.873826026916504, 3
    46, 6211, 2.491271495819092, 3
    26, 2432, 3.4978578090667725, 3
    509, 1487, 2.7303946018218994, 3
    379, 1738, 3.8315467834472656, 2
    192, 510, 4.06579065322876, 3
    273, 6510, 3.1081955432891846, 4
    89, 520, 4.180683135986328, 5
    273, 78, 2.6915791034698486, 3
    602, 1349, 3.2658002376556396, 4
    533, 8218, 3.8173470497131348, 4
    390, 3719, 3.525545120239258, 1
    468, 898, 4.089640140533447, 5
    609, 4881, 3.1122934818267822, 3
    67, 3986, 2.6944448947906494, 2
    139, 3697, 3.3923916816711426, 4
    379, 3661, 3.461113929748535, 2
    306, 1615, 2.875958204269409, 3
    494, 659, 4.162876129150391, 4
    598, 5919, 2.5906331539154053, 3
    168, 980, 4.091578483581543, 4
    473, 468, 3.322089195251465, 4
    353, 3849, 3.571620464324951, 4
    161, 295, 4.006120681762695, 3
    246, 3868, 3.5134994983673096, 4
    231, 5988, 3.225639581680298, 3
    26, 2843, 3.6843814849853516, 1
    476, 578, 2.926598072052002, 4
    607, 3346, 2.632375717163086, 2
    587, 275, 3.202568769454956, 2
    413, 4509, 2.880154609680176, 4
    379, 3719, 3.5736641883850098, 4
    602, 155, 3.364670991897583, 4
    311, 2610, 3.8511624336242676, 4
    317, 8375, 3.5645699501037598, 3
    413, 3491, 3.483208656311035, 2
    589, 4600, 3.289661407470703, 3
    559, 7209, 3.1064817905426025, 4
    433, 6045, 3.3788223266601562, 3
    183, 8585, 3.3501574993133545, 4
    189, 3563, 3.650913715362549, 4
    267, 2236, 2.977174997329712, 4
    413, 2995, 2.9759702682495117, 3
    479, 2592, 3.349738359451294, 3
    297, 4926, 2.5887813568115234, 4
    379, 2867, 3.4506516456604004, 2
    60, 4402, 3.6288585662841797, 4
    104, 4893, 3.663213014602661, 4
    43, 675, 3.0744779109954834, 3
    473, 694, 3.852741241455078, 5
    165, 6014, 3.4573721885681152, 4
    560, 5249, 3.173147678375244, 3
    147, 8551, 3.4741768836975098, 4
    220, 952, 3.336817741394043, 4
    40, 6592, 2.7571980953216553, 2
    410, 142, 2.9029479026794434, 3
    304, 1961, 3.326948642730713, 4
    200, 983, 4.255773544311523, 4
    304, 8108, 3.5011239051818848, 3
    427, 220, 2.510467767715454, 2
    351, 1294, 3.5424909591674805, 4
    255, 8256, 3.7346572875976562, 4
    411, 1231, 3.877750873565674, 5
    418, 6671, 3.497150421142578, 3
    317, 6960, 3.630727767944336, 3
    482, 6953, 3.4521918296813965, 3
    131, 41, 2.743720054626465, 2
    605, 5369, 3.1695609092712402, 2
    304, 7958, 3.692349672317505, 4
    551, 314, 3.605551242828369, 4
    599, 2102, 2.961793899536133, 4
    604, 4805, 2.8827619552612305, 2
    139, 733, 3.943720817565918, 3
    296, 656, 2.3597710132598877, 1
    102, 7163, 3.7883353233337402, 4
    524, 6041, 3.1731669902801514, 4
    331, 947, 3.190136194229126, 2
    40, 8611, 2.875612258911133, 4
    67, 2455, 2.0921790599823, 4
    379, 9069, 3.653961420059204, 3
    67, 910, 3.1847288608551025, 5
    519, 326, 3.9070072174072266, 3
    139, 4289, 3.430893659591675, 4
    479, 6023, 3.182751417160034, 1
    447, 6058, 3.1063709259033203, 3
    541, 5363, 3.702038288116455, 3
    583, 253, 3.650409698486328, 5
    553, 920, 3.9105148315429688, 1
    427, 908, 3.2270021438598633, 3
    317, 3404, 2.931731700897217, 3
    551, 897, 3.3958020210266113, 5
    371, 1132, 3.5093798637390137, 1
    397, 6755, 4.338877201080322, 5
    186, 849, 3.540544033050537, 4
    427, 140, 2.09566593170166, 2
    352, 138, 3.3286750316619873, 5
    428, 221, 3.8209519386291504, 4
    350, 4279, 2.954857349395752, 3
    220, 4725, 3.3903846740722656, 4
    461, 8169, 2.6827151775360107, 4
    187, 785, 4.951075077056885, 5
    508, 5905, 2.8570337295532227, 3
    516, 3790, 1.9653536081314087, 1
    554, 1468, 3.496330738067627, 4
    351, 1348, 3.3209681510925293, 3
    453, 6997, 3.493020534515381, 4
    437, 1075, 3.1988425254821777, 4
    63, 2824, 3.0295026302337646, 2
    532, 6645, 3.1724631786346436, 5
    413, 6298, 3.8199191093444824, 4
    27, 913, 3.2301783561706543, 4
    171, 910, 3.5257298946380615, 4
    601, 164, 3.225156784057617, 2
    8, 1486, 2.83583402633667, 4
    134, 1093, 3.016636848449707, 4
    526, 2102, 3.8288044929504395, 5
    3, 2543, 3.3594682216644287, 4
    350, 3003, 3.1291909217834473, 4
    51, 5780, 3.8544559478759766, 3
    488, 6396, 2.077073097229004, 1
    100, 2531, 2.815713405609131, 1
    211, 6693, 3.806697130203247, 3
    231, 4326, 3.25203800201416, 3
    287, 2088, 3.0019636154174805, 3
    293, 2689, 2.6434621810913086, 2
    461, 1309, 2.909245491027832, 2
    24, 6693, 4.392782211303711, 5
    67, 8407, 2.8646938800811768, 3
    473, 2615, 3.483133554458618, 4
    489, 4421, 2.992687702178955, 3
    291, 7609, 2.7267098426818848, 3
    297, 6663, 2.126089572906494, 0
    138, 2957, 1.75531005859375, 0
    465, 1938, 4.014341354370117, 4
    386, 1960, 2.9733481407165527, 3
    50, 3294, 3.3752264976501465, 4
    116, 30, 3.4089293479919434, 3
    234, 383, 3.43959903717041, 4
    569, 287, 3.240825653076172, 3
    139, 3730, 3.275169849395752, 4
    273, 3463, 2.8566086292266846, 1
    609, 6433, 3.3410837650299072, 4
    225, 5380, 3.281811237335205, 2
    598, 2569, 2.2485930919647217, 2
    447, 9079, 2.8974690437316895, 3
    390, 2903, 3.9913744926452637, 4
    609, 7672, 3.1925954818725586, 4
    430, 1805, 2.869935989379883, 3
    221, 8209, 2.97700834274292, 2
    307, 4131, 2.7943665981292725, 4
    602, 76, 3.2588326930999756, 5
    120, 84, 2.9112393856048584, 3
    88, 2353, 3.305427312850952, 3
    355, 908, 4.3851776123046875, 5
    413, 1001, 3.505117893218994, 5
    8, 4141, 3.5478644371032715, 4
    482, 2799, 3.0640876293182373, 4
    437, 5865, 3.3697774410247803, 4
    607, 257, 3.3464181423187256, 5
    50, 1764, 3.56892728805542, 2
    166, 779, 2.9422106742858887, 2
    174, 1901, 2.6599934101104736, 0
    33, 4755, 3.3835082054138184, 4
    99, 1680, 3.7622666358947754, 4
    199, 6510, 3.876814365386963, 5
    461, 1762, 2.830629348754883, 4
    318, 1938, 4.423460960388184, 5
    273, 4164, 2.51003360748291, 3
    605, 1002, 3.0903842449188232, 4
    554, 2673, 3.539891242980957, 1
    602, 432, 2.9445247650146484, 5
    142, 3981, 3.2716116905212402, 3
    596, 1037, 4.246241569519043, 5
    248, 1190, 3.3341875076293945, 2
    562, 696, 3.049217939376831, 4
    245, 7195, 3.9024343490600586, 4
    369, 4131, 3.6494271755218506, 4
    386, 2224, 3.7863457202911377, 4
    317, 5999, 3.8042678833007812, 3
    39, 97, 3.9566922187805176, 4
    287, 307, 3.35135817527771, 3
    176, 7808, 3.4445812702178955, 4
    62, 3898, 3.3173136711120605, 2
    468, 1938, 4.064791202545166, 4
    297, 4648, 2.032928943634033, 2
    380, 4354, 3.6875104904174805, 5
    77, 1474, 2.844439744949341, 4
    131, 792, 3.0967016220092773, 3
    473, 1470, 2.463460922241211, 1
    63, 2916, 3.5390114784240723, 3
    609, 8480, 3.3249690532684326, 4
    61, 4416, 3.3001105785369873, 3
    380, 1299, 3.5241196155548096, 4
    38, 46, 4.401641845703125, 5
    72, 4926, 3.856132984161377, 4
    524, 3026, 3.038925886154175, 3
    324, 3677, 3.6316444873809814, 4
    313, 412, 2.994131565093994, 3
    23, 7039, 3.174642562866211, 3
    306, 2261, 2.4792916774749756, 2
    598, 1732, 1.9605790376663208, 2
    181, 3089, 3.1026782989501953, 2
    437, 1622, 3.2863402366638184, 4
    452, 3910, 3.428562879562378, 4
    3, 1403, 3.697345733642578, 3
    102, 4522, 3.734591484069824, 5
    413, 3318, 3.393545389175415, 3
    112, 934, 3.858938455581665, 5
    185, 1370, 3.918577194213867, 4
    561, 2109, 4.281981468200684, 1
    438, 7338, 4.132555961608887, 4
    386, 5163, 3.022552251815796, 3
    83, 1002, 3.005167007446289, 3
    602, 1528, 3.270509719848633, 4
    602, 1180, 2.962146043777466, 5
    599, 1720, 2.804391384124756, 4
    500, 116, 3.3046021461486816, 5
    473, 967, 3.407482385635376, 3
    404, 4174, 3.2587809562683105, 4
    566, 9202, 1.7075462341308594, 1
    298, 914, 3.7259271144866943, 3
    479, 4440, 3.0152974128723145, 4
    19, 2979, 3.7979605197906494, 3
    473, 3483, 3.2142066955566406, 3
    540, 403, 3.553161144256592, 3
    110, 4617, 2.7404065132141113, 0
    605, 6103, 3.543606758117676, 4
    598, 3326, 2.88643479347229, 1
    185, 3739, 3.7624130249023438, 4
    231, 6020, 2.7034316062927246, 3
    257, 7285, 3.819460153579712, 4
    488, 7395, 2.5529158115386963, 2
    469, 229, 3.4690780639648438, 4
    183, 4068, 3.3921399116516113, 3
    566, 6320, 1.9549145698547363, 2
    513, 3404, 2.5367188453674316, 3
    19, 2285, 3.6479568481445312, 5
    168, 2940, 3.758755683898926, 5
    273, 7241, 3.0154857635498047, 4
    244, 2341, 3.2624497413635254, 1
    83, 1067, 3.2945122718811035, 3
    593, 3981, 3.341060161590576, 4
    293, 1296, 2.5274853706359863, 1
    379, 2220, 3.993072509765625, 5
    287, 751, 3.099377393722534, 3
    85, 2110, 3.4361844062805176, 4
    605, 717, 3.6508736610412598, 4
    508, 6132, 2.9029006958007812, 3
    181, 3909, 3.1956372261047363, 1
    605, 1835, 3.7317328453063965, 4
    468, 1648, 3.5150012969970703, 5
    90, 2452, 3.447624683380127, 4
    447, 455, 2.680136203765869, 3
    52, 1970, 3.8443236351013184, 5
    233, 1055, 3.56559419631958, 3
    282, 314, 3.8981990814208984, 5
    431, 2916, 3.2539918422698975, 4
    352, 126, 2.8178791999816895, 3
    413, 6951, 2.975762367248535, 3
    353, 2192, 3.5846469402313232, 3
    41, 837, 3.670154094696045, 4
    155, 1854, 3.581671714782715, 2
    306, 2729, 2.6242613792419434, 4
    598, 7910, 2.179443836212158, 4
    541, 6298, 3.7782294750213623, 3
    212, 2601, 3.8351128101348877, 4
    413, 8676, 3.2865498065948486, 4
    598, 3743, 2.2897379398345947, 3
    38, 509, 3.7988362312316895, 4
    73, 707, 4.012432098388672, 5
    322, 53, 2.610501289367676, 3
    274, 487, 4.343578338623047, 4
    542, 6771, 4.079878807067871, 5
    356, 1082, 3.6026086807250977, 3
    575, 2325, 2.9993371963500977, 4
    287, 2811, 2.956986665725708, 3
    67, 84, 2.4357645511627197, 3
    82, 980, 3.1780147552490234, 4
    609, 7909, 3.1463701725006104, 3
    63, 3312, 2.9958653450012207, 4
    609, 6765, 2.813199043273926, 4
    605, 969, 3.4551076889038086, 4
    313, 592, 2.830599069595337, 4
    287, 4968, 2.6698195934295654, 3
    38, 1294, 3.7441272735595703, 5
    110, 8733, 3.406769037246704, 4
    413, 6758, 3.2759833335876465, 3
    297, 2744, 1.757089614868164, 2
    287, 2250, 3.1389071941375732, 3
    325, 461, 4.1932783126831055, 4
    121, 8863, 3.821225166320801, 4
    390, 1218, 3.496939182281494, 2
    356, 1062, 3.4448137283325195, 4
    312, 3633, 3.8752007484436035, 5
    368, 3565, 3.0437874794006348, 4
    176, 6416, 3.2723898887634277, 2
    104, 5723, 3.7797913551330566, 4
    166, 1443, 3.4479169845581055, 4
    592, 2804, 2.919386625289917, 2
    566, 6314, 2.5224645137786865, 3
    598, 1670, 2.7087812423706055, 2
    383, 2078, 3.463073492050171, 3
    20, 257, 3.426751136779785, 3
    386, 6119, 2.5049750804901123, 2
    306, 6313, 2.576554536819458, 3
    40, 826, 2.720198631286621, 1
    535, 509, 3.313477039337158, 2
    568, 508, 3.8464691638946533, 4
    10, 337, 3.392253875732422, 4
    379, 8259, 3.862271785736084, 4
    413, 3765, 2.9507055282592773, 5
    56, 1945, 3.7226977348327637, 3
    579, 1290, 2.8469583988189697, 4
    494, 3147, 3.7488324642181396, 4
    559, 6329, 3.4082069396972656, 4
    181, 4596, 3.166402578353882, 1
    5, 257, 3.7736732959747314, 2
    192, 613, 3.47526216506958, 4
    181, 5084, 3.1654443740844727, 4
    500, 46, 3.8969404697418213, 3
    51, 3849, 4.300690650939941, 5
    421, 2048, 3.5543479919433594, 5
    237, 2619, 4.103801250457764, 4
    386, 2108, 3.0514447689056396, 2
    541, 3005, 3.0778615474700928, 3
    420, 4555, 3.3773913383483887, 4
    434, 6868, 3.863995313644409, 5
    176, 184, 3.212080240249634, 2
    598, 3014, 2.593691349029541, 2
    602, 936, 3.5746378898620605, 4
    306, 4220, 2.3218138217926025, 1
    473, 2546, 3.155783176422119, 3
    483, 1542, 3.7752811908721924, 5
    215, 714, 3.4477598667144775, 5
    368, 4495, 3.1695075035095215, 3
    102, 8045, 3.7331018447875977, 4
    508, 8008, 2.7084743976593018, 3
    324, 2248, 3.5722694396972656, 3
    476, 4936, 3.105842351913452, 4
    409, 697, 3.547553539276123, 4
    17, 8681, 3.3563592433929443, 3
    524, 1601, 3.0531692504882812, 4
    116, 609, 3.2704410552978516, 3
    595, 8358, 3.4064781665802, 3
    447, 6434, 3.107876777648926, 3
    429, 3868, 3.4836478233337402, 5
    56, 167, 3.2039220333099365, 3
    293, 435, 2.5717148780822754, 2
    598, 2760, 2.397226572036743, 3
    67, 901, 2.9121434688568115, 2
    390, 2505, 3.4779112339019775, 2
    176, 920, 3.4573445320129395, 3
    447, 2390, 2.840268611907959, 3
    560, 3490, 3.4419384002685547, 4
    218, 1525, 2.9217536449432373, 3
    297, 3407, 2.2910983562469482, 3
    413, 2251, 3.782005786895752, 3
    111, 1072, 3.4555437564849854, 4
    585, 6195, 4.0394158363342285, 5
    407, 1153, 3.7525885105133057, 4
    67, 6647, 2.764763116836548, 2
    290, 3569, 3.89377760887146, 5
    413, 559, 2.9141294956207275, 2
    599, 788, 2.779341220855713, 2
    563, 7374, 3.078911066055298, 4
    390, 900, 4.265019416809082, 5
    273, 5156, 3.2010908126831055, 3
    306, 5732, 2.2586097717285156, 2
    479, 980, 3.4252192974090576, 3
    309, 1075, 3.3779897689819336, 4
    327, 418, 3.290337085723877, 2
    201, 329, 3.7326762676239014, 4
    591, 238, 3.3283276557922363, 3
    524, 2995, 2.899714946746826, 2
    98, 274, 3.6081976890563965, 3
    364, 8293, 2.3517205715179443, 2
    414, 4794, 3.889237642288208, 4
    371, 383, 3.1385715007781982, 4
    483, 504, 3.5723209381103516, 4
    67, 2756, 2.7230184078216553, 2
    561, 983, 3.9058375358581543, 4
    197, 1543, 3.43182373046875, 5
    479, 1290, 2.928729295730591, 4
    273, 3765, 2.497093439102173, 3
    589, 2567, 3.092005968093872, 3
    367, 1700, 3.2439093589782715, 2
    572, 6204, 3.404407501220703, 5
    553, 1606, 3.7957470417022705, 1
    451, 1882, 4.850541114807129, 5
    413, 225, 3.6630492210388184, 4
    508, 8049, 3.076597213745117, 3
    413, 431, 3.3229706287384033, 4
    389, 5388, 3.2758898735046387, 5
    390, 512, 4.012441158294678, 3
    118, 9005, 3.5565686225891113, 1
    317, 1540, 3.502575635910034, 4
    605, 4796, 3.1179659366607666, 4
    607, 2453, 2.4300737380981445, 2
    598, 287, 2.4971377849578857, 4
    563, 7827, 3.4342753887176514, 4
    56, 2709, 3.4121413230895996, 3
    554, 2667, 3.3029825687408447, 1
    571, 2076, 4.026064395904541, 4
    136, 2077, 4.014613628387451, 4
    159, 1721, 2.6424131393432617, 5
    413, 2325, 3.308042526245117, 3
    463, 1502, 3.9439280033111572, 4
    287, 995, 3.1945393085479736, 3
    218, 1795, 2.864098072052002, 2
    67, 5999, 3.046928882598877, 3
    17, 5360, 3.161292791366577, 3
    437, 838, 3.0176455974578857, 4
    447, 3299, 2.5741870403289795, 0
    418, 4251, 3.6937808990478516, 3
    488, 509, 2.858548641204834, 4
    124, 9047, 3.8007779121398926, 2
    16, 910, 4.181413173675537, 5
    110, 649, 3.1347098350524902, 4
    243, 832, 3.2261970043182373, 1
    65, 2013, 3.414116859436035, 4
    176, 4607, 3.5753421783447266, 3
    379, 2491, 3.7509841918945312, 4
    312, 1825, 3.1750380992889404, 2
    67, 485, 2.909173011779785, 3
    79, 6726, 4.11798095703125, 4
    523, 1056, 3.1849560737609863, 5
    379, 1732, 3.318793296813965, 3
    229, 461, 3.2476487159729004, 1
    317, 560, 3.456162452697754, 4
    609, 8909, 3.399722099304199, 3
    413, 3418, 3.11752986907959, 3
    225, 2746, 2.7556025981903076, 3
    304, 8165, 3.113522529602051, 4
    609, 2903, 3.5457139015197754, 5
    524, 4360, 3.650017261505127, 4
    304, 6512, 3.541250228881836, 4
    296, 62, 2.7035608291625977, 2
    281, 2637, 3.7400436401367188, 4
    413, 2046, 3.4362406730651855, 4
    325, 5834, 3.5497894287109375, 2
    444, 4355, 3.561066150665283, 3
    452, 2625, 4.001351356506348, 4
    287, 2986, 2.7604126930236816, 3
    112, 885, 3.9305367469787598, 5
    222, 418, 3.211095094680786, 3
    437, 3572, 3.1470861434936523, 3
    198, 4604, 3.1618919372558594, 4
    371, 793, 3.551255464553833, 5
    304, 733, 4.126331806182861, 4
    124, 6755, 4.110448837280273, 4
    598, 2137, 2.2991697788238525, 3
    63, 5724, 3.1889052391052246, 5
    46, 4007, 2.9603312015533447, 4
    289, 1530, 3.6168558597564697, 4
    433, 510, 3.977968692779541, 4
    9, 4136, 3.062657356262207, 3
    67, 4631, 2.705171585083008, 3
    386, 3633, 3.6622653007507324, 3
    527, 7855, 2.950317859649658, 0
    599, 6226, 2.7885546684265137, 2
    44, 1318, 3.4447360038757324, 4
    589, 4633, 3.3658480644226074, 2
    225, 3697, 3.1728832721710205, 2
    49, 976, 2.6358635425567627, 3
    554, 166, 3.263309955596924, 3
    580, 8861, 3.7477898597717285, 4
    609, 6490, 3.2166690826416016, 3
    554, 1770, 3.1334707736968994, 3
    473, 2993, 3.0081870555877686, 2
    166, 862, 4.074127197265625, 4
    379, 2580, 3.4153692722320557, 4
    560, 6134, 3.470210075378418, 3
    570, 956, 3.0005695819854736, 5
    60, 4607, 3.808255195617676, 4
    599, 308, 2.685551404953003, 3
    598, 254, 2.762357711791992, 5
    509, 5938, 3.0161166191101074, 0
    231, 483, 3.4748892784118652, 3
    136, 2796, 3.562380790710449, 4
    238, 1515, 3.165264129638672, 4
    132, 409, 2.7454452514648438, 4
    381, 4631, 3.3740901947021484, 3
    572, 6910, 3.80415678024292, 5
    245, 2192, 4.0028204917907715, 4
    284, 4853, 3.8278889656066895, 3
    108, 24, 3.013427972793579, 5
    491, 815, 3.783719062805176, 3
    338, 7710, 3.699706554412842, 3
    403, 450, 2.8609657287597656, 3
    110, 4926, 3.6242923736572266, 3
    222, 2257, 3.293937921524048, 3
    367, 2921, 2.5366225242614746, 1
    508, 5955, 2.9753541946411133, 1
    317, 2760, 3.435619354248047, 3
    159, 2194, 2.9707584381103516, 2
    231, 938, 3.5017857551574707, 4
    7, 1, 3.285299301147461, 4
    138, 8354, 2.173828601837158, 3
    474, 5250, 4.3460493087768555, 4
    479, 443, 3.0766453742980957, 4
    152, 461, 2.6740822792053223, 2
    602, 2421, 3.707807779312134, 5
    482, 4551, 3.655811071395874, 3
    448, 1971, 2.5428404808044434, 2
    447, 7241, 3.08293080329895, 4
    306, 5834, 2.2694716453552246, 4
    380, 6182, 3.3414430618286133, 3
    79, 5317, 3.8269214630126953, 4
    447, 9150, 2.6438910961151123, 2
    263, 828, 3.951223850250244, 3
    185, 3827, 3.9047114849090576, 4
    18, 2670, 2.9474568367004395, 3
    218, 2475, 2.9659762382507324, 4
    225, 1080, 3.120448350906372, 3
    563, 3774, 2.958104372024536, 3
    59, 277, 4.209514141082764, 4
    605, 147, 3.486804246902466, 3
    239, 398, 4.492747783660889, 3
    3, 189, 3.592761754989624, 1
    187, 912, 4.698354721069336, 5
    605, 1085, 3.3471055030822754, 4
    293, 659, 3.348057746887207, 5
    604, 224, 3.7025222778320312, 5
    287, 2925, 3.2130651473999023, 3
    291, 3189, 3.728659152984619, 4
    87, 2596, 3.167433261871338, 4
    483, 2803, 3.5435352325439453, 2
    366, 3164, 3.7779009342193604, 4
    380, 5213, 3.0636794567108154, 4
    127, 693, 4.116211414337158, 4
    605, 5884, 3.2520434856414795, 4
    67, 5253, 2.689661979675293, 4
    304, 510, 4.3265790939331055, 5
    398, 224, 4.239384651184082, 5
    413, 1100, 3.2525134086608887, 3
    63, 1373, 3.631173610687256, 2
    476, 295, 3.7927825450897217, 4
    74, 709, 2.8294880390167236, 4
    554, 3020, 3.5963997840881348, 3
    50, 3808, 3.227825403213501, 3
    229, 1066, 3.168895959854126, 1
    505, 6321, 3.2373688220977783, 3
    68, 3283, 4.185297012329102, 4
    376, 6614, 3.55413818359375, 0
    331, 3189, 3.568181037902832, 4
    186, 6276, 3.489915370941162, 0
    14, 7058, 3.365746259689331, 4
    533, 7613, 3.2307326793670654, 2
    513, 3568, 3.1295106410980225, 2
    367, 1118, 2.595717430114746, 3
    110, 2400, 3.2374491691589355, 4
    56, 2325, 3.3541698455810547, 2
    413, 1557, 3.160719156265259, 3
    287, 197, 2.8006227016448975, 3
    226, 6512, 3.827927589416504, 2
    607, 1486, 2.7006189823150635, 2
    583, 9, 3.7991995811462402, 5
    447, 131, 2.558690309524536, 3
    324, 1484, 3.4042234420776367, 4
    198, 5379, 3.1111226081848145, 3
    463, 6755, 3.8457841873168945, 2
    554, 2038, 3.291181802749634, 2
    473, 5939, 2.8265366554260254, 4
    585, 8377, 4.49196195602417, 5
    352, 398, 3.7625315189361572, 5
    545, 328, 2.788555383682251, 4
    473, 4654, 2.801823616027832, 3
    17, 1916, 3.4131600856781006, 4
    159, 888, 2.955000877380371, 5
    447, 4648, 2.739459753036499, 4
    44, 1128, 3.5653228759765625, 3
    363, 659, 4.580896377563477, 5
    502, 419, 2.928632974624634, 4
    488, 1059, 2.54425048828125, 3
    232, 6298, 3.5378618240356445, 4
    218, 4527, 2.8194808959960938, 2
    317, 8646, 3.3171117305755615, 3
    154, 1882, 4.104535102844238, 5
    600, 8677, 4.067282676696777, 4
    128, 2192, 3.7463326454162598, 4
    171, 2328, 2.930748701095581, 5
    155, 905, 4.063242435455322, 4
    413, 7763, 3.0082473754882812, 4
    272, 379, 2.863987445831299, 4
    420, 1729, 4.101990699768066, 5
    607, 2030, 2.7563438415527344, 2
    447, 9139, 2.8111047744750977, 1
    88, 7042, 2.8325650691986084, 5
    605, 4708, 3.4213035106658936, 2
    248, 9434, 3.4615938663482666, 4
    537, 7061, 4.292609691619873, 4
    14, 910, 3.79160213470459, 5
    82, 6416, 2.8737356662750244, 1
    181, 3849, 3.412259340286255, 2
    598, 1429, 2.750338077545166, 3
    67, 1059, 2.4256999492645264, 3
    579, 938, 3.3559792041778564, 3
    386, 2013, 2.8072354793548584, 3
    41, 2300, 3.71321439743042, 3
    380, 3617, 4.002018451690674, 5
    37, 140, 2.6756999492645264, 4
    166, 311, 3.704953908920288, 5
    424, 3617, 4.002711296081543, 4
    595, 5259, 3.1709070205688477, 3
    186, 97, 3.933868408203125, 3
    112, 189, 3.5899031162261963, 4
    381, 4327, 2.7194719314575195, 3
    218, 862, 3.576427936553955, 3
    598, 6294, 2.6941897869110107, 3
    473, 4690, 3.170833110809326, 4
    65, 2296, 3.496873617172241, 3
    273, 1754, 2.9218006134033203, 3
    104, 2832, 4.0941243171691895, 3
    296, 84, 2.409632682800293, 1
    598, 5651, 2.275297164916992, 3
    208, 1502, 4.357327938079834, 4
    595, 6517, 3.3118016719818115, 4
    273, 2401, 2.5229909420013428, 2
    524, 3544, 3.101886034011841, 3
    576, 1497, 2.871866226196289, 4
    404, 2505, 3.2398886680603027, 4
    35, 1134, 2.4715263843536377, 0
    478, 1163, 2.999673843383789, 2
    81, 3629, 2.7337968349456787, 4
    213, 533, 2.90982985496521, 3
    409, 965, 4.497566223144531, 4
    533, 6737, 3.1856353282928467, 3
    304, 7670, 3.6340713500976562, 4
    367, 1210, 3.0641860961914062, 5
    447, 7425, 2.964848041534424, 4
    104, 3191, 4.121247291564941, 3
    604, 7244, 2.9233031272888184, 3
    263, 31, 4.1029863357543945, 1
    248, 6237, 3.4282829761505127, 3
    559, 4875, 3.0969107151031494, 3
    186, 4760, 3.483248472213745, 5
    414, 3282, 3.380427122116089, 3
    598, 709, 2.295222043991089, 4
    331, 6693, 3.609471321105957, 4
    287, 6734, 3.1101911067962646, 3
    424, 1521, 3.1363332271575928, 2
    385, 99, 2.62534236907959, 4
    491, 224, 4.231431007385254, 4
    118, 6396, 3.384460926055908, 3
    221, 5550, 3.2484796047210693, 4
    559, 5333, 2.896317720413208, 3
    179, 2806, 3.03704833984375, 3
    536, 7879, 3.765244483947754, 3
    317, 8507, 3.0692429542541504, 4
    41, 1772, 3.594648599624634, 5
    366, 2806, 3.669553756713867, 4
    199, 5298, 3.9899532794952393, 5
    134, 899, 3.9616289138793945, 4
    379, 7628, 3.7802066802978516, 5
    457, 156, 3.6150474548339844, 5
    523, 422, 2.78006649017334, 4
    180, 162, 3.455232858657837, 1
    56, 2075, 3.1877341270446777, 4
    392, 5997, 3.3415884971618652, 4
    139, 883, 3.499898672103882, 3
    379, 8421, 3.9371843338012695, 4
    45, 197, 3.445631742477417, 3
    44, 1182, 3.702564239501953, 3
    596, 797, 3.950967311859131, 4
    345, 859, 3.4254419803619385, 4
    70, 23, 3.079838991165161, 2
    551, 2055, 2.6482415199279785, 1
    41, 1149, 3.3115150928497314, 3
    159, 2037, 2.974161148071289, 4
    142, 6641, 3.492280960083008, 3
    609, 6195, 3.1551074981689453, 3
    333, 2832, 3.2855618000030518, 3
    127, 705, 4.602500915527344, 4
    18, 1490, 2.9423232078552246, 3
    389, 5132, 3.4105451107025146, 5
    590, 2899, 2.7772436141967773, 3
    379, 9128, 3.6582605838775635, 3
    379, 6320, 3.6763525009155273, 5
    386, 2649, 2.8335278034210205, 3
    473, 1308, 3.062779188156128, 3
    598, 3375, 2.687300443649292, 2
    413, 380, 3.44596004486084, 2
    65, 2407, 3.8331618309020996, 4
    433, 2347, 3.385530471801758, 3
    67, 4637, 2.6448261737823486, 4
    185, 1070, 3.9842870235443115, 4
    238, 6517, 3.8975822925567627, 5
    63, 32, 3.49101185798645, 4
    609, 6785, 2.959735155105591, 2
    0, 2285, 4.3084797859191895, 5
    152, 8287, 2.3160390853881836, 1
    353, 1477, 3.4682207107543945, 4
    118, 7902, 4.087143421173096, 4
    231, 5986, 3.0203640460968018, 3
    152, 6225, 2.2571778297424316, 4
    566, 6886, 1.9936283826828003, 2
    476, 2001, 3.355930805206299, 4
    139, 2727, 3.6379597187042236, 3
    561, 6490, 3.7397217750549316, 4
    482, 4858, 3.061396598815918, 4
    404, 97, 3.9403421878814697, 4
    423, 3536, 3.0091819763183594, 3
    167, 4494, 4.251328468322754, 4
    589, 3002, 3.1619372367858887, 3
    390, 2037, 3.905418634414673, 4
    559, 5693, 3.010657787322998, 2
    461, 5993, 3.5068671703338623, 3
    589, 2096, 3.4514734745025635, 3
    61, 6059, 3.615440607070923, 4
    263, 124, 4.1124982833862305, 2
    609, 7390, 2.5927412509918213, 2
    176, 7749, 3.0202040672302246, 4
    333, 3470, 2.9776511192321777, 3
    304, 6341, 3.716442346572876, 4
    475, 472, 3.5269455909729004, 4
    297, 472, 2.1649227142333984, 2
    519, 6755, 4.215044975280762, 4
    248, 4541, 3.6787075996398926, 3
    605, 6505, 3.6025631427764893, 3
    561, 994, 3.5912113189697266, 3
    4, 126, 3.134214162826538, 3
    5, 15, 3.6147079467773438, 4
    379, 7853, 3.5167388916015625, 3
    609, 8119, 3.082310914993286, 2
    32, 1330, 3.6137866973876953, 4
    301, 607, 3.2177891731262207, 2
    287, 1231, 3.031116485595703, 4
    598, 8208, 2.3367233276367188, 3
    361, 8237, 3.68076753616333, 4
    566, 9655, 2.132873773574829, 1
    99, 1822, 3.3944883346557617, 5
    462, 1502, 4.096864700317383, 4
    371, 862, 3.8148245811462402, 3
    228, 18, 2.957446336746216, 3
    242, 422, 3.4162049293518066, 3
    322, 145, 2.045104503631592, 3
    476, 313, 2.9182803630828857, 0
    274, 35, 4.0175628662109375, 4
    27, 2389, 2.5527725219726562, 2
    447, 6817, 2.99749493598938, 2
    218, 6471, 2.891571044921875, 4
    240, 1729, 3.787813901901245, 4
    609, 2970, 3.116464853286743, 2
    386, 953, 2.761265516281128, 1
    481, 838, 3.16304349899292, 4
    380, 989, 3.686619758605957, 3
    410, 504, 3.0229873657226562, 4
    427, 1703, 2.7912676334381104, 3
    121, 5190, 4.2837724685668945, 4
    27, 6538, 2.3931164741516113, 2
    358, 546, 3.230849027633667, 3
    538, 461, 4.016116142272949, 3
    41, 2832, 3.8384523391723633, 2
    570, 2764, 2.510529041290283, 1
    124, 9225, 3.6153340339660645, 2
    267, 895, 3.8295109272003174, 4
    317, 7919, 3.26611328125, 3
    596, 3668, 4.237743377685547, 5
    181, 5915, 3.056915283203125, 2
    317, 8133, 3.400526523590088, 3
    317, 2942, 3.3923492431640625, 4
    3, 2677, 3.4228296279907227, 1
    87, 4392, 2.8900835514068604, 0
    72, 6726, 3.606522798538208, 4
    392, 7602, 3.735255718231201, 5
    299, 8435, 3.0013227462768555, 5
    65, 4664, 3.4906983375549316, 4
    516, 1808, 2.4320359230041504, 0
    598, 785, 2.9891867637634277, 2
    287, 545, 2.99617600440979, 3
    274, 2555, 4.187383651733398, 5
    598, 2340, 2.13470458984375, 2
    474, 1043, 4.356963157653809, 5
    355, 92, 3.7457234859466553, 3
    554, 2832, 3.888887882232666, 3
    479, 514, 3.079497814178467, 3
    482, 1070, 3.110938549041748, 3
    473, 2619, 3.367133855819702, 3
    473, 1372, 2.2027318477630615, 2
    248, 7789, 3.3143486976623535, 4
    248, 7902, 3.4598567485809326, 3
    453, 1445, 3.18528151512146, 2
    413, 1806, 3.134122848510742, 4
    599, 1029, 2.6451220512390137, 2
    356, 461, 4.221920013427734, 5
    211, 7970, 3.1463139057159424, 4
    479, 3609, 3.441032886505127, 3
    589, 4904, 2.701753616333008, 2
    291, 6676, 2.8660614490509033, 3
    356, 1562, 3.5649726390838623, 3
    605, 2217, 3.7415361404418945, 4
    306, 1643, 2.098431348800659, 2
    599, 1134, 2.540031909942627, 3
    367, 2927, 3.085855007171631, 2
    368, 4631, 2.941697835922241, 3
    343, 504, 3.3463165760040283, 5
    176, 7620, 2.9813530445098877, 3
    184, 2652, 2.9789819717407227, 3
    286, 4131, 2.5812909603118896, 4
    327, 1423, 3.184488296508789, 3
    386, 5427, 3.0593972206115723, 4
    216, 1274, 2.60627818107605, 1
    27, 4568, 3.08378267288208, 2
    67, 6757, 2.42582368850708, 3
    419, 4696, 3.5304059982299805, 3
    600, 9586, 3.9072318077087402, 4
    19, 3537, 3.5201408863067627, 2
    109, 6049, 3.0101442337036133, 3
    488, 6501, 2.9390664100646973, 3
    482, 792, 3.6799581050872803, 3
    356, 2224, 4.35820198059082, 4
    17, 8271, 2.8601174354553223, 3
    599, 1810, 2.7123422622680664, 3
    413, 4875, 3.2383456230163574, 3
    483, 1742, 3.279406785964966, 3
    429, 1796, 3.568577289581299, 5
    297, 3133, 2.100766897201538, 2
    338, 7354, 3.760529041290283, 4
    501, 906, 3.5621795654296875, 5
    90, 192, 3.1712372303009033, 3
    598, 1235, 2.367810010910034, 3
    306, 3569, 2.438225269317627, 3
    320, 238, 3.245028018951416, 3
    508, 4131, 3.3518869876861572, 5
    218, 4112, 3.1724491119384766, 3
    379, 527, 3.7532215118408203, 3
    291, 6886, 3.2359843254089355, 4
    17, 3633, 3.752211332321167, 4
    279, 7637, 3.7268171310424805, 3
    44, 2405, 3.6685495376586914, 4
    238, 6314, 4.173378944396973, 4
    297, 4787, 2.4828238487243652, 2
    104, 6298, 4.38795280456543, 5
    90, 1485, 3.156726837158203, 2
    433, 1476, 3.1458585262298584, 0
    561, 1107, 3.7218034267425537, 4
    381, 6075, 2.7740800380706787, 3
    425, 7647, 3.279874801635742, 4
    44, 3056, 3.5360403060913086, 1
    274, 3133, 3.9575142860412598, 4
    329, 32, 3.3318283557891846, 3
    524, 8574, 3.237116813659668, 3
    89, 1050, 4.115630149841309, 4
    598, 6476, 2.7586424350738525, 2
    561, 7149, 3.774324893951416, 5
    27, 6654, 2.391040325164795, 2
    599, 1820, 2.532344102859497, 3
    526, 2219, 3.6412806510925293, 4
    602, 2979, 3.536940574645996, 5
    277, 1229, 3.1190927028656006, 5
    586, 1486, 3.6173391342163086, 4
    483, 686, 4.332769393920898, 2
    366, 707, 3.799316883087158, 4
    488, 3539, 2.939110517501831, 2
    62, 4918, 3.447941780090332, 2
    287, 1307, 2.8498427867889404, 2
    435, 494, 3.0220634937286377, 3
    273, 6395, 2.6474032402038574, 2
    379, 648, 3.586254119873047, 2
    31, 277, 4.319576740264893, 5
    110, 8204, 2.9413466453552246, 4
    473, 594, 3.085000991821289, 2
    279, 945, 4.1649580001831055, 4
    231, 4541, 3.360353469848633, 3
    524, 6277, 3.3362669944763184, 4
    104, 2035, 3.6737561225891113, 4
    0, 2432, 4.172501564025879, 3
    229, 6416, 2.6833183765411377, 3
    599, 783, 2.6114084720611572, 3
    513, 3735, 2.988553762435913, 2
    426, 3821, 2.8368310928344727, 4
    49, 5682, 3.083070755004883, 3
    139, 930, 3.6769213676452637, 4
    322, 295, 3.270601511001587, 3
    451, 62, 4.422282695770264, 4
    115, 507, 3.5213708877563477, 4
    562, 7414, 2.739088535308838, 3
    247, 1938, 3.9362995624542236, 4
    231, 3409, 2.8908870220184326, 3
    386, 97, 3.530379295349121, 3
    589, 3022, 3.271096706390381, 4
    75, 483, 3.109116792678833, 0
    554, 1861, 3.5949225425720215, 3
    312, 1054, 3.3693532943725586, 2
    574, 1961, 3.015690803527832, 3
    413, 9577, 2.9995615482330322, 3
    433, 98, 3.6321053504943848, 5
    482, 5994, 3.3351502418518066, 2
    330, 7627, 3.106673002243042, 5
    218, 1290, 2.736715078353882, 0
    468, 709, 3.436415672302246, 3
    132, 277, 3.5070292949676514, 4
    572, 6429, 3.743297576904297, 4
    181, 1528, 3.2038397789001465, 2
    65, 914, 3.989049196243286, 4
    83, 164, 3.477262020111084, 4
    108, 383, 3.126492500305176, 3
    287, 2149, 3.0632565021514893, 3
    598, 398, 3.015261173248291, 3
    287, 1423, 3.2808990478515625, 3
    90, 2095, 3.313777446746826, 3
    137, 2979, 3.323256015777588, 5
    609, 2466, 2.842998504638672, 3
    185, 815, 4.3532562255859375, 5
    488, 1705, 2.64371657371521, 1
    439, 3224, 3.5168113708496094, 4
    566, 3136, 2.7702465057373047, 3
    114, 2803, 3.53212308883667, 4
    384, 511, 3.578784227371216, 3
    166, 4787, 3.693633556365967, 4
    370, 2258, 4.137264251708984, 5
    176, 7731, 3.1301801204681396, 2
    413, 3359, 3.1543948650360107, 3
    386, 6148, 3.258836030960083, 4
    201, 1444, 4.004052639007568, 5
    413, 4602, 3.3054885864257812, 3
    153, 8673, 4.014988899230957, 5
    490, 3785, 3.5187370777130127, 3
    473, 1596, 3.002202272415161, 4
    565, 485, 3.417635917663574, 1
    0, 2300, 4.375554084777832, 5
    589, 4303, 3.176168203353882, 2
    138, 2670, 2.302549362182617, 2
    305, 8577, 3.026827096939087, 4
    386, 6382, 3.319425582885742, 2
    473, 4343, 2.9772579669952393, 3
    366, 1434, 3.869978666305542, 2
    49, 7026, 2.6306369304656982, 1
    108, 504, 3.060791254043579, 3
    73, 5148, 4.273447513580322, 3
    465, 7001, 3.7293577194213867, 4
    602, 1960, 3.2492871284484863, 5
    306, 1403, 2.4775047302246094, 3
    159, 1057, 2.628580093383789, 2
    118, 6999, 3.731545925140381, 4
    592, 2224, 3.807581663131714, 4
    246, 733, 3.910367727279663, 2
    605, 2887, 3.506619930267334, 3
    306, 4191, 2.190958023071289, 2
    418, 4191, 3.416384220123291, 3
    50, 3946, 3.4463610649108887, 4
    605, 4715, 3.2072455883026123, 3
    607, 2437, 3.245225667953491, 4
    479, 3683, 2.7295875549316406, 4
    56, 952, 2.996703863143921, 4
    143, 123, 3.732069969177246, 3
    216, 1610, 2.304556131362915, 3
    532, 901, 3.837286949157715, 4
    390, 1035, 3.783181667327881, 4
    598, 1134, 2.1099748611450195, 2
    300, 1072, 3.067140817642212, 3
    62, 1283, 3.8863816261291504, 3
    104, 7181, 3.935947895050049, 4
    281, 4239, 3.5734877586364746, 3
    317, 6895, 2.7357099056243896, 3
    598, 9642, 2.3908400535583496, 3
    297, 6805, 1.7493460178375244, 3
    437, 4070, 3.58927321434021, 4
    287, 2665, 2.8168859481811523, 3
    248, 6993, 4.010679244995117, 5
    75, 906, 3.44730544090271, 4
    601, 134, 3.190958023071289, 4
    508, 6629, 2.8762218952178955, 3
    491, 637, 3.3760299682617188, 3
    293, 797, 2.6914329528808594, 2
    523, 156, 2.8834426403045654, 3
    199, 546, 3.6583986282348633, 3
    293, 960, 2.8735427856445312, 4
    521, 3136, 4.028872489929199, 3
    27, 6596, 2.8672595024108887, 4
    379, 6501, 3.8976759910583496, 2
    285, 1075, 3.209061861038208, 3
    367, 1270, 2.8630640506744385, 3
    133, 43, 3.7841620445251465, 4
    566, 7749, 1.8082141876220703, 1
    233, 322, 4.351413726806641, 5
    602, 2235, 3.979078769683838, 5
    607, 311, 3.1591968536376953, 4
    86, 862, 4.1148786544799805, 5
    554, 1128, 3.511960983276367, 5
    596, 691, 4.240222930908203, 4
    211, 8888, 3.491875648498535, 4
    306, 6948, 2.367422342300415, 4
    367, 1030, 3.117720365524292, 3
    225, 302, 2.802238941192627, 4
    473, 5943, 3.0148701667785645, 4
    67, 3785, 2.3249623775482178, 3
    447, 5278, 3.2269434928894043, 2
    109, 2077, 3.9958510398864746, 3
    379, 454, 3.2273712158203125, 2
    80, 277, 3.7518739700317383, 3
    447, 2550, 3.0746967792510986, 3
    176, 5719, 3.5507493019104004, 5
    118, 5245, 4.131138324737549, 3
    364, 7407, 2.422992467880249, 1
    437, 5753, 2.7053182125091553, 1
    306, 5303, 2.1730265617370605, 2
    473, 6465, 3.1945064067840576, 3
    201, 470, 4.0164713859558105, 3
    20, 3740, 3.107452869415283, 4
    352, 61, 3.06392765045166, 3
    155, 1521, 3.3251280784606934, 2
    599, 913, 3.3801703453063965, 2
    220, 901, 3.9022204875946045, 5
    401, 333, 3.2976744174957275, 4
    607, 6334, 2.821573257446289, 3
    287, 224, 3.678196668624878, 5
    607, 1157, 3.0503361225128174, 4
    464, 1978, 3.660795211791992, 2
    179, 1986, 3.1518375873565674, 3
    531, 3217, 4.410928726196289, 3
    451, 2441, 4.021400451660156, 4
    138, 3814, 2.084851026535034, 3
    559, 862, 3.8145806789398193, 4
    107, 1938, 4.381115913391113, 5
    303, 549, 3.5186359882354736, 4
    552, 2903, 4.072268009185791, 5
    327, 1266, 3.2680282592773438, 3
    90, 504, 3.188880205154419, 2
    152, 8565, 1.9675301313400269, 2
    572, 3671, 3.710819959640503, 4
    473, 3829, 3.124908685684204, 3
    229, 507, 3.0836517810821533, 0
    26, 1186, 3.5734477043151855, 4
    413, 1283, 3.762873411178589, 5
    607, 1208, 3.0936341285705566, 3
    123, 6388, 3.914429187774658, 4
    584, 1733, 4.57197380065918, 5
    466, 1794, 3.761317253112793, 2
    289, 2875, 3.8427963256835938, 4
    81, 899, 3.6376771926879883, 4
    78, 1671, 4.104642868041992, 4
    306, 2691, 2.0604043006896973, 2
    273, 6592, 2.784109354019165, 4
    88, 7550, 2.9355268478393555, 5
    489, 7684, 2.6586766242980957, 3
    124, 6850, 3.128647565841675, 4
    115, 5721, 3.4777932167053223, 4
    603, 257, 3.826667547225952, 5
    533, 3605, 3.4042696952819824, 3
    554, 1147, 3.526541233062744, 4
    338, 418, 3.9229159355163574, 4
    20, 1519, 3.0074338912963867, 3
    6, 5331, 2.5303845405578613, 1
    386, 3136, 3.7955799102783203, 4
    325, 5259, 3.674321174621582, 5
    366, 2697, 3.578375816345215, 4
    423, 7142, 3.4636688232421875, 2
    508, 8035, 2.988450050354004, 3
    589, 1324, 3.1463797092437744, 3
    605, 2945, 3.450298547744751, 3
    447, 8364, 2.9809603691101074, 1
    399, 968, 3.885003089904785, 4
    482, 7995, 3.4148662090301514, 4
    554, 2763, 3.5134992599487305, 3
    99, 66, 3.492063045501709, 4
    554, 1080, 3.606510639190674, 3
    476, 3189, 3.8114256858825684, 4
    190, 33, 4.080999374389648, 5
    26, 1503, 3.9813296794891357, 2
    533, 8161, 3.5361979007720947, 2
    488, 4428, 2.5137104988098145, 0
    265, 1673, 3.295494556427002, 1
    596, 2316, 3.985248327255249, 5
    274, 2996, 3.9087748527526855, 3
    605, 3085, 3.826950788497925, 4
    3, 1198, 3.399533987045288, 1
    97, 2996, 3.4325506687164307, 4
    599, 1437, 3.003204345703125, 3
    79, 1661, 4.173534393310547, 4
    226, 277, 4.611972808837891, 4
    92, 1, 3.898123264312744, 5
    604, 1986, 3.0293593406677246, 3
    291, 1996, 3.1005911827087402, 1
    473, 1548, 3.143265962600708, 4
    90, 4153, 3.412186622619629, 2
    605, 1756, 3.6160101890563965, 3
    297, 2486, 2.298495054244995, 3
    88, 6233, 3.365692615509033, 2
    589, 2610, 3.549389600753784, 3
    384, 792, 3.7983596324920654, 3
    473, 1443, 3.183943033218384, 3
    591, 18, 2.9114584922790527, 4
    598, 2076, 2.4918079376220703, 3
    392, 7195, 3.671842098236084, 0
    578, 2543, 3.4342377185821533, 5
    447, 7792, 2.7053308486938477, 2
    386, 1757, 2.818723678588867, 3
    413, 1403, 3.437387466430664, 5
    110, 2476, 3.2471156120300293, 4
    204, 224, 3.8234105110168457, 4
    104, 5916, 3.9744374752044678, 4
    364, 7285, 2.7415153980255127, 2
    166, 506, 3.6329588890075684, 3
    386, 1389, 3.2244818210601807, 2
    454, 97, 3.6448636054992676, 4
    410, 376, 2.634976625442505, 3
    523, 403, 3.32827091217041, 2
    248, 8677, 3.552013397216797, 4
    253, 6293, 3.3421783447265625, 2
    482, 5291, 3.579712152481079, 5
    304, 6136, 3.699774980545044, 5
    609, 6045, 3.4281225204467773, 3
    380, 6320, 3.295322895050049, 3
    605, 202, 3.4005050659179688, 3
    404, 2992, 3.871799945831299, 4
    248, 7308, 3.30690336227417, 3
    390, 1945, 4.101775646209717, 4
    90, 906, 3.911435127258301, 5
    462, 1274, 3.1674141883850098, 4
    50, 1683, 3.60375714302063, 2
    479, 5324, 3.386613368988037, 4
    67, 3868, 2.846773386001587, 3
    218, 3951, 2.641794204711914, 3
    62, 3554, 3.5418272018432617, 3
    273, 147, 2.852787733078003, 3
    386, 897, 3.534029483795166, 4
    181, 520, 3.6870322227478027, 5
    116, 31, 3.872767448425293, 3
    155, 3140, 3.2908754348754883, 3
    273, 1134, 2.541252374649048, 3
    598, 3488, 2.0257580280303955, 1
    516, 4494, 2.525416135787964, 3
    516, 9309, 2.098940849304199, 3
    285, 3624, 2.9045677185058594, 3
    83, 767, 3.192537307739258, 3
    473, 765, 2.753826141357422, 2
    361, 8035, 3.8021411895751953, 4
    248, 1939, 3.630953073501587, 3
    336, 163, 3.6793296337127686, 4
    47, 5250, 3.865281820297241, 4
    593, 472, 3.5668179988861084, 5
    67, 396, 2.300123929977417, 2
    62, 2650, 3.667161226272583, 2
    398, 2144, 4.1017560958862305, 0
    32, 31, 4.214508533477783, 3
    17, 1397, 2.7673559188842773, 3
    50, 1670, 3.8384757041931152, 5
    566, 8004, 2.0771284103393555, 2
    176, 2037, 3.444089889526367, 4
    431, 7787, 2.6116673946380615, 4
    90, 1615, 3.8106603622436523, 4
    424, 7208, 3.336531400680542, 4
    124, 6993, 4.138875484466553, 2
    609, 6453, 2.686641216278076, 3
    570, 2754, 2.507246971130371, 2
    445, 133, 2.6632065773010254, 3
    595, 3979, 3.473832130432129, 4
    482, 1733, 3.83811354637146, 3
    560, 8451, 3.2122764587402344, 3
    404, 563, 3.7481045722961426, 4
    231, 6526, 3.305461883544922, 3
    598, 6189, 2.4454753398895264, 2
    427, 141, 2.4011831283569336, 0
    4, 322, 4.107696533203125, 3
    487, 744, 3.57792592048645, 2
    317, 7008, 3.446791172027588, 3
    447, 7937, 2.9994046688079834, 4
    572, 4541, 3.751434326171875, 4
    607, 291, 3.0744376182556152, 4
    599, 244, 2.8880434036254883, 2
    569, 217, 3.1933884620666504, 3
    488, 6008, 2.9152090549468994, 4
    559, 4192, 2.9586896896362305, 3
    371, 1072, 3.167296886444092, 3
    336, 2, 3.852203369140625, 4
    602, 925, 3.9273858070373535, 4
    581, 7448, 3.7489256858825684, 5
    390, 2466, 3.28865909576416, 1
    51, 6298, 4.6948394775390625, 4
    482, 7468, 3.0563864707946777, 3
    118, 6195, 3.9976658821105957, 4
    0, 2156, 4.0249223709106445, 5
    128, 3282, 3.398467540740967, 3
    447, 7658, 3.106041193008423, 2
    176, 2042, 2.7681710720062256, 2
    259, 1401, 3.6624770164489746, 4
    513, 1471, 3.2859692573547363, 4
    273, 1935, 2.979844808578491, 3
    560, 5938, 3.411893844604492, 4
    218, 3766, 2.9249279499053955, 2
    522, 6621, 4.436416149139404, 4
    545, 2918, 2.932103157043457, 5
    306, 5999, 2.6908814907073975, 3
    386, 418, 3.361112117767334, 3
    220, 2741, 4.038084030151367, 4
    61, 9627, 3.867393970489502, 3
    384, 1059, 3.2002158164978027, 3
    265, 224, 4.059207916259766, 4
    554, 1572, 3.6165037155151367, 3
    479, 5995, 2.6419339179992676, 2
    367, 20, 2.6670987606048584, 3
    516, 700, 2.413088321685791, 4
    5, 434, 3.751845359802246, 3
    498, 1800, 3.657595157623291, 4
    421, 2535, 3.103259563446045, 3
    67, 4950, 2.5633885860443115, 3
    605, 81, 3.1328201293945312, 3
    473, 3584, 3.5490410327911377, 4
    598, 1084, 1.7107648849487305, 1
    201, 898, 4.366791248321533, 4
    301, 33, 3.8745667934417725, 4
    559, 6204, 2.995429754257202, 3
    328, 8943, 3.2606523036956787, 5
    491, 551, 3.5948476791381836, 3
    178, 619, 2.949069023132324, 3
    176, 325, 3.370360851287842, 3
    181, 2246, 3.403627634048462, 2
    246, 898, 3.896226406097412, 5
    63, 1235, 3.290902614593506, 2
    285, 2077, 3.818567991256714, 5
    14, 3456, 3.0525407791137695, 5
    436, 940, 4.087031841278076, 5
    201, 1002, 3.4434351921081543, 4
    0, 2019, 4.391442775726318, 5
    41, 1474, 3.4793128967285156, 5
    366, 659, 4.510711669921875, 5
    279, 1397, 3.3436193466186523, 3
    104, 2205, 3.4686026573181152, 4
    296, 431, 2.693002700805664, 2
    419, 1971, 3.376269578933716, 3
    233, 1084, 3.018087387084961, 5
    433, 5901, 3.620774030685425, 4
    404, 4781, 3.4832916259765625, 4
    479, 2632, 3.382906436920166, 2
    552, 592, 3.743420124053955, 3
    536, 7355, 4.1294097900390625, 5
    379, 7114, 3.555121898651123, 3
    73, 6629, 4.073603630065918, 4
    333, 1210, 3.3855297565460205, 4
    364, 7137, 2.4825220108032227, 3
    473, 3955, 3.292513132095337, 2
    609, 5648, 3.1879405975341797, 5
    592, 6134, 3.396078586578369, 4
    104, 4907, 4.165689468383789, 5
    419, 2224, 4.251934051513672, 4
    521, 7043, 3.5162620544433594, 2
    273, 784, 2.732638120651245, 2
    27, 2620, 2.552112340927124, 3
    494, 6697, 3.4705634117126465, 4
    488, 901, 3.0306942462921143, 4
    403, 395, 3.5777769088745117, 3
    566, 7601, 1.768223524093628, 2
    46, 8336, 2.734253168106079, 3
    297, 6298, 2.7272212505340576, 3
    37, 229, 3.1235928535461426, 1
    413, 3326, 3.7713241577148438, 3
    608, 123, 3.4067935943603516, 3
    63, 1297, 3.595458507537842, 5
    476, 4112, 3.6290459632873535, 4
    259, 3449, 3.6755969524383545, 4
    86, 2911, 3.734086513519287, 4
    247, 275, 3.324519157409668, 3
    103, 3785, 3.1829073429107666, 4
    561, 6629, 3.78092098236084, 5
    96, 3699, 3.663602828979492, 4
    409, 2640, 3.7944278717041016, 2
    353, 5869, 3.499650478363037, 4
    369, 2962, 2.659640073776245, 3
    582, 1939, 3.2812209129333496, 5
    90, 2173, 3.3936820030212402, 2
    37, 257, 3.576213836669922, 3
    473, 4209, 3.355595111846924, 1
    293, 1858, 2.591482162475586, 1
    216, 1360, 2.719702959060669, 2
    601, 224, 3.7184224128723145, 5
    353, 1397, 3.0460400581359863, 3
    476, 197, 3.0747761726379395, 3
    221, 3241, 2.6968095302581787, 2
    310, 36, 2.344050407409668, 3
    286, 1145, 2.103956699371338, 2
    273, 1144, 2.7091665267944336, 3
    532, 1882, 4.083095073699951, 4
    437, 6192, 3.174227714538574, 0
    90, 1971, 3.1082804203033447, 3
    413, 2825, 3.4807257652282715, 2
    231, 5253, 3.1699600219726562, 3
    333, 6668, 3.1019370555877686, 3
    34, 156, 3.015392303466797, 5
    322, 890, 2.9435017108917236, 4
    390, 2030, 3.609194278717041, 3
    159, 454, 2.2479941844940186, 1
    191, 276, 3.6655092239379883, 3
    427, 40, 2.1765971183776855, 2
    592, 5363, 3.5421838760375977, 3
    523, 138, 3.3613953590393066, 3
    231, 4223, 2.646979808807373, 4
    306, 404, 2.021979570388794, 1
    572, 2599, 3.3708879947662354, 1
    598, 4606, 1.9805066585540771, 2
    293, 2172, 3.081757068634033, 3
    219, 2224, 4.467280387878418, 1
    447, 8416, 2.8291895389556885, 1
    18, 2723, 2.510582447052002, 3
    600, 9448, 3.809680461883545, 4
    559, 8882, 3.14105224609375, 3
    437, 1198, 3.1030001640319824, 3
    329, 325, 3.2855043411254883, 1
    45, 46, 4.394503593444824, 5
    201, 890, 3.922128677368164, 4
    386, 483, 3.3756473064422607, 3
    413, 7756, 3.4801251888275146, 2
    478, 650, 3.3305788040161133, 5
    67, 4607, 3.007629871368408, 3
    273, 2636, 2.2828574180603027, 2
    63, 2631, 3.478682041168213, 4
    197, 961, 3.598933219909668, 3
    422, 418, 3.877939224243164, 5
    16, 1733, 4.178683280944824, 4
    305, 8457, 3.1784486770629883, 3
    38, 2285, 3.788780689239502, 3
    248, 8675, 3.537374496459961, 4
    447, 4756, 2.9506261348724365, 4
    579, 1217, 3.4045188426971436, 4
    330, 7758, 3.3485608100891113, 5
    198, 938, 3.424609422683716, 5
    201, 5, 4.232549667358398, 5
    413, 6629, 3.2783234119415283, 4
    375, 615, 3.682020664215088, 4
    479, 630, 2.956676483154297, 3
    473, 5393, 2.866190195083618, 4
    16, 2038, 3.321232795715332, 3
    262, 1796, 3.869938373565674, 4
    61, 7498, 3.970609664916992, 4
    598, 8665, 2.431614875793457, 3
    15, 6693, 3.742067813873291, 4
    602, 523, 3.7258338928222656, 3
    382, 944, 3.704824686050415, 5
    106, 133, 3.0589303970336914, 4
    562, 8611, 2.92513108253479, 3
    479, 201, 3.1746838092803955, 3
    273, 1486, 2.674652099609375, 3
    248, 8035, 3.5853676795959473, 4
    297, 3814, 2.226357936859131, 2
    605, 1883, 3.118011474609375, 4
    447, 462, 2.9232215881347656, 2
    79, 3609, 4.314718246459961, 4
    589, 1996, 3.0682578086853027, 3
    273, 7283, 2.7401537895202637, 3
    324, 2037, 3.5000455379486084, 4
    181, 943, 3.3845090866088867, 4
    453, 2189, 3.4736623764038086, 4
    77, 418, 3.261380910873413, 4
    287, 2642, 3.2598867416381836, 3
    311, 3409, 3.278106927871704, 3
    413, 15, 3.615097999572754, 3
    447, 1072, 2.9223203659057617, 4
    524, 7073, 3.27089262008667, 4
    329, 7355, 3.4372401237487793, 5
    390, 476, 3.7207279205322266, 4
    176, 7439, 3.168882369995117, 3
    571, 895, 4.589654922485352, 5
    424, 1324, 3.277458429336548, 4
    231, 2856, 2.8666555881500244, 3
    589, 2940, 3.2418789863586426, 2
    413, 2350, 2.8942739963531494, 4
    181, 3619, 3.015249729156494, 3
    413, 4223, 2.770517587661743, 3
    595, 989, 3.4113450050354004, 3
    273, 5039, 2.718313455581665, 3
    304, 7008, 3.5721423625946045, 3
    311, 2077, 4.108606338500977, 4
    421, 1913, 2.974083423614502, 3
    304, 6505, 3.7010130882263184, 3
    585, 2353, 4.275799751281738, 4
    516, 6201, 2.1214494705200195, 2
    293, 1970, 2.230663299560547, 1
    487, 3889, 3.401437759399414, 4
    239, 340, 3.920583724975586, 5
    212, 1051, 3.4435553550720215, 3
    197, 613, 3.685180902481079, 4
    476, 944, 3.6427621841430664, 4
    598, 153, 2.340367555618286, 1
    50, 2419, 3.4216372966766357, 4
    304, 7756, 3.758979320526123, 4
    360, 5368, 2.552584409713745, 3
    305, 7911, 3.197462797164917, 3
    364, 474, 2.9626941680908203, 1
    421, 2729, 3.6111817359924316, 4
    606, 337, 3.724936008453369, 4
    99, 1877, 3.2684483528137207, 4
    368, 6206, 3.225700616836548, 3
    181, 1939, 3.42262601852417, 4
    384, 484, 3.298201084136963, 3
    473, 4062, 2.7763330936431885, 3
    537, 2729, 4.322381496429443, 4
    543, 84, 3.4742329120635986, 4
    599, 4791, 3.2956061363220215, 5
    118, 6563, 3.933187484741211, 4
    107, 24, 3.7393267154693604, 4
    473, 2951, 3.2223422527313232, 3
    304, 5161, 3.303772449493408, 2
    300, 315, 3.2576026916503906, 2
    42, 297, 4.416114807128906, 4
    413, 2071, 3.1936168670654297, 4
    231, 6540, 3.356034755706787, 4
    88, 6478, 3.1413450241088867, 5
    605, 901, 3.696383476257324, 3
    605, 899, 3.99597430229187, 3
    602, 992, 3.2150590419769287, 2
    218, 6613, 2.8454599380493164, 3
    479, 3047, 3.086394786834717, 3
    447, 2166, 2.5480263233184814, 3
    124, 3044, 3.470216751098633, 5
    318, 921, 4.459860801696777, 5
    286, 1134, 1.8221666812896729, 0
    609, 4471, 3.1900758743286133, 3
    507, 1485, 2.539710521697998, 4
    609, 2162, 3.050349235534668, 4
    473, 5728, 3.24099063873291, 4
    159, 1052, 2.966362237930298, 2
    533, 8460, 3.047417402267456, 4
    413, 3203, 3.2434322834014893, 4
    554, 1495, 3.6104135513305664, 3
    606, 1043, 4.0340704917907715, 1
    226, 4354, 4.160747051239014, 4
    181, 3135, 3.248884677886963, 4
    358, 877, 3.4632272720336914, 4
    62, 5856, 3.5191221237182617, 3
    414, 2257, 3.956230640411377, 4
    413, 2944, 3.428518772125244, 4
    217, 144, 2.26906156539917, 2
    427, 1389, 2.6693780422210693, 2
    225, 797, 3.1486568450927734, 3
    104, 9032, 3.4277658462524414, 4
    571, 123, 4.268906593322754, 4
    31, 815, 3.7007646560668945, 5
    506, 508, 3.3383703231811523, 3
    210, 7355, 3.5260934829711914, 4
    5, 428, 3.207174062728882, 4
    139, 307, 3.6447458267211914, 4
    609, 4918, 3.303978204727173, 5
    473, 4747, 2.8972222805023193, 3
    181, 833, 3.230524778366089, 4
    589, 1157, 3.4398903846740723, 3
    261, 181, 2.6666693687438965, 5
    176, 6664, 3.262468099594116, 3
    572, 7602, 3.6693673133850098, 5
    401, 84, 3.3920817375183105, 4
    557, 3741, 3.928715229034424, 3
    399, 254, 3.9201364517211914, 5
    58, 510, 4.817675590515137, 5
    435, 138, 3.478872776031494, 3
    11, 1044, 4.207103252410889, 5
    186, 86, 3.434180498123169, 3
    102, 7452, 3.3913121223449707, 5
    482, 914, 3.6572265625, 4
    421, 2189, 3.5376839637756348, 3
    572, 3454, 3.041977882385254, 4
    90, 3998, 2.9200730323791504, 4
    289, 733, 4.445725440979004, 5
    437, 4403, 3.0311238765716553, 0
    572, 1085, 3.434244155883789, 3
    571, 1494, 4.199337959289551, 5
    607, 159, 2.6134274005889893, 3
    181, 694, 3.9850924015045166, 5
    443, 311, 3.724867105484009, 5
    131, 921, 3.3139090538024902, 3
    67, 1082, 2.6496968269348145, 2
    225, 3212, 2.851090669631958, 3
    152, 7052, 2.029475688934326, 0
    561, 835, 3.969818592071533, 3
    473, 6371, 3.072256565093994, 3
    273, 349, 2.6662871837615967, 2
    99, 3018, 3.3923237323760986, 5
    248, 6913, 3.556291103363037, 3
    0, 2257, 4.641436576843262, 4
    473, 4408, 2.9995837211608887, 4
    63, 257, 3.8122665882110596, 4
    83, 840, 3.62203311920166, 4
    210, 6517, 3.4626543521881104, 4
    274, 1669, 3.7946574687957764, 4
    306, 5316, 1.8476455211639404, 1
    215, 2414, 3.6646251678466797, 4
    18, 706, 2.925807476043701, 3
    413, 4157, 2.8406224250793457, 4
    597, 8114, 3.394672155380249, 5
    18, 585, 2.681291341781616, 4
    231, 1052, 3.348876476287842, 3
    355, 6527, 3.767522096633911, 3
    576, 2117, 3.2792983055114746, 2
    476, 2096, 3.5665736198425293, 4
    479, 385, 2.9125475883483887, 0
    27, 134, 2.743051767349243, 3
    281, 3609, 3.728196620941162, 4
    609, 6028, 3.2384397983551025, 3
    27, 1322, 2.716989517211914, 3
    290, 7626, 3.879566192626953, 5
    562, 6465, 2.9093639850616455, 3
    289, 1615, 4.434089660644531, 4
    551, 2030, 2.822981357574463, 3
    317, 511, 3.5616073608398438, 3
    519, 819, 4.003741264343262, 4
    482, 4685, 2.9679996967315674, 4
    62, 899, 3.9390780925750732, 5
    413, 4238, 3.0019681453704834, 3
    200, 1403, 4.289920806884766, 5
    306, 3209, 2.4741575717926025, 2
    468, 1841, 3.118682384490967, 3
    426, 3852, 2.409788131713867, 4
    159, 910, 3.2825124263763428, 5
    277, 277, 3.9877305030822754, 5
    606, 507, 4.208710193634033, 4
    439, 2523, 3.809818983078003, 3
    598, 7154, 2.467327117919922, 3
    473, 3333, 3.560704469680786, 3
    27, 6046, 3.0248725414276123, 2
    367, 1475, 2.6981475353240967, 3
    248, 1157, 3.672797441482544, 4
    488, 2371, 2.9581878185272217, 4
    56, 1214, 2.9545373916625977, 2
    155, 1320, 3.243086814880371, 3
    356, 6329, 3.8987176418304443, 3
    324, 3639, 3.196134090423584, 4
    520, 33, 3.849437713623047, 4
    273, 1205, 2.8491809368133545, 3
    293, 2642, 2.876542091369629, 3
    201, 46, 4.480093955993652, 4
    598, 1770, 1.8857839107513428, 3
    220, 5485, 3.552133560180664, 4
    604, 720, 2.9143612384796143, 4
    460, 1321, 3.1744327545166016, 4
    555, 6045, 3.6998658180236816, 4
    216, 1609, 2.3922371864318848, 2
    404, 5512, 3.7487916946411133, 4
    607, 217, 2.906944751739502, 3
    602, 183, 3.2971439361572266, 5
    157, 285, 2.47891902923584, 4
    72, 2957, 3.1641688346862793, 4
    49, 3250, 2.710905075073242, 3
    3, 2326, 3.7050833702087402, 5
    598, 3804, 2.064509868621826, 3
    551, 2940, 2.918962240219116, 3
    519, 3428, 3.4657435417175293, 4
    523, 973, 3.2073757648468018, 5
    589, 138, 3.4282031059265137, 3
    26, 55, 3.707331418991089, 5
    386, 4235, 3.1704070568084717, 3
    441, 2884, 1.7734862565994263, 0
    108, 197, 2.844496965408325, 3
    265, 1374, 3.5676794052124023, 5
    216, 1672, 3.158477783203125, 3
    216, 2443, 2.5669775009155273, 3
    488, 690, 3.055232286453247, 3
    120, 55, 3.278912305831909, 5
    427, 508, 2.909437417984009, 2
    598, 3858, 1.8604050874710083, 0
    50, 3212, 3.2191600799560547, 3
    71, 277, 4.3061394691467285, 5
    68, 3555, 4.034235954284668, 3
    599, 6204, 2.6820321083068848, 4
    219, 3623, 3.8257699012756348, 2
    423, 1030, 3.6983628273010254, 4
    508, 5856, 2.9935123920440674, 4
    420, 6011, 3.548818588256836, 4
    215, 877, 3.810903310775757, 4
    176, 4824, 3.464550495147705, 2
    63, 830, 3.102780818939209, 4
    297, 2198, 2.588931083679199, 3
    602, 173, 2.9842355251312256, 4
    220, 92, 3.7516727447509766, 2
    473, 1321, 3.3339221477508545, 3
    308, 2335, 3.214325428009033, 3
    521, 5526, 3.4086363315582275, 4
    605, 3175, 3.460197925567627, 4
    166, 3645, 3.4359641075134277, 4
    306, 2912, 2.5541791915893555, 3
    18, 2805, 2.2502493858337402, 4
    600, 3633, 4.595130920410156, 5
    67, 1476, 2.611778497695923, 3
    602, 682, 3.4579007625579834, 4
    479, 2109, 3.591810941696167, 5
    603, 247, 3.1168148517608643, 4
    488, 472, 2.7723357677459717, 4
    375, 355, 3.553386688232422, 4
    164, 4090, 3.001971960067749, 3
    331, 5901, 3.524294376373291, 4
    67, 895, 3.336451768875122, 3
    118, 8935, 3.6929638385772705, 3
    513, 3189, 3.4929068088531494, 4
    598, 1550, 2.5293009281158447, 1
    483, 5895, 3.4838504791259766, 2
    262, 1585, 3.792302131652832, 3
    398, 2353, 3.7759904861450195, 4
    168, 822, 4.004936695098877, 4
    424, 915, 3.8348875045776367, 3
    433, 862, 3.8862595558166504, 5
    473, 4542, 3.159846067428589, 3
    607, 4228, 2.4805595874786377, 1
    379, 785, 4.347400665283203, 4
    0, 1430, 4.61092472076416, 5
    317, 7137, 3.442354440689087, 4
    63, 3189, 3.7726197242736816, 3
    579, 6221, 2.8550803661346436, 3
    607, 862, 3.5283701419830322, 3
    410, 313, 2.6501975059509277, 3
    413, 959, 3.6352882385253906, 4
    195, 6153, 3.0794670581817627, 4
    324, 1767, 3.604641914367676, 4
    598, 3916, 2.3723349571228027, 3
    411, 974, 4.4577765464782715, 5
    569, 1575, 3.4575037956237793, 4
    527, 7439, 3.07038950920105, 1
    189, 3192, 3.168861150741577, 3
    40, 8449, 2.9087629318237305, 3
    474, 6469, 3.7592029571533203, 3
    598, 2009, 2.2654531002044678, 1
    598, 5999, 2.7658751010894775, 3
    609, 4805, 3.0351250171661377, 3
    201, 900, 4.373269081115723, 4
    114, 1971, 3.480308771133423, 4
    437, 2224, 3.9725501537323, 3
    286, 3617, 2.737025737762451, 5
    306, 1749, 2.318995714187622, 3
    584, 1383, 3.9144482612609863, 4
    67, 2325, 2.704206705093384, 3
    424, 31, 3.8353431224823, 4
    116, 655, 3.7798337936401367, 3
    604, 6045, 3.275759696960449, 3
    413, 7181, 3.3679141998291016, 3
    67, 2601, 2.843777894973755, 2
    61, 8599, 3.554715394973755, 4
    331, 1996, 2.940113067626953, 4
    598, 6748, 2.0378429889678955, 2
    176, 2543, 3.0633864402770996, 4
    554, 2128, 3.647024631500244, 1
    598, 8673, 2.2362425327301025, 3
    63, 2903, 3.604372501373291, 4
    273, 2019, 2.9631288051605225, 2
    72, 8493, 3.355384349822998, 3
    473, 1809, 3.0233778953552246, 2
    393, 398, 3.684807777404785, 4
    447, 1294, 2.9029572010040283, 4
    120, 210, 3.5247466564178467, 3
    309, 376, 2.9686222076416016, 2
    131, 853, 2.7529191970825195, 3
    307, 313, 1.8816492557525635, 3
    447, 1151, 2.635450839996338, 2
    44, 3011, 3.9458367824554443, 4
    481, 899, 3.924391984939575, 4
    297, 899, 2.722872018814087, 4
    152, 6911, 2.0477893352508545, 2
    155, 2486, 3.672975540161133, 4
    482, 902, 3.4133336544036865, 5
    572, 2077, 4.112467288970947, 4
    323, 2896, 2.532698631286621, 2
    390, 138, 3.8914992809295654, 2
    603, 322, 3.9815845489501953, 4
    376, 4732, 3.844665050506592, 5
    216, 422, 2.535113573074341, 3
    306, 4792, 2.004918098449707, 2
    487, 1433, 3.891724109649658, 3
    469, 192, 3.3440535068511963, 3
    90, 197, 2.972585916519165, 3
    546, 1644, 3.90144681930542, 3
    588, 98, 4.114461421966553, 5
    410, 1, 2.933894395828247, 4
    49, 7422, 2.5674469470977783, 3
    442, 7749, 3.3964803218841553, 5
    20, 8997, 2.6821706295013428, 3
    598, 412, 2.5159640312194824, 2
    213, 1043, 3.109995126724243, 3
    372, 413, 3.8314218521118164, 5
    445, 444, 2.877164602279663, 3
    605, 1687, 3.2373909950256348, 3
    17, 8045, 3.4368107318878174, 4
    463, 6506, 3.2545394897460938, 4
    43, 592, 2.8644490242004395, 5
    325, 7101, 3.7196052074432373, 4
    526, 508, 4.099499225616455, 5
    27, 6833, 2.930402994155884, 3
    99, 509, 3.502506732940674, 4
    231, 6621, 3.3673510551452637, 4
    461, 9387, 2.975733995437622, 4
    604, 6641, 3.0797135829925537, 4
    349, 577, 3.443979501724243, 3
    248, 3152, 3.662562847137451, 3
    287, 2990, 2.7434616088867188, 3
    366, 2978, 3.679960250854492, 4
    248, 7946, 3.400662660598755, 4
    384, 952, 3.1212568283081055, 4
    1, 7137, 3.4637410640716553, 3
    246, 956, 3.681856155395508, 4
    513, 3067, 3.096466064453125, 1
    598, 2006, 2.3149189949035645, 2
    411, 706, 4.164595603942871, 5
    457, 506, 4.1415534019470215, 4
    166, 1597, 3.4581170082092285, 4
    317, 8863, 2.9591128826141357, 3
    287, 1084, 2.398510217666626, 2
    81, 2670, 3.3588616847991943, 4
    355, 1231, 3.6085524559020996, 4
    482, 6621, 3.5431675910949707, 4
    369, 4052, 3.174506664276123, 3
    559, 4080, 2.8817174434661865, 4
    201, 1052, 4.0058698654174805, 4
    445, 41, 3.0377755165100098, 3
    136, 2615, 3.798685073852539, 4
    609, 823, 3.5176901817321777, 4
    488, 4787, 3.0902366638183594, 0
    325, 944, 3.8861875534057617, 4
    351, 3136, 4.271726608276367, 5
    291, 2697, 2.9904534816741943, 3
    599, 2429, 2.482389450073242, 3
    176, 1101, 3.179500102996826, 2
    331, 176, 2.7170040607452393, 2
    27, 6676, 2.2669947147369385, 0
    569, 97, 3.6119577884674072, 3
    104, 6540, 4.047606468200684, 3
    476, 488, 3.085805654525757, 3
    289, 2550, 4.059112548828125, 4
    80, 506, 3.2202799320220947, 2
    110, 6191, 3.015084981918335, 4
    411, 1903, 3.952115535736084, 2
    273, 2962, 2.31058931350708, 2
    560, 1321, 3.3523716926574707, 3
    345, 3579, 3.4295425415039062, 4
    139, 823, 3.6343894004821777, 4
    569, 6329, 3.408439874649048, 4
    386, 3304, 2.904604911804199, 2
    199, 224, 4.190347671508789, 5
    598, 9369, 2.1034016609191895, 2
    232, 4926, 3.3994219303131104, 3
    168, 383, 3.7585482597351074, 4
    447, 2639, 2.982783079147339, 3
    413, 186, 3.413322925567627, 2
    231, 5370, 3.3344247341156006, 4
    366, 1153, 3.937772750854492, 1
    160, 6208, 3.200810670852661, 0
    487, 33, 3.816300392150879, 4
    598, 3442, 2.2878830432891846, 2
    504, 780, 3.577141284942627, 4
    142, 7022, 3.8054161071777344, 5
    18, 2766, 2.450704574584961, 2
    158, 315, 3.210660457611084, 2
    3, 510, 4.307682991027832, 5
    410, 309, 2.9460625648498535, 3
    437, 3650, 2.9290578365325928, 3
    143, 1521, 3.1558761596679688, 3
    518, 7629, 3.4825289249420166, 5
    194, 3759, 2.90946626663208, 3
    118, 3281, 3.739100456237793, 3
    427, 778, 2.248049736022949, 2
    353, 5955, 3.523305892944336, 3
    181, 1861, 3.218614101409912, 4
    361, 1223, 3.682617664337158, 4
    139, 989, 3.690568208694458, 3
    413, 815, 3.427629232406616, 4
    259, 4520, 3.4452991485595703, 4
    367, 1746, 2.6941399574279785, 3
    376, 7195, 3.7808749675750732, 4
    366, 694, 4.580768585205078, 5
    605, 4761, 3.2269949913024902, 2
    513, 8947, 3.1004066467285156, 2
    238, 1660, 3.8805840015411377, 4
    508, 3281, 2.5148961544036865, 3
    375, 134, 3.823909282684326, 5
    583, 44, 3.484950542449951, 5
    73, 5113, 4.1227006912231445, 3
    419, 31, 3.985163450241089, 3
    602, 2274, 3.4103777408599854, 5
    287, 2748, 2.9556875228881836, 1
    139, 512, 3.6834793090820312, 5
    297, 7646, 1.87978994846344, 0
    526, 1548, 3.7013063430786133, 3
    486, 8448, 3.495056390762329, 3
    379, 6288, 3.7112104892730713, 4
    75, 6317, 2.509444236755371, 4
    337, 5682, 3.2233834266662598, 4
    447, 784, 2.8000829219818115, 3
    461, 2010, 3.207667112350464, 4
    181, 938, 3.6118125915527344, 2
    166, 4155, 3.5784192085266113, 3
    371, 213, 3.2817904949188232, 4
    447, 8147, 2.8394100666046143, 2
    306, 2552, 2.4784507751464844, 4
    592, 4354, 3.3936715126037598, 3
    297, 5730, 2.088862895965576, 2
    602, 1688, 3.169825315475464, 3
    317, 6936, 3.417202949523926, 4
    273, 4416, 2.431018352508545, 1
    104, 7631, 3.9826536178588867, 4
    139, 3506, 3.274738073348999, 3
    513, 9277, 2.883016586303711, 2
    427, 827, 2.992560625076294, 4
    95, 622, 3.4797887802124023, 1
    45, 275, 3.6445717811584473, 4
    516, 3224, 2.370591402053833, 1
    0, 2144, 4.712414741516113, 5
    367, 1066, 3.2322187423706055, 5
    380, 1857, 3.3202297687530518, 3
    331, 6726, 3.2656333446502686, 3
    234, 176, 3.043076276779175, 1
    297, 8798, 2.282289981842041, 1
    473, 5125, 2.90069317817688, 3
    124, 4112, 3.875049591064453, 3
    554, 1605, 3.701559066772461, 2
    413, 1289, 3.288050651550293, 4
    447, 2348, 2.9027347564697266, 3
    283, 364, 3.0420148372650146, 3
    165, 4402, 3.4859187602996826, 4
    131, 4345, 2.7003049850463867, 3
    44, 1283, 4.179032802581787, 5
    368, 474, 3.4017138481140137, 2
    449, 2729, 4.391883850097656, 4
    413, 2977, 3.0831146240234375, 4
    248, 7276, 3.240948438644409, 3
    453, 5682, 3.7665328979492188, 3
    351, 314, 4.219925880432129, 5
    571, 985, 4.1950860023498535, 5
    181, 1180, 2.8954761028289795, 3
    146, 513, 3.144655227661133, 3
    427, 2246, 2.6392548084259033, 2
    179, 1938, 3.7581472396850586, 3
    344, 901, 3.8873486518859863, 4
    589, 511, 3.3700132369995117, 5
    437, 2653, 3.1827025413513184, 4
    56, 1796, 3.5816659927368164, 5
    449, 2504, 3.9185643196105957, 3
    360, 4355, 3.1106131076812744, 4
    119, 598, 2.5608668327331543, 3
    413, 4095, 3.3656582832336426, 4
    554, 1182, 3.649202346801758, 2
    325, 6854, 3.4765729904174805, 4
    211, 8008, 3.141566276550293, 4
    605, 5845, 3.376274347305298, 4
    427, 4617, 2.019709587097168, 2
    595, 6755, 3.604456663131714, 4
    576, 97, 3.560419797897339, 4
    181, 183, 3.2304739952087402, 4
    598, 1189, 2.2939062118530273, 3
    273, 6520, 3.013550281524658, 3
    595, 3623, 3.184635639190674, 3
    306, 2155, 2.7724990844726562, 3
    176, 7357, 2.965211868286133, 4
    155, 20, 3.510753870010376, 5
    18, 2802, 2.572737216949463, 3
    598, 2888, 2.301128625869751, 3
    447, 9206, 2.815650463104248, 3
    49, 2353, 2.691500425338745, 3
    372, 15, 3.7447588443756104, 4
    605, 3907, 3.045546054840088, 3
    139, 830, 3.1608214378356934, 3
    503, 3668, 3.7363033294677734, 4
    287, 5800, 3.4071907997131348, 2
    513, 546, 3.101882219314575, 4
    291, 8554, 3.3218061923980713, 4
    482, 5255, 3.000792980194092, 4
    513, 852, 3.1244120597839355, 4
    454, 297, 3.275020122528076, 3
    379, 963, 4.167156219482422, 5
    598, 800, 2.3376846313476562, 2
    95, 1796, 4.151745796203613, 4
    598, 5273, 1.9846909046173096, 2
    584, 372, 3.9789466857910156, 5
    600, 8045, 4.279729843139648, 4
    429, 3282, 3.1239702701568604, 3
    437, 2805, 2.8029708862304688, 2
    579, 1614, 2.9414710998535156, 2
    447, 5333, 2.651585340499878, 3
    579, 828, 3.3212485313415527, 4
    28, 899, 4.08806037902832, 4
    523, 398, 3.7952518463134766, 5
    609, 7122, 3.6470112800598145, 3
    317, 7870, 3.319420337677002, 4
    476, 3863, 3.414085865020752, 3
    65, 1403, 3.821488857269287, 4
    5, 58, 2.768089532852173, 3
    598, 8586, 2.0265212059020996, 2
    308, 1648, 3.2056713104248047, 5
    17, 7243, 2.9733784198760986, 4
    63, 3147, 3.552706003189087, 3
    181, 4335, 2.8713319301605225, 4
    317, 8865, 3.234083414077759, 4
    197, 1242, 3.807194232940674, 5
    436, 540, 3.5490825176239014, 2
    255, 7022, 3.8796067237854004, 4
    180, 112, 3.06860613822937, 3
    56, 176, 2.9293670654296875, 1
    218, 431, 2.94338321685791, 2
    464, 1320, 3.563084363937378, 4
    273, 3850, 2.6041579246520996, 3
    598, 2209, 2.5070009231567383, 2
    437, 4900, 4.012022018432617, 4
    231, 3211, 2.942903757095337, 2
    82, 4402, 2.997290849685669, 3
    116, 311, 3.717254638671875, 3
    138, 2995, 1.7417653799057007, 1
    114, 2824, 3.338146686553955, 4
    211, 8727, 3.365978240966797, 3
    287, 5365, 2.8916707038879395, 3
    134, 3633, 4.03110408782959, 5
    381, 8148, 2.9234156608581543, 3
    347, 685, 4.434025287628174, 5
    289, 930, 4.17892599105835, 4
    84, 1229, 3.242377996444702, 4
    74, 702, 3.1231610774993896, 3
    225, 1198, 3.0163111686706543, 4
    134, 2596, 3.349437713623047, 4
    491, 629, 3.4245190620422363, 3
    282, 56, 3.3915183544158936, 3
    139, 3393, 3.232297897338867, 4
    297, 5938, 2.446608781814575, 4
    609, 3234, 3.523754596710205, 4
    128, 6046, 3.9372329711914062, 4
    33, 5780, 2.785677909851074, 2
    569, 134, 3.206674575805664, 3
    67, 1744, 2.7634568214416504, 3
    413, 5904, 3.184128761291504, 2
    151, 6805, 3.134117603302002, 3
    104, 7987, 3.8381495475769043, 4
    279, 6335, 3.521328926086426, 3
    609, 8554, 3.3071084022521973, 4
    226, 314, 4.532090187072754, 5
    49, 1266, 2.8411366939544678, 2
    316, 224, 4.031095504760742, 4
    597, 898, 3.848748207092285, 2
    227, 1487, 3.3125524520874023, 4
    41, 2446, 3.3684868812561035, 2
    554, 1957, 3.6185414791107178, 3
    529, 253, 3.3362462520599365, 4
    465, 3228, 3.0542514324188232, 3
    351, 1708, 3.5340898036956787, 4
    67, 277, 3.442605495452881, 3
    278, 7967, 3.726372241973877, 3
    287, 6453, 2.5099525451660156, 3
    589, 2887, 3.2881247997283936, 3
    413, 6352, 3.1866753101348877, 3
    561, 2596, 3.7059760093688965, 4
    338, 6770, 3.430302143096924, 3
    104, 6930, 3.6000163555145264, 4
    468, 703, 3.677180290222168, 3
    44, 656, 3.405898094177246, 3
    20, 9424, 2.9762160778045654, 4
    499, 2306, 3.6076109409332275, 3
    110, 8926, 2.855774402618408, 2
    248, 1971, 3.328277587890625, 4
    532, 8710, 3.0944244861602783, 5
    215, 1701, 3.608687162399292, 4
    446, 412, 3.8223283290863037, 5
    447, 3352, 2.818220376968384, 3
    304, 5076, 3.636399984359741, 4
    176, 8271, 2.9568276405334473, 3
    22, 147, 3.2913267612457275, 3
    176, 6338, 3.1343913078308105, 4
    40, 43, 3.1681084632873535, 3
    479, 504, 3.0264878273010254, 4
    488, 2945, 2.784609317779541, 1
    479, 835, 3.2796478271484375, 5
    168, 3645, 3.796638250350952, 4
    483, 2992, 4.042876720428467, 3
    231, 6314, 3.6470401287078857, 4
    297, 3645, 2.2251546382904053, 0
    468, 1521, 3.299649477005005, 3
    350, 1287, 3.2736616134643555, 2
    57, 116, 3.7041962146759033, 3
    598, 3575, 2.057288408279419, 2
    599, 157, 2.4235761165618896, 2
    125, 512, 3.262968063354492, 4
    65, 3087, 3.2663097381591797, 4
    589, 4959, 3.0636839866638184, 2
    278, 956, 3.6195478439331055, 5
    248, 6293, 3.427899122238159, 4
    272, 31, 4.13620662689209, 5
    62, 914, 3.7284560203552246, 2
    593, 216, 3.670236110687256, 5
    12, 43, 3.8358726501464844, 5
    286, 5245, 2.1363377571105957, 4
    273, 6237, 2.7798547744750977, 3
    533, 8895, 3.3113861083984375, 2
    200, 1823, 3.9357500076293945, 4
    476, 4352, 3.4707720279693604, 4
    413, 3283, 3.2710683345794678, 4
    589, 1004, 3.0867414474487305, 3
    433, 615, 3.1362316608428955, 2
    452, 1947, 3.3540186882019043, 5
    186, 510, 4.228433609008789, 4
    602, 3213, 3.518342971801758, 4
    427, 509, 2.5659499168395996, 3
    27, 954, 2.5771090984344482, 4
    225, 4851, 3.1880149841308594, 4
    297, 6693, 2.683008909225464, 3
    9, 6813, 2.7254600524902344, 2
    494, 6090, 3.409989356994629, 2
    560, 6416, 3.181100845336914, 3
    273, 6712, 2.856401205062866, 4
    372, 308, 3.2700445652008057, 3
    176, 4903, 2.9545907974243164, 4
    238, 5915, 3.473227024078369, 3
    139, 2341, 3.5802290439605713, 4
    418, 461, 4.1383867263793945, 5
    158, 957, 3.270129680633545, 2
    231, 5963, 2.5913243293762207, 2
    229, 505, 2.7368717193603516, 2
    598, 956, 2.734076499938965, 3
    450, 630, 3.3780107498168945, 3
    384, 781, 3.178905963897705, 4
    452, 4057, 3.7076973915100098, 4
    474, 8886, 4.021728992462158, 4
    604, 511, 3.235285520553589, 3
    476, 325, 3.4834938049316406, 1
    41, 2632, 3.8828418254852295, 2
    19, 1272, 3.7863929271698, 4
    17, 6294, 3.4462454319000244, 4
    122, 897, 4.282511234283447, 4
    199, 1882, 4.076794147491455, 4
    376, 5103, 3.7267513275146484, 4
    456, 418, 4.020298004150391, 2
    479, 1724, 3.479053258895874, 1
    400, 9205, 3.1427130699157715, 4
    0, 2632, 4.545181751251221, 4
    413, 3065, 3.3092916011810303, 3
    351, 7355, 3.811587333679199, 5
    6, 5363, 3.422412633895874, 4
    17, 3635, 3.536102533340454, 4
    150, 598, 2.9839463233947754, 4
    181, 4091, 3.0101943016052246, 4
    216, 448, 2.700683832168579, 3
    88, 7823, 2.6787517070770264, 2
    273, 4903, 2.537102222442627, 2
    390, 130, 3.5726161003112793, 3
    579, 1486, 2.858919858932495, 3
    128, 2670, 3.844290256500244, 4
    110, 1051, 2.9988694190979004, 2
    381, 5920, 2.8687002658843994, 3
    607, 5915, 2.6427807807922363, 4
    317, 2546, 3.455148696899414, 4
    379, 933, 3.967256784439087, 5
    198, 6644, 2.9452216625213623, 3
    57, 18, 3.0872650146484375, 1
    307, 44, 2.138671875, 4
    379, 8354, 3.8813579082489014, 4
    418, 249, 3.6441078186035156, 1
    605, 971, 3.777942657470703, 4
    201, 2512, 3.822758436203003, 4
    491, 598, 3.14400053024292, 4
    410, 546, 3.152318239212036, 5
    273, 7289, 2.94425892829895, 4
    18, 1476, 2.626317024230957, 3
    465, 6241, 3.7292933464050293, 5
    379, 3557, 4.109985828399658, 4
    185, 1004, 4.050459861755371, 3
    609, 4786, 3.417332649230957, 3
    23, 3868, 3.462292194366455, 3
    308, 1531, 3.552419662475586, 3
    369, 1939, 3.331575632095337, 3
    297, 6726, 2.3391709327697754, 3
    463, 24, 3.225045680999756, 4
    181, 1661, 3.4739115238189697, 4
    367, 2804, 2.55905818939209, 3
    305, 4354, 3.366382122039795, 4
    186, 1927, 3.731982946395874, 4
    306, 2043, 1.8434669971466064, 1
    338, 5360, 3.633150339126587, 3
    452, 97, 4.173121929168701, 5
    473, 2979, 3.337919235229492, 4
    216, 2325, 2.95819091796875, 3
    252, 6581, 3.511281967163086, 4
    494, 809, 3.705455780029297, 5
    609, 899, 3.795114517211914, 5
    607, 30, 2.8508715629577637, 3
    437, 1208, 3.484703779220581, 3
    476, 1739, 3.2513439655303955, 3
    605, 980, 3.7931971549987793, 4
    579, 5200, 2.6706717014312744, 0
    88, 8830, 3.0911405086517334, 3
    550, 5922, 3.073561668395996, 4
    329, 1112, 3.297022581100464, 3
    297, 7266, 1.7818266153335571, 2
    63, 1123, 3.443373680114746, 4
    63, 24, 3.2049009799957275, 3
    88, 2377, 3.076133966445923, 4
    262, 2194, 3.8112103939056396, 3
    345, 2911, 3.7243337631225586, 3
    218, 3982, 2.4614760875701904, 1
    513, 7958, 3.171985626220703, 4
    386, 2533, 3.2031242847442627, 4
    176, 7627, 3.0349137783050537, 2
    566, 6405, 2.3630828857421875, 3
    304, 7467, 3.3268213272094727, 4
    447, 598, 2.4017434120178223, 3
    18, 2536, 2.2336745262145996, 3
    218, 2248, 3.172849655151367, 5
    577, 6670, 3.160553455352783, 4
    473, 6046, 3.483834743499756, 4
    273, 6174, 2.6193320751190186, 3
    169, 510, 3.9931225776672363, 4
    194, 2912, 3.3861236572265625, 4
    220, 3746, 3.709238052368164, 4
    273, 6202, 2.5301618576049805, 2
    67, 907, 3.1041722297668457, 3
    216, 2510, 2.871941089630127, 2
    29, 901, 3.9016647338867188, 3
    103, 5667, 3.169053316116333, 2
    519, 3827, 3.40669322013855, 2
    71, 910, 4.048263072967529, 4
    544, 1319, 2.79060959815979, 3
    413, 2970, 3.13692045211792, 1
    175, 592, 3.5596461296081543, 5
    8, 37, 3.2570741176605225, 3
    281, 3006, 3.746579170227051, 4
    343, 2941, 3.4600870609283447, 0
    123, 46, 4.092378616333008, 4
    579, 176, 2.6138954162597656, 4
    607, 613, 3.029550552368164, 3
    413, 1643, 3.058314323425293, 5
    220, 429, 3.642807722091675, 4
    110, 1542, 3.359834671020508, 3
    304, 9013, 3.42002534866333, 4
    450, 520, 3.9343037605285645, 5
    386, 429, 3.0337865352630615, 3
    598, 476, 2.410633087158203, 2
    366, 3101, 3.6963424682617188, 2
    293, 1285, 2.571669340133667, 2
    62, 5190, 3.39166522026062, 4
    67, 1397, 2.296354055404663, 3
    231, 815, 3.304091215133667, 4
    67, 6596, 2.868248224258423, 3
    151, 659, 4.220620155334473, 5
    447, 7041, 2.5998659133911133, 1
    598, 3539, 2.539505958557129, 4
    56, 133, 2.946256399154663, 1
    550, 7022, 3.4314613342285156, 4
    516, 785, 2.9261600971221924, 5
    231, 3700, 3.301450252532959, 3
    579, 2832, 3.2567460536956787, 4
    602, 2798, 2.9523520469665527, 4
    149, 2, 3.310851573944092, 3
    367, 601, 2.5211715698242188, 2
    273, 5836, 3.0859875679016113, 2
    243, 968, 3.7233924865722656, 5
    67, 6416, 2.7046778202056885, 4
    560, 4541, 3.356478691101074, 2
    609, 2884, 2.816953182220459, 3
    273, 6922, 3.454272747039795, 3
    18, 1527, 2.6603596210479736, 3
    306, 2797, 2.3452093601226807, 1
    3, 3470, 3.4781382083892822, 3
    433, 6294, 3.5093233585357666, 5
    479, 395, 3.313058853149414, 2
    345, 1615, 3.9849514961242676, 4
    317, 7357, 3.1548385620117188, 3
    295, 7355, 3.1582159996032715, 5
    103, 5420, 3.137040138244629, 3
    586, 785, 4.3631510734558105, 4
    287, 3554, 3.221175193786621, 4
    413, 3493, 2.7139272689819336, 3
    181, 987, 3.5700488090515137, 3
    265, 16, 3.612940549850464, 1
    229, 594, 2.605668067932129, 4
    90, 33, 3.5629220008850098, 3
    433, 257, 3.704307794570923, 5
    246, 520, 3.7634334564208984, 4
    210, 4631, 3.2768802642822266, 1
    474, 322, 4.751736640930176, 4
    41, 398, 4.212512493133545, 4
    473, 700, 3.215142250061035, 3
    218, 4522, 3.19154691696167, 4
    201, 1157, 4.011436462402344, 3
    287, 131, 2.747713565826416, 2
    602, 182, 2.9830827713012695, 4
    381, 257, 3.839146614074707, 5
    353, 6416, 3.4543638229370117, 3
    273, 1546, 2.765655040740967, 3
    593, 16, 3.7382705211639404, 4
    447, 4461, 2.729801654815674, 1
    533, 6870, 3.3224987983703613, 4
    386, 2708, 3.2269046306610107, 3
    563, 7058, 3.3283212184906006, 4
    309, 3898, 3.3363773822784424, 3
    67, 17, 2.7547950744628906, 2
    104, 7572, 3.7991631031036377, 4
    166, 3609, 3.746717929840088, 4
    435, 355, 3.089932441711426, 3
    602, 3481, 2.889854669570923, 4
    392, 7001, 3.8569347858428955, 5
    598, 4832, 2.1581814289093018, 2
    540, 134, 3.467867612838745, 5
    380, 3569, 3.490403413772583, 4
    379, 9649, 4.231509208679199, 4
    589, 950, 3.4489316940307617, 3
    317, 8141, 3.6066105365753174, 3
    107, 2802, 3.734663486480713, 3
    609, 9445, 3.5342438220977783, 5
    0, 789, 4.301226615905762, 5
    111, 8045, 3.716701030731201, 4
    73, 919, 4.435701370239258, 4
    194, 2301, 3.0997867584228516, 3
    605, 2404, 3.609802722930908, 4
    447, 9236, 2.5187408924102783, 2
    90, 476, 3.2703418731689453, 3
    17, 1163, 2.8818657398223877, 3
    523, 20, 3.124072551727295, 3
    356, 2248, 3.90151309967041, 3
    425, 7448, 3.7949719429016113, 5
    176, 3037, 2.939725160598755, 4
    490, 2432, 3.787738800048828, 4
    211, 1066, 3.8250813484191895, 3
    403, 235, 3.144477128982544, 3
    171, 1479, 3.0512168407440186, 4
    461, 2848, 2.940211772918701, 3
    265, 1291, 3.22598934173584, 3
    590, 2670, 3.2603402137756348, 5
    3, 2306, 3.8185155391693115, 4
    325, 7080, 2.9440393447875977, 5
    63, 6225, 3.494142532348633, 4
    380, 4136, 3.307293176651001, 4
    392, 8457, 3.740713357925415, 5
    293, 2789, 2.549715757369995, 4
    413, 3569, 3.3981082439422607, 2
    513, 1423, 3.2365336418151855, 2
    281, 2263, 3.5608749389648438, 4
    452, 1795, 3.6636481285095215, 2
    447, 3017, 2.6567673683166504, 3
    231, 7289, 3.2743330001831055, 4
    397, 7449, 4.038452625274658, 5
    88, 1153, 3.2490878105163574, 2
    488, 3428, 2.552849769592285, 3
    478, 783, 3.0512149333953857, 3
    579, 486, 3.4572768211364746, 2
    83, 650, 3.440791606903076, 5
    468, 1439, 3.301191568374634, 3
    345, 43, 3.7977423667907715, 3
    32, 6, 3.628082752227783, 1
    236, 254, 3.3973655700683594, 4
    218, 1403, 3.057800054550171, 2
    599, 3229, 2.4088034629821777, 3
    39, 522, 3.251692771911621, 4
    602, 3086, 3.146705150604248, 2
    413, 787, 3.1093413829803467, 2
    18, 1332, 2.3361570835113525, 3
    293, 2193, 2.942064046859741, 3
    65, 17, 3.742732048034668, 4
    589, 1403, 3.399296760559082, 2
    375, 835, 3.9432544708251953, 3
    139, 5719, 3.6831164360046387, 4
    0, 1558, 4.387678146362305, 4
    194, 2555, 3.2953953742980957, 4
    598, 7924, 2.4308600425720215, 4
    410, 260, 3.2577600479125977, 3
    592, 6488, 2.692167043685913, 2
    131, 1486, 2.5972862243652344, 3
    219, 4927, 3.6401305198669434, 3
    65, 1485, 3.566009044647217, 3
    90, 2940, 3.254788875579834, 2
    372, 313, 2.9709320068359375, 1
    412, 4858, 3.405665874481201, 5
    413, 961, 3.370948314666748, 4
    54, 5869, 2.7192232608795166, 4
    124, 257, 4.097076416015625, 4
    27, 2037, 2.8753890991210938, 1
    570, 856, 2.4911234378814697, 2
    543, 0, 4.044167518615723, 3
    605, 2341, 3.6643898487091064, 3
    65, 701, 3.9105048179626465, 3
    259, 2511, 3.7554514408111572, 2
    306, 3551, 1.9352850914001465, 2
    600, 7750, 4.453253746032715, 4
    598, 6753, 2.324761390686035, 3
    399, 913, 4.107892036437988, 4
    413, 3218, 3.18814754486084, 3
    488, 1082, 2.768247604370117, 2
    347, 948, 4.3365478515625, 4
    481, 5255, 3.0573363304138184, 4
    61, 2940, 3.6954500675201416, 4
    437, 1318, 2.9920010566711426, 2
    559, 9152, 2.7769105434417725, 4
    41, 322, 4.241342544555664, 4
    598, 6141, 2.241318941116333, 2
    447, 9258, 3.050039291381836, 2
    487, 3189, 3.962614059448242, 5
    541, 1182, 3.2447152137756348, 3
    273, 2376, 2.848376989364624, 3
    16, 898, 4.226184844970703, 3
    418, 295, 3.981316328048706, 0
    284, 8464, 3.1629951000213623, 3
    488, 7675, 2.994624137878418, 4
    566, 8421, 2.2157464027404785, 2
    336, 68, 3.623401165008545, 5
    447, 9143, 2.7557075023651123, 3
    314, 907, 3.8274683952331543, 5
    313, 383, 2.873040199279785, 3
    462, 1622, 3.534200668334961, 3
    329, 4153, 3.316387176513672, 4
    218, 6276, 2.9296188354492188, 3
    94, 1401, 3.9931581020355225, 4
    45, 176, 3.3311047554016113, 3
    605, 1499, 3.738400459289551, 3
    65, 5294, 3.749088764190674, 4
    386, 4996, 3.13264536857605, 4
    473, 585, 3.1247265338897705, 4
    134, 1813, 3.6942920684814453, 3
    100, 2410, 3.4556636810302734, 4
    67, 6620, 2.585158109664917, 3
    218, 6780, 3.132111072540283, 4
    599, 6225, 3.0011069774627686, 4
    554, 963, 4.056629180908203, 4
    540, 18, 2.857360601425171, 2
    437, 5895, 3.089015007019043, 2
    560, 334, 3.099238634109497, 3
    50, 4492, 3.6777446269989014, 4
    42, 238, 4.187036991119385, 5
    386, 3677, 3.3890323638916016, 4
    90, 1996, 3.081167697906494, 3
    324, 1712, 3.2848989963531494, 3
    607, 1590, 2.9171700477600098, 2
    110, 3926, 3.1443722248077393, 3
    304, 5721, 3.9441237449645996, 5
    203, 2034, 3.3033745288848877, 5
    605, 1280, 3.300090789794922, 4
    563, 7026, 3.3166956901550293, 3
    168, 898, 4.312121868133545, 4
    313, 116, 2.9475784301757812, 3
    306, 5386, 2.2086594104766846, 2
    554, 358, 3.5621156692504883, 3
    102, 1938, 3.9719443321228027, 4
    519, 1971, 3.561069965362549, 5
    148, 1556, 2.719928503036499, 3
    576, 897, 3.564069986343384, 3
    447, 7913, 3.1242575645446777, 2
    408, 715, 3.5973377227783203, 2
    255, 8620, 3.600160598754883, 4
    494, 8572, 3.3432188034057617, 3
    289, 257, 4.372311592102051, 5
    211, 6659, 3.9404640197753906, 3
    566, 4145, 2.1454391479492188, 2
    201, 2440, 4.050546646118164, 4
    225, 3358, 3.094381809234619, 5
    367, 2560, 2.119983434677124, 3
    316, 7629, 3.5167174339294434, 3
    304, 3152, 3.746601104736328, 4
    379, 2729, 4.057468891143799, 5
    265, 486, 3.9104886054992676, 5
    62, 1157, 3.601489543914795, 3
    194, 224, 3.7474021911621094, 4
    468, 2739, 3.694399356842041, 5
    44, 1495, 3.6637754440307617, 3
    5, 177, 3.265516757965088, 3
    479, 6175, 3.783844232559204, 3
    148, 835, 3.166079044342041, 4
    303, 1180, 3.2483344078063965, 4
    605, 510, 4.228128910064697, 4
    27, 5915, 2.465601682662964, 3
    599, 1596, 2.6932320594787598, 4
    73, 1484, 4.179671287536621, 4
    381, 2998, 3.3365554809570312, 3
    331, 5379, 3.1456010341644287, 2
    488, 2013, 2.5447304248809814, 1
    338, 8277, 3.505308151245117, 3
    67, 7078, 2.49033784866333, 2
    274, 2006, 3.963858127593994, 5
    379, 9177, 3.0949954986572266, 4
    437, 2604, 2.986792802810669, 3
    607, 919, 3.212775945663452, 3
    201, 857, 3.893099546432495, 3
    413, 3938, 3.348254442214966, 3
    598, 3175, 2.394904136657715, 2
    559, 6780, 3.3702638149261475, 4
    230, 793, 3.777160167694092, 5
    303, 1004, 3.464179754257202, 4
    56, 757, 3.750540256500244, 4
    389, 2301, 3.2430925369262695, 4
    228, 472, 3.4776980876922607, 4
    246, 8354, 3.470923662185669, 3
    447, 2956, 2.671941041946411, 3
    461, 9053, 3.344841957092285, 4
    216, 1170, 2.7377686500549316, 3
    489, 1321, 3.0432660579681396, 3
    413, 2911, 3.575223445892334, 4
    424, 910, 3.8815526962280273, 3
    468, 1059, 3.285839557647705, 4
    605, 1162, 3.204432249069214, 4
    327, 902, 3.067500114440918, 2
    468, 1946, 3.436284303665161, 3
    272, 506, 3.908698797225952, 5
    350, 4170, 3.9492998123168945, 4
    497, 418, 4.393194198608398, 4
    607, 2009, 2.7226972579956055, 4
    176, 551, 3.20263409614563, 3
    482, 7367, 2.9499258995056152, 4
    231, 6976, 3.130951404571533, 4
    17, 4999, 3.658637046813965, 3
    424, 1217, 3.7668511867523193, 3
    464, 2441, 3.5344271659851074, 4
    487, 1073, 3.9767422676086426, 2
    225, 520, 3.5772783756256104, 4
    76, 3136, 4.3126020431518555, 2
    321, 1403, 3.0183088779449463, 4
    114, 99, 3.48040771484375, 2
    306, 1029, 2.140071392059326, 2
    494, 5780, 3.213865041732788, 2
    110, 3673, 2.9648637771606445, 1
    90, 1986, 3.176996946334839, 3
    303, 1978, 3.398362398147583, 5
    272, 508, 4.081172466278076, 2
    338, 9422, 3.640889883041382, 4
    483, 61, 3.5598063468933105, 4
    492, 2670, 3.5475776195526123, 3
    464, 711, 4.229039669036865, 5
    72, 8457, 3.5819358825683594, 5
    291, 7749, 3.050570249557495, 4
    263, 5880, 3.6455862522125244, 4
    111, 510, 4.1947808265686035, 4
    287, 960, 3.256887197494507, 4
    554, 2721, 3.4247994422912598, 3
    395, 974, 3.3679466247558594, 5
    2, 1552, 3.126713514328003, 0
    413, 3780, 3.1630709171295166, 4
    607, 1242, 3.151564121246338, 5
    110, 8619, 3.2833027839660645, 4
    607, 5212, 2.9871914386749268, 3
    602, 2078, 3.696061134338379, 2
    487, 733, 4.075675010681152, 4
    447, 3167, 2.690117835998535, 1
    598, 9586, 2.3122568130493164, 3
    566, 8212, 2.02903413772583, 0
    421, 277, 4.073478698730469, 5
    407, 6671, 3.6285884380340576, 3
    274, 2565, 4.057145118713379, 5
    428, 137, 3.8236823081970215, 3
    455, 113, 3.041637897491455, 3
    4, 32, 3.6315245628356934, 4
    17, 6572, 3.0584824085235596, 4
    286, 2284, 2.0812759399414062, 4
    306, 1026, 2.5044703483581543, 4
    188, 277, 4.2588090896606445, 4
    121, 1, 4.1405839920043945, 4
    90, 1527, 3.2244760990142822, 2
    598, 2040, 1.9673455953598022, 1
    218, 1400, 2.5894367694854736, 2
    231, 2194, 3.3532729148864746, 3
    181, 626, 3.305602550506592, 2
    609, 9372, 2.6555678844451904, 3
    273, 1034, 2.9874207973480225, 3
    143, 257, 3.886594295501709, 4
    607, 79, 2.8600316047668457, 2
    139, 929, 3.608224868774414, 2
    155, 1066, 4.075873851776123, 3
    390, 2451, 4.079041481018066, 5
    104, 7545, 3.621978282928467, 3
    104, 6755, 4.355470180511475, 5
    398, 3563, 3.8714258670806885, 4
    316, 7398, 3.6372995376586914, 4
    166, 914, 3.72305965423584, 2
    146, 192, 2.8583433628082275, 5
    523, 337, 3.120173931121826, 4
    281, 921, 3.9444773197174072, 5
    473, 6259, 3.2704687118530273, 3
    598, 1914, 2.5463855266571045, 2
    228, 398, 4.120227813720703, 5
    265, 2192, 3.622663974761963, 5
    513, 986, 3.234764814376831, 4
    605, 3236, 3.126408576965332, 3
    598, 2882, 2.1533360481262207, 1
    163, 3275, 3.4379570484161377, 5
    554, 382, 3.4634790420532227, 4
    423, 5988, 3.367947578430176, 4
    413, 3126, 3.2414207458496094, 3
    18, 1681, 2.6525514125823975, 2
    531, 1767, 4.213341236114502, 4
    516, 906, 2.9886999130249023, 0
    447, 3136, 3.632192850112915, 4
    356, 819, 3.925208806991577, 5
    5, 510, 4.047334671020508, 4
    604, 3563, 3.334563732147217, 3
    383, 828, 3.410764217376709, 4
    331, 7675, 3.313673496246338, 2
    200, 1186, 4.125923156738281, 5
    424, 509, 3.436821460723877, 3
    273, 6631, 3.2200703620910645, 3
    480, 1252, 2.7938170433044434, 2
    452, 3146, 3.6958744525909424, 4
    413, 319, 3.062880039215088, 2
    386, 702, 3.251004457473755, 3
    488, 6757, 2.544374465942383, 3
    63, 3926, 3.239762306213379, 3
    306, 6935, 2.4712390899658203, 2
    609, 6693, 3.755251407623291, 4
    201, 838, 3.587676525115967, 3
    366, 123, 4.201704025268555, 4
    299, 1733, 3.663808822631836, 4
    49, 819, 2.8556861877441406, 3
    516, 1182, 2.3384885787963867, 3
    605, 6388, 3.9490935802459717, 4
    599, 4930, 2.720889091491699, 3
    386, 3430, 3.213125228881836, 1
    560, 7967, 3.5983777046203613, 4
    316, 5294, 3.5207419395446777, 5
    477, 308, 2.424358367919922, 4
    18, 1392, 2.382439374923706, 2
    476, 904, 3.8091464042663574, 4
    461, 9444, 3.2368016242980957, 5
    131, 815, 2.896651268005371, 3
    246, 5156, 3.717593193054199, 3
    176, 3282, 3.0548081398010254, 4
    479, 249, 3.1909916400909424, 2
    27, 6726, 2.8270444869995117, 4
    602, 2384, 3.4932851791381836, 4
    599, 428, 2.7527318000793457, 3
    353, 5927, 3.419905185699463, 4
    225, 6236, 3.071079969406128, 2
    47, 2696, 3.3604869842529297, 2
    476, 418, 3.6609013080596924, 4
    461, 8451, 3.106287956237793, 4
    262, 1145, 3.6110541820526123, 3
    478, 1413, 3.682910919189453, 5
    411, 2041, 3.270941734313965, 2
    63, 4256, 3.0303616523742676, 4
    609, 6613, 3.204591751098633, 4
    209, 8133, 3.574158191680908, 4
    356, 898, 4.182412147521973, 4
    381, 396, 2.9690425395965576, 3
    519, 1095, 3.4984302520751953, 3
    547, 7245, 3.7907776832580566, 3
    573, 378, 3.3810300827026367, 4
    473, 1327, 3.5047836303710938, 4
    598, 7553, 2.3110744953155518, 3
    106, 376, 2.984851837158203, 4
    23, 7001, 3.535186767578125, 4
    533, 6712, 3.5183539390563965, 3
    44, 3742, 3.6365628242492676, 2
    210, 7338, 3.6961073875427246, 4
    27, 1297, 2.9524309635162354, 4
    223, 1627, 4.041684150695801, 5
    181, 3186, 3.0257081985473633, 4
    18, 2798, 2.3098955154418945, 2
    220, 1670, 3.979912042617798, 4
    132, 472, 2.7182087898254395, 3
    273, 779, 2.3704869747161865, 3
    488, 919, 3.1551363468170166, 3
    104, 3554, 3.9863529205322266, 5
    609, 5716, 3.2335333824157715, 3
    541, 1242, 3.537519693374634, 4
    482, 3133, 3.2457435131073, 4
    579, 867, 3.1083149909973145, 3
    139, 3283, 3.367311954498291, 3
    381, 2216, 3.519808769226074, 4
    86, 1483, 3.559983253479004, 4
    55, 287, 3.815577507019043, 3
    473, 4328, 3.400287389755249, 5
    364, 8413, 2.2079274654388428, 4
    558, 308, 3.1192500591278076, 3
    607, 508, 3.2596757411956787, 3
    449, 690, 4.348257064819336, 4
    423, 3520, 3.3228909969329834, 3
    553, 1311, 3.39467716217041, 2
    476, 6529, 3.212555170059204, 4
    582, 7637, 3.1284713745117188, 3
    539, 1636, 3.3219645023345947, 2
    218, 507, 3.3292593955993652, 4
    41, 1242, 3.891571283340454, 5
    609, 6512, 3.241940498352051, 4
    598, 2144, 2.852823257446289, 5
    379, 2040, 3.325559616088867, 3
    390, 989, 4.019529819488525, 4
    200, 1502, 4.7381134033203125, 5
    181, 1403, 3.4238767623901367, 3
    607, 1502, 3.457934856414795, 4
    20, 9165, 2.6808218955993652, 1
    502, 418, 3.6110599040985107, 3
    596, 1671, 4.039407253265381, 3
    158, 5217, 2.80159592628479, 4
    379, 4522, 4.044458866119385, 5
    176, 8329, 2.92323899269104, 4
    413, 2386, 3.4692015647888184, 4
    270, 2231, 3.264968156814575, 2
    218, 328, 2.645024299621582, 3
    306, 3875, 2.0087313652038574, 1
    304, 7236, 3.407071352005005, 4
    231, 6444, 3.209392547607422, 3
    56, 2083, 3.0167534351348877, 4
    99, 1434, 3.446486711502075, 5
    453, 3557, 3.599677085876465, 3
    297, 7330, 1.8721635341644287, 3
    216, 762, 2.772338390350342, 3
    104, 705, 4.1661787033081055, 4
    220, 905, 4.167701244354248, 3
    238, 2013, 3.432816505432129, 3
    50, 3563, 3.7521870136260986, 4
    287, 6017, 3.4067845344543457, 1
    508, 3867, 3.156020164489746, 3
    384, 594, 3.401543378829956, 4
    353, 138, 3.6121442317962646, 4
    444, 6204, 3.25138258934021, 3
    283, 484, 3.255342960357666, 3
    473, 1865, 2.975034475326538, 3
    366, 701, 4.1085686683654785, 5
    41, 811, 3.59350323677063, 4
    6, 3609, 3.30729079246521, 4
    274, 2072, 3.9613189697265625, 4
    27, 398, 3.2953262329101562, 3
    281, 3282, 3.1905219554901123, 3
    273, 3136, 3.5647478103637695, 4
    426, 4341, 3.0085299015045166, 1
    494, 8278, 3.5674080848693848, 0
    451, 3554, 4.507073402404785, 4
    505, 6631, 3.680082082748413, 4
    57, 225, 4.012945175170898, 4
    82, 97, 3.3183815479278564, 4
    566, 7751, 1.8600763082504272, 2
    513, 9618, 2.7905077934265137, 4
    482, 6629, 3.3306021690368652, 4
    248, 9415, 3.5735111236572266, 5
    262, 5291, 3.861833095550537, 3
    413, 8141, 3.4531073570251465, 3
    83, 852, 3.461108922958374, 5
    225, 1938, 3.685222625732422, 4
    572, 6753, 3.477193832397461, 4
    29, 7909, 3.552511215209961, 5
    231, 599, 3.2765631675720215, 3
    379, 6022, 3.5222790241241455, 4
    118, 314, 4.788661479949951, 4
    437, 3212, 2.937779664993286, 3
    245, 7004, 3.711073398590088, 5
    473, 2158, 2.8164665699005127, 3
    248, 8676, 3.481365919113159, 4
    379, 6112, 3.719453811645508, 3
    245, 7043, 4.069773197174072, 5
    5, 49, 3.1438779830932617, 4
    607, 2505, 2.625060558319092, 3
    524, 8013, 2.8070473670959473, 3
    306, 4537, 2.370400905609131, 2
    513, 5038, 2.900113582611084, 3
    155, 474, 4.050806045532227, 4
    452, 418, 4.003854751586914, 4
    591, 136, 3.637862205505371, 5
    367, 142, 2.532149314880371, 2
    605, 1474, 3.3473551273345947, 2
    199, 2044, 3.159121513366699, 4
    176, 53, 3.019549608230591, 3
    219, 7001, 3.9816579818725586, 5
    287, 1518, 3.232247829437256, 2
    253, 5891, 3.654249668121338, 4
    413, 4311, 3.1304433345794678, 4
    297, 6164, 1.6393170356750488, 2
    201, 2290, 3.8955583572387695, 4
    103, 1730, 3.554551601409912, 3
    72, 8706, 3.6253821849823, 2
    380, 4131, 3.8462836742401123, 4
    248, 9586, 3.3919625282287598, 4
    509, 2194, 2.953620672225952, 3
    437, 184, 3.2116281986236572, 3
    540, 510, 4.167716026306152, 4
    605, 1485, 3.362311840057373, 3
    528, 0, 3.4516143798828125, 3
    385, 40, 2.446262836456299, 2
    554, 657, 3.0915520191192627, 3
    392, 8407, 3.8019609451293945, 5
    293, 40, 2.3739919662475586, 3
    278, 1802, 3.548830270767212, 4
    58, 961, 4.140899658203125, 5
    571, 2316, 3.9555697441101074, 4
    500, 2, 3.20224666595459, 5
    112, 691, 3.81827712059021, 4
    219, 32, 3.9109630584716797, 4
    110, 4124, 3.0259146690368652, 3
    559, 1823, 2.941782236099243, 3
    479, 1556, 2.8334975242614746, 3
    88, 6214, 3.04443359375, 4
    44, 618, 3.6878576278686523, 3
    468, 2001, 3.5352256298065186, 3
    598, 7905, 1.7104648351669312, 4
    164, 3900, 3.0948855876922607, 3
    273, 6217, 2.235107898712158, 1
    248, 1374, 3.5786283016204834, 4
    431, 5217, 2.8430685997009277, 5
    414, 1913, 3.236541509628296, 4
    0, 1575, 4.573407173156738, 5
    228, 97, 3.973236560821533, 5
    218, 2492, 3.0759735107421875, 3
    541, 4607, 3.5697760581970215, 2
    155, 2339, 3.245408535003662, 5
    447, 1492, 2.7449615001678467, 4
    413, 6210, 3.0247552394866943, 3
    482, 7784, 3.5826919078826904, 4
    576, 2386, 3.2764620780944824, 2
    609, 2214, 3.5686583518981934, 4
    482, 7287, 2.7997186183929443, 4
    221, 5212, 3.118685245513916, 4
    476, 907, 3.785017251968384, 4
    387, 4644, 3.162881374359131, 2
    317, 7354, 3.5750083923339844, 3
    165, 895, 3.9941375255584717, 4
    447, 1819, 2.9962804317474365, 2
    9, 7849, 3.1106553077697754, 4
    2, 1823, 3.035250186920166, 0
    17, 7545, 2.9211106300354004, 3
    568, 156, 3.147489547729492, 4
    607, 5155, 2.983489513397217, 4
    279, 1182, 3.7298343181610107, 3
    376, 7604, 3.669565200805664, 4
    287, 1236, 2.9062306880950928, 3
    346, 31, 3.744939088821411, 3
    451, 3086, 4.182300090789795, 4
    19, 483, 3.9126062393188477, 5
    5, 225, 3.662659168243408, 4
    240, 659, 3.968411445617676, 4
    416, 6993, 4.442749977111816, 5
    222, 520, 3.3277461528778076, 3
    370, 956, 4.243287563323975, 5
    317, 9616, 3.0387794971466064, 5
    609, 8960, 3.391007900238037, 3
    389, 4900, 4.063965320587158, 5
    366, 999, 4.088320732116699, 5
    551, 485, 3.152001142501831, 1
    324, 898, 3.85316801071167, 3
    324, 1009, 3.336564779281616, 4
    104, 5519, 3.78671932220459, 5
    380, 2996, 3.237020492553711, 4
    307, 253, 2.304131507873535, 1
    497, 509, 4.153135776519775, 3
    412, 3002, 3.59657621383667, 4
    607, 253, 2.8361082077026367, 2
    5, 184, 3.2478137016296387, 4
    304, 5834, 3.508208990097046, 5
    482, 375, 3.448917865753174, 4
    51, 5968, 4.169843673706055, 5
    199, 2998, 3.5864791870117188, 4
    601, 334, 3.069733142852783, 4
    81, 4460, 2.958096504211426, 3
    248, 906, 4.131432056427002, 4
    426, 3609, 3.429732322692871, 4
    50, 62, 3.5783333778381348, 5
    282, 520, 3.632183313369751, 4
    440, 1978, 3.79766845703125, 4
    324, 2918, 3.1879916191101074, 4
    605, 1078, 3.531663656234741, 4
    473, 5792, 2.88472580909729, 4
    329, 2077, 3.7239441871643066, 4
    209, 1043, 3.861342191696167, 4
    62, 7043, 3.6292576789855957, 4
    159, 217, 2.828537702560425, 3
    28, 6411, 3.504331588745117, 4
    452, 1401, 3.952399253845215, 4
    437, 2886, 3.0731253623962402, 4
    386, 6317, 2.775974750518799, 3
    510, 8727, 3.8358817100524902, 4
    185, 732, 4.349783897399902, 4
    44, 436, 3.570822238922119, 4
    386, 2216, 3.231945514678955, 4
    6, 5212, 3.0935213565826416, 4
    234, 134, 3.5077128410339355, 3
    516, 933, 2.546015977859497, 2
    524, 1, 3.048712968826294, 3
    536, 4421, 4.000395774841309, 5
    413, 2210, 3.515580654144287, 4
    380, 897, 3.849104404449463, 3
    219, 6726, 3.8900232315063477, 4
    553, 694, 4.415650367736816, 5
    233, 1284, 3.473680019378662, 1
    609, 7179, 3.7214722633361816, 3
    139, 263, 3.0270376205444336, 3
    582, 512, 3.432319402694702, 4
    413, 1398, 3.0666146278381348, 2
    607, 910, 3.36091947555542, 4
    593, 4934, 3.69107723236084, 4
    20, 7827, 3.1213507652282715, 3
    181, 6138, 3.547621726989746, 3
    542, 951, 3.9780967235565186, 5
    104, 5955, 3.9454894065856934, 3
    598, 249, 2.493675708770752, 3
    282, 1283, 3.6945137977600098, 5
    211, 5856, 3.4266042709350586, 4
    561, 1072, 3.811084747314453, 3
    263, 1008, 3.374223470687866, 2
    150, 630, 3.3402857780456543, 5
    517, 618, 2.6662464141845703, 3
    607, 1176, 2.849421501159668, 4
    389, 4787, 3.5908889770507812, 4
    176, 7687, 3.081686496734619, 3
    386, 2449, 2.8040874004364014, 1
    427, 145, 1.7123925685882568, 3
    327, 1733, 3.4922800064086914, 3
    128, 2019, 3.7242770195007324, 4
    173, 326, 3.8583767414093018, 4
    609, 5084, 3.1584997177124023, 4
    287, 5916, 3.209259510040283, 3
    376, 2512, 3.7317676544189453, 5
    226, 1938, 4.374018669128418, 4
    231, 5785, 2.6754302978515625, 3
    327, 1389, 3.1537067890167236, 3
    602, 2354, 2.880765199661255, 4
    139, 331, 3.6346659660339355, 3
    123, 277, 4.19218111038208, 4
    542, 506, 4.15761661529541, 5
    316, 4607, 3.7672202587127686, 5
    473, 5034, 2.6948978900909424, 3
    199, 1744, 3.6822993755340576, 4
    531, 277, 4.6749725341796875, 4
    553, 583, 4.087933540344238, 5
    566, 9227, 1.7055835723876953, 3
    266, 2592, 4.154951095581055, 5
    5, 381, 2.846406936645508, 3
    176, 7723, 3.1857123374938965, 4
    609, 4584, 3.2826757431030273, 4
    607, 5151, 2.4798121452331543, 4
    291, 1805, 3.3727211952209473, 3
    159, 324, 2.8357982635498047, 1
    597, 8785, 3.149818181991577, 5
    380, 4007, 3.621877670288086, 3
    609, 7271, 3.150512933731079, 3
    239, 197, 3.5903642177581787, 4
    201, 1210, 4.159513473510742, 4
    139, 5869, 3.4500439167022705, 4
    273, 2964, 2.5609183311462402, 3
    262, 314, 4.300958633422852, 4
    602, 1907, 2.7649223804473877, 4
    473, 2333, 3.244115114212036, 3
    17, 950, 3.3541886806488037, 4
    445, 123, 3.3826167583465576, 3
    384, 706, 3.6857852935791016, 4
    371, 35, 3.1123228073120117, 3
    338, 3228, 3.1874213218688965, 3
    353, 224, 4.021191120147705, 4
    236, 1188, 2.5889787673950195, 3
    39, 509, 3.5473663806915283, 4
    566, 9367, 1.974738359451294, 3
    65, 3938, 3.7323555946350098, 4
    292, 0, 3.0470917224884033, 3
    67, 1182, 2.6825690269470215, 4
    336, 229, 3.921700954437256, 5
    379, 5320, 3.774801254272461, 3
    289, 823, 4.136394023895264, 5
    275, 534, 4.226930618286133, 3
    379, 7159, 3.4351840019226074, 4
    44, 1970, 3.227311134338379, 4
    92, 378, 3.8721508979797363, 5
    413, 3565, 3.4110970497131348, 4
    193, 1242, 3.5355477333068848, 3
    297, 942, 2.4403984546661377, 3
    413, 3512, 2.8446035385131836, 3
    603, 31, 3.79495906829834, 4
    333, 5938, 3.2987780570983887, 4
    324, 3765, 2.9705376625061035, 4
    274, 1001, 4.269167900085449, 5
    473, 4695, 3.2749624252319336, 3
    131, 6209, 2.737304449081421, 3
    144, 126, 2.8811581134796143, 3
    482, 6726, 3.484147548675537, 4
    216, 2578, 3.1089119911193848, 1
    88, 4488, 3.113109827041626, 1
    56, 2611, 2.770083427429199, 4
    221, 6075, 2.4128453731536865, 2
    482, 2324, 3.3506975173950195, 4
    345, 4926, 3.8305892944335938, 4
    596, 1815, 4.0925068855285645, 4
    35, 1644, 3.1642825603485107, 2
    307, 7784, 2.570791244506836, 1
    6, 5365, 2.7674994468688965, 1
    297, 615, 2.1132895946502686, 2
    413, 3983, 2.8899800777435303, 4
    67, 1548, 2.6852924823760986, 3
    488, 682, 2.919456958770752, 3
    523, 592, 3.13242244720459, 4
    291, 1075, 3.229660749435425, 3
    247, 4345, 3.359095335006714, 3
    168, 826, 3.6795079708099365, 4
    280, 2021, 2.6891579627990723, 3
    510, 7723, 3.7227296829223633, 4
    56, 1444, 3.5167245864868164, 4
    181, 2887, 3.3127048015594482, 3
    49, 9454, 2.4971773624420166, 3
    103, 1473, 3.431044101715088, 0
    120, 508, 3.5589599609375, 4
    91, 773, 2.8728086948394775, 4
    263, 792, 3.9883103370666504, 5
    111, 378, 3.2460520267486572, 0
    155, 1639, 3.621411085128784, 4
    67, 5363, 3.139892339706421, 3
    218, 1099, 2.9386045932769775, 3
    5, 166, 2.9001221656799316, 3
    473, 1021, 3.25801682472229, 4
    8, 4135, 2.941134452819824, 4
    380, 4345, 3.323577880859375, 4
    595, 1332, 2.7424745559692383, 3
    99, 1562, 3.3745696544647217, 3
    507, 968, 2.969917058944702, 4
    67, 4221, 2.449594020843506, 5
    49, 4819, 2.701706886291504, 3
    139, 3803, 2.9783263206481934, 2
    248, 520, 3.8953592777252197, 4
    447, 4422, 2.7692575454711914, 1
    598, 8340, 2.1190905570983887, 2
    609, 2909, 3.443528652191162, 2
    155, 325, 3.688267230987549, 3
    371, 920, 3.3522775173187256, 3
    364, 7516, 2.3358306884765625, 4
    602, 2312, 3.4122371673583984, 4
    110, 6061, 3.028411626815796, 3
    5, 3, 3.251974582672119, 3
    65, 1314, 3.2933316230773926, 4
    454, 10, 3.5059313774108887, 4
    469, 43, 3.796267509460449, 3
    331, 6613, 3.058811664581299, 3
    386, 2475, 3.122783660888672, 3
    274, 3575, 3.7062275409698486, 4
    607, 3428, 2.6104896068573, 3
    353, 6298, 3.9657692909240723, 4
    262, 463, 3.8945038318634033, 4
    479, 3899, 2.970951795578003, 3
    367, 1748, 3.037022352218628, 3
    465, 5213, 3.1772384643554688, 4
    304, 740, 3.5563478469848633, 4
    560, 2113, 2.9145426750183105, 3
    97, 8565, 3.4541168212890625, 4
    263, 1082, 3.614163398742676, 3
    524, 6520, 3.390907049179077, 3
    464, 2729, 4.185924530029297, 5
    233, 2568, 3.613696575164795, 2
    544, 40, 2.880689859390259, 2
    265, 828, 3.7744603157043457, 5
    88, 3850, 2.9512505531311035, 5
    168, 3744, 3.7834396362304688, 4
    431, 4131, 3.5071723461151123, 4
    324, 596, 3.7294716835021973, 3
    482, 4342, 3.7441132068634033, 4
    583, 378, 3.485652208328247, 5
    168, 10, 4.093012809753418, 4
    488, 3000, 2.490851402282715, 3
    447, 8611, 2.969968795776367, 3
    15, 2043, 2.7697110176086426, 3
    153, 7061, 4.448229789733887, 5
    110, 3147, 3.4573159217834473, 3
    553, 1542, 3.8340682983398438, 4
    225, 1543, 3.0805742740631104, 3
    495, 8354, 3.15423846244812, 4
    561, 5619, 3.843162775039673, 4
    185, 901, 4.441606521606445, 5
    575, 923, 3.28829288482666, 4
    447, 8107, 3.2503085136413574, 1
    376, 5000, 3.8584156036376953, 5
    589, 2706, 3.3218748569488525, 3
    158, 7058, 3.0744199752807617, 5
    255, 8035, 3.704491138458252, 3
    430, 1223, 2.7624855041503906, 4
    447, 315, 3.1127822399139404, 3
    225, 503, 2.926236391067505, 3
    306, 1163, 2.054816722869873, 2
    338, 98, 4.040884971618652, 4
    32, 969, 3.746856451034546, 3
    524, 1978, 2.982759714126587, 3
    461, 9069, 2.9472358226776123, 2
    444, 2224, 4.12364387512207, 5
    413, 1831, 2.9983115196228027, 2
    462, 1480, 3.1425857543945312, 3
    598, 5151, 2.0225679874420166, 1
    216, 2066, 2.9883739948272705, 3
    293, 827, 3.189955711364746, 3
    311, 1468, 3.397214889526367, 4
    602, 472, 3.310779571533203, 4
    22, 819, 3.5610597133636475, 3
    405, 1234, 3.3459300994873047, 2
    176, 1149, 2.963029623031616, 4
    27, 5122, 2.599433183670044, 3
    364, 7061, 2.7480432987213135, 2
    609, 1374, 3.363356590270996, 5
    598, 5929, 2.100186824798584, 2
    231, 5214, 3.325735569000244, 3
    231, 7102, 3.1677849292755127, 3
    347, 4007, 4.388143539428711, 4
    413, 1262, 3.1762120723724365, 3
    447, 8678, 2.8882274627685547, 3
    198, 2026, 2.7938055992126465, 4
    413, 7967, 3.725790500640869, 4
    563, 7088, 2.955430507659912, 4
    579, 3226, 2.3775711059570312, 4
    120, 21, 3.0489656925201416, 3
    116, 660, 3.733470916748047, 3
    273, 1099, 2.864579916000366, 3
    203, 3633, 4.207695484161377, 4
    304, 4918, 3.603288173675537, 4
    342, 3234, 3.730684995651245, 4
    379, 7167, 3.246704339981079, 4
    447, 1153, 2.969440460205078, 4
    468, 899, 4.071874141693115, 4
    551, 2544, 2.8817169666290283, 0
    379, 4956, 3.9774742126464844, 4
    554, 2762, 3.4641337394714355, 4
    147, 3191, 3.44327974319458, 4
    521, 1217, 3.684375524520874, 3
    380, 7008, 3.38558292388916, 4
    369, 2994, 3.376560926437378, 3
    75, 6993, 3.326552391052246, 4
    463, 1072, 3.366835355758667, 4
    165, 2739, 3.491945266723633, 4
    335, 46, 4.581425189971924, 5
    473, 2785, 3.2510898113250732, 4
    199, 4900, 4.363604545593262, 4
    413, 1938, 3.8084874153137207, 5
    44, 2434, 3.2641003131866455, 3
    61, 7197, 3.9228713512420654, 5
    473, 2604, 2.8775064945220947, 4
    67, 5290, 2.517106771469116, 3
    180, 23, 3.0077617168426514, 3
    416, 2035, 3.732609748840332, 5
    524, 8448, 3.572866678237915, 3
    576, 314, 3.7738194465637207, 5
    355, 793, 4.072738170623779, 3
    598, 3322, 2.349360942840576, 1
    494, 1502, 4.119909763336182, 4
    283, 313, 2.969093084335327, 3
    524, 3462, 3.009896755218506, 1
    306, 4907, 2.637772798538208, 3
    602, 831, 3.175035238265991, 2
    297, 6310, 2.5160882472991943, 3
    380, 2962, 2.8564963340759277, 3
    527, 1938, 3.6738710403442383, 5
    480, 1188, 2.7306456565856934, 3
    248, 6173, 3.0363402366638184, 2
    198, 832, 2.9142043590545654, 3
    121, 8184, 4.315890789031982, 5
    63, 545, 3.2315235137939453, 3
    442, 6613, 3.565200090408325, 4
    569, 1066, 3.6528892517089844, 3
    604, 1971, 2.9606428146362305, 5
    425, 3006, 3.70027756690979, 2
    461, 1768, 2.731484889984131, 2
    447, 5730, 2.795393466949463, 2
    413, 6193, 2.9433116912841797, 2
    340, 8882, 3.7095518112182617, 4
    297, 7061, 2.461674690246582, 3
    289, 940, 4.461874961853027, 5
    287, 6700, 2.803938627243042, 3
    598, 2013, 2.1451258659362793, 3
    121, 1437, 4.473652362823486, 5
    433, 793, 3.622690200805664, 4
    576, 379, 2.2773964405059814, 3
    479, 92, 3.177858352661133, 3
    447, 8045, 3.1834774017333984, 3
    473, 6877, 3.246756076812744, 4
    121, 314, 4.982174396514893, 5
    306, 43, 2.688748836517334, 4
    549, 9040, 3.7539377212524414, 3
    473, 3913, 2.953767776489258, 4
    274, 468, 4.232001304626465, 5
    72, 6372, 3.1228928565979004, 4
    599, 2215, 2.7608284950256348, 3
    312, 1230, 3.275653839111328, 4
    58, 695, 4.1223249435424805, 5
    447, 2888, 2.7998509407043457, 4
    293, 1542, 2.8365325927734375, 4
    118, 8105, 3.9382574558258057, 4
    598, 3693, 2.323087215423584, 1
    392, 3539, 3.757826805114746, 3
    238, 337, 3.627873420715332, 3
    49, 906, 3.216169595718384, 4
    47, 1644, 4.029608726501465, 4
    181, 25, 3.254850387573242, 4
    88, 8971, 2.808436155319214, 4
    391, 2144, 3.3672258853912354, 4
    591, 9, 3.58663272857666, 3
    306, 2453, 1.8978358507156372, 1
    327, 7751, 2.8146347999572754, 2
    296, 505, 2.7320990562438965, 1
    554, 1060, 3.8306779861450195, 4
    183, 6726, 3.503512144088745, 4
    221, 1397, 2.6040382385253906, 2
    379, 2968, 3.7197585105895996, 3
    598, 2761, 2.6388070583343506, 5
    287, 1552, 2.977536201477051, 3
    246, 8420, 3.5129714012145996, 4
    306, 2347, 2.495403289794922, 2
    20, 8187, 2.410567283630371, 4
    476, 6493, 3.3159055709838867, 4
    303, 191, 3.850645065307617, 5
    312, 2567, 3.1202526092529297, 1
    124, 7124, 3.5486011505126953, 4
    253, 510, 4.156819820404053, 4
    521, 7073, 3.3576602935791016, 4
    598, 8643, 2.350977659225464, 1
    447, 1527, 2.863489866256714, 4
    7, 46, 4.106969833374023, 5
    231, 3982, 2.7175257205963135, 3
    437, 5834, 3.192779064178467, 4
    215, 2899, 3.2887911796569824, 4
    168, 3005, 3.598336935043335, 4
    282, 633, 3.494710922241211, 3
    488, 337, 2.7397875785827637, 3
    407, 2391, 3.747702121734619, 5
    554, 287, 3.7448248863220215, 3
    27, 933, 2.8891077041625977, 3
    215, 2959, 3.3539042472839355, 5
    483, 503, 3.4077608585357666, 4
    413, 326, 3.479398488998413, 4
    329, 1182, 3.1654245853424072, 4
    138, 6138, 2.326927900314331, 2
    479, 166, 2.7129390239715576, 1
    448, 1795, 2.653064489364624, 4
    350, 7026, 3.5190200805664062, 3
    221, 8000, 2.797459363937378, 2
    176, 7524, 3.3501009941101074, 3
    413, 2984, 3.3418991565704346, 5
    454, 217, 3.2262940406799316, 3
    467, 0, 3.7323575019836426, 4
    413, 3698, 3.301975727081299, 3
    589, 6693, 3.7376160621643066, 4
    176, 6614, 3.0755505561828613, 2
    143, 16, 3.5416040420532227, 4
    102, 1430, 3.799680233001709, 5
    273, 4609, 2.5072596073150635, 2
    289, 98, 4.300109386444092, 5
    413, 7428, 3.196812868118286, 2
    516, 5324, 2.6262705326080322, 1
    508, 3897, 2.730614185333252, 1
    572, 2982, 3.3017640113830566, 4
    18, 1084, 2.006356954574585, 3
    606, 1182, 3.7862679958343506, 3
    110, 8669, 3.19549822807312, 3
    461, 2979, 3.250380277633667, 3
    181, 2637, 3.6269423961639404, 4
    324, 1577, 3.238084316253662, 4
    351, 6045, 3.7019448280334473, 4
    106, 275, 3.355508327484131, 3
    121, 8648, 4.263613224029541, 5
    589, 1438, 3.2022619247436523, 4
    61, 1938, 4.223967552185059, 5
    287, 2952, 3.130037307739258, 3
    566, 8187, 1.5097662210464478, 1
    589, 314, 3.9284677505493164, 5
    604, 701, 3.3535847663879395, 2
    473, 184, 3.102341651916504, 1
    121, 6329, 4.5652570724487305, 4
    18, 1961, 2.458796977996826, 2
    92, 1054, 4.152352333068848, 5
    287, 65, 3.2343382835388184, 3
    579, 1903, 3.033280849456787, 2
    212, 7061, 3.9418716430664062, 2
    6, 5880, 2.9636402130126953, 1
    554, 2648, 3.5280425548553467, 4
    598, 9462, 2.2490034103393555, 1
    44, 2638, 3.838573694229126, 4
    220, 2109, 4.16562557220459, 3
    606, 989, 4.0941877365112305, 4
    589, 3841, 2.7525038719177246, 1
    151, 7398, 3.773618698120117, 3
    41, 3296, 3.3897945880889893, 2
    27, 2475, 2.740739345550537, 4
    464, 2779, 4.270610809326172, 4
    579, 2392, 3.0525686740875244, 4
    312, 2978, 3.0879509449005127, 5
    293, 167, 2.577306032180786, 2
    200, 126, 3.8080315589904785, 4
    559, 6441, 3.297687530517578, 3
    176, 6192, 3.174679756164551, 4
    516, 1, 2.1770520210266113, 3
    293, 2512, 2.7088146209716797, 4
    602, 1578, 3.403935670852661, 3
    152, 7876, 1.902429461479187, 5
    23, 4342, 3.7035176753997803, 4
    181, 585, 3.257077932357788, 4
    482, 1479, 3.36633038520813, 4
    246, 3827, 3.0419747829437256, 2
    591, 283, 3.4389243125915527, 3
    113, 9433, 3.1463687419891357, 3
    367, 811, 2.719268798828125, 3
    524, 7626, 3.3076417446136475, 3
    146, 486, 3.3885464668273926, 4
    465, 8358, 3.79531192779541, 4
    159, 2691, 2.514235019683838, 4
    322, 8377, 3.1829376220703125, 1
    225, 1297, 3.433990478515625, 5
    427, 2218, 2.5126240253448486, 3
    295, 9460, 3.143698215484619, 4
    333, 3979, 3.416283130645752, 3
    488, 6317, 2.513469696044922, 0
    287, 1677, 2.9483120441436768, 2
    289, 2085, 3.59213924407959, 3
    500, 91, 3.4176864624023438, 3
    447, 5245, 2.922868251800537, 3
    27, 800, 2.6177496910095215, 3
    447, 1988, 2.3103954792022705, 2
    437, 1170, 3.051044464111328, 3
    35, 701, 3.003065347671509, 1
    50, 3202, 3.7737512588500977, 4
    90, 4600, 3.3025712966918945, 2
    255, 17, 3.6725704669952393, 5
    508, 549, 2.7771871089935303, 3
    38, 1627, 3.7956738471984863, 3
    27, 7013, 2.6265437602996826, 4
    291, 7586, 3.071969747543335, 2
    79, 6416, 3.9946255683898926, 4
    79, 7669, 3.5982275009155273, 4
    176, 123, 3.5834155082702637, 4
    20, 1486, 2.780951976776123, 5
    351, 2189, 3.7640132904052734, 4
    56, 116, 3.40042781829834, 4
    273, 906, 3.483003854751587, 4
    406, 224, 3.931321859359741, 4
    413, 3762, 3.2701590061187744, 2
    89, 863, 3.864396095275879, 4
    413, 8295, 3.6116297245025635, 4
    351, 5939, 3.2257657051086426, 3
    361, 659, 4.3401360511779785, 5
    568, 509, 3.5029819011688232, 3
    355, 1938, 4.188778877258301, 4
    108, 145, 2.3370063304901123, 2
    499, 2248, 3.6014907360076904, 4
    482, 167, 3.210073232650757, 4
    273, 5811, 2.544222831726074, 3
    50, 2255, 2.8120908737182617, 3
    201, 334, 3.7601065635681152, 3
    605, 4606, 3.0458004474639893, 2
    189, 3189, 3.8779478073120117, 3
    379, 3340, 3.734898567199707, 3
    605, 1131, 3.855247974395752, 3
    287, 2596, 3.006234645843506, 4
    67, 7038, 2.0760815143585205, 4
    557, 815, 4.043446063995361, 5
    211, 819, 3.6071231365203857, 4
    413, 5753, 2.741894006729126, 2
    128, 4631, 3.6165432929992676, 3
    42, 140, 3.906348705291748, 5
    186, 508, 3.868029832839966, 1
    437, 3147, 3.4779272079467773, 3
    268, 642, 3.132460594177246, 3
    41, 3347, 3.6043624877929688, 4
    599, 1857, 2.773102283477783, 3
    466, 1793, 3.0999624729156494, 1
    415, 2144, 3.3174774646759033, 3
    609, 4191, 3.130385637283325, 3
    238, 3785, 3.3315987586975098, 4
    65, 3774, 3.3765933513641357, 4
    140, 1153, 3.139268159866333, 3
    216, 2533, 3.076052665710449, 3
    273, 355, 2.623741388320923, 2
    126, 1438, 2.820819139480591, 3
    221, 7679, 3.1402745246887207, 2
    5, 815, 3.427238941192627, 3
    386, 3434, 2.072068214416504, 3
    609, 7627, 3.050581932067871, 5
    427, 431, 2.5450868606567383, 2
    197, 59, 3.564368486404419, 1
    273, 6314, 3.3169658184051514, 3
    604, 5450, 3.2766168117523193, 3
    599, 1955, 2.8313698768615723, 3
    390, 1075, 3.660623550415039, 3
    316, 7587, 3.8484630584716797, 4
    591, 14, 3.133810520172119, 4
    605, 1954, 3.280364990234375, 3
    181, 1567, 2.934901714324951, 4
    409, 810, 3.718010425567627, 4
    412, 7137, 3.685399293899536, 5
    317, 6153, 3.5568175315856934, 4
    413, 3204, 3.51932954788208, 4
    297, 8419, 2.2862160205841064, 0
    468, 2615, 3.8852996826171875, 4
    430, 976, 2.8477675914764404, 4
    41, 694, 4.310965538024902, 5
    576, 1574, 3.0659239292144775, 4
    587, 9, 3.418405055999756, 3
    98, 275, 3.682429552078247, 2
    353, 3827, 3.124934673309326, 3
    602, 1171, 3.323683977127075, 3
    579, 910, 3.5192203521728516, 4
    576, 1556, 2.828331232070923, 5
    67, 999, 2.9023196697235107, 3
    447, 3065, 2.923124313354492, 4
    598, 7230, 2.2876338958740234, 2
    338, 1756, 3.774629592895508, 3
    438, 314, 4.370880603790283, 5
    63, 5064, 2.9208478927612305, 4
    413, 6138, 3.5611326694488525, 4
    317, 7207, 3.1081600189208984, 3
    157, 6575, 3.1245687007904053, 3
    293, 1311, 2.397141695022583, 1
    595, 2077, 3.6619443893432617, 4
    559, 6464, 3.3619680404663086, 2
    278, 334, 3.2272331714630127, 3
    287, 2511, 3.428267002105713, 3
    550, 8551, 3.4499049186706543, 4
    541, 257, 3.7323737144470215, 3
    27, 5869, 2.748975992202759, 2
    452, 2125, 3.527982711791992, 2
    72, 9164, 3.390537977218628, 2
    168, 1881, 4.102460861206055, 5
    72, 123, 3.7941930294036865, 2
    364, 6613, 2.418718099594116, 3
    151, 6797, 3.5655698776245117, 3
    386, 249, 3.155785083770752, 4
    262, 3915, 3.3504257202148438, 4
    170, 819, 4.589654445648193, 5
    359, 1236, 3.520845890045166, 1
    0, 1479, 4.288753509521484, 5
    516, 2222, 1.8451142311096191, 5
    167, 6993, 4.59385871887207, 5
    306, 1082, 2.293649673461914, 0
    521, 7675, 3.4904212951660156, 5
    331, 2224, 3.8428902626037598, 4
    65, 945, 4.105629920959473, 5
    199, 5980, 3.299072504043579, 2
    273, 2325, 2.8544304370880127, 3
    194, 910, 3.66062593460083, 4
    248, 8674, 3.660438299179077, 4
    452, 3380, 3.656731367111206, 4
    293, 959, 3.0547995567321777, 4
    67, 3241, 2.389125347137451, 2
    41, 3355, 3.396801710128784, 5
    473, 2296, 2.9669101238250732, 3
    231, 6819, 3.2010154724121094, 4
    573, 156, 3.2703754901885986, 2
    354, 1290, 3.2897675037384033, 4
    176, 1066, 3.757967472076416, 3
    447, 224, 3.489173650741577, 5
    144, 197, 2.923426866531372, 3
    259, 43, 3.778672218322754, 4
    413, 3953, 2.9633097648620605, 5
    482, 5078, 3.5692214965820312, 4
    20, 6770, 2.7439658641815186, 3
    198, 73, 2.9838738441467285, 3
    508, 4342, 3.2897329330444336, 3
    27, 5968, 2.690098524093628, 4
    330, 7961, 3.3635849952697754, 3
    181, 3953, 2.949798822402954, 4
    479, 4134, 3.144845962524414, 3
    125, 123, 3.295271635055542, 4
    413, 877, 3.575770378112793, 4
    609, 5788, 3.0786547660827637, 3
    201, 2864, 3.9894094467163086, 5
    436, 8, 3.5445220470428467, 3
    264, 1778, 3.056934118270874, 2
    489, 2224, 3.5726072788238525, 3
    414, 147, 3.595895528793335, 4
    49, 4662, 2.5198676586151123, 3
    602, 139, 3.1607115268707275, 4
    221, 613, 3.1610443592071533, 3
    3, 2984, 3.6018574237823486, 3
    225, 4604, 3.2393412590026855, 4
    89, 1045, 4.038712501525879, 5
    17, 8377, 3.4952754974365234, 3
    598, 537, 2.309471607208252, 1
    404, 2300, 3.5880353450775146, 3
    602, 2263, 3.5144436359405518, 4
    16, 899, 4.208418369293213, 4
    473, 4631, 3.163145065307617, 2
    415, 2077, 3.4246888160705566, 2
    199, 514, 3.582077980041504, 5
    563, 8431, 2.9187281131744385, 4
    423, 1703, 3.587921380996704, 4
    220, 877, 3.9620113372802734, 4
    225, 4153, 3.3141028881073, 4
    602, 685, 3.6286232471466064, 5
    136, 1734, 3.4804019927978516, 4
    255, 7902, 3.5789802074432373, 3
    609, 3229, 2.8431804180145264, 3
    380, 4786, 3.530083179473877, 4
    437, 3884, 3.2930915355682373, 3
    329, 1266, 3.440603017807007, 3
    598, 2964, 2.129641056060791, 1
    411, 1395, 3.802260398864746, 4
    589, 4126, 3.2002413272857666, 2
    479, 3557, 3.4490878582000732, 5
    201, 992, 3.6953554153442383, 4
    356, 4977, 3.72469425201416, 5
    65, 4213, 3.5660629272460938, 3
    605, 6472, 3.7754411697387695, 3
    181, 4552, 3.1501803398132324, 4
    447, 9313, 2.9644620418548584, 2
    304, 8972, 3.8863325119018555, 3
    411, 1429, 4.284717559814453, 4
    176, 694, 3.962479591369629, 5
    130, 4131, 3.43863582611084, 3
    386, 5344, 2.716606378555298, 3
    609, 2781, 3.28983211517334, 4
    181, 3647, 3.386753559112549, 4
    609, 2912, 3.4936068058013916, 4
    558, 527, 3.258763074874878, 3
    246, 1283, 3.825763702392578, 4
    566, 8377, 2.379995822906494, 3
    229, 2438, 2.64487624168396, 1
    233, 793, 4.114879131317139, 4
    413, 4083, 3.233351707458496, 2
    304, 6148, 3.760470390319824, 5
    473, 6229, 3.3053395748138428, 4
    479, 260, 3.261260509490967, 4
    563, 6708, 3.4048428535461426, 4
    284, 919, 3.716327428817749, 4
    602, 474, 3.8221821784973145, 3
    232, 322, 3.646923542022705, 3
    44, 619, 3.3725218772888184, 4
    474, 8668, 4.078304767608643, 4
    92, 835, 4.2403764724731445, 5
    155, 1439, 3.3266701698303223, 3
    206, 1785, 3.004826784133911, 0
    41, 1571, 3.8251166343688965, 4
    50, 2884, 3.0822136402130127, 5
    130, 98, 3.3865082263946533, 4
    312, 43, 3.6387877464294434, 5
    609, 3350, 3.510040521621704, 4
    306, 1378, 2.193909168243408, 3
    468, 1242, 3.8355133533477783, 5
    124, 8420, 3.7730937004089355, 4
    401, 254, 3.9997286796569824, 3
    267, 1352, 3.032090663909912, 3
    413, 4070, 3.6258490085601807, 3
    225, 197, 2.874502182006836, 5
    325, 4796, 3.2579963207244873, 4
    605, 1733, 3.9662392139434814, 4
    598, 332, 1.9758172035217285, 1
    56, 1474, 3.213078260421753, 5
    124, 7752, 3.766893148422241, 3
    375, 867, 3.8536925315856934, 5
    240, 8434, 2.9720394611358643, 4
    333, 690, 3.2999887466430664, 3
    91, 1881, 3.602203369140625, 4
    278, 8298, 3.459080696105957, 2
    165, 963, 3.7476818561553955, 4
    576, 1066, 3.601351499557495, 5
    3, 2994, 3.7410805225372314, 1
    444, 681, 3.710750102996826, 3
    447, 1372, 1.9624269008636475, 3
    348, 510, 4.014538764953613, 5
    516, 8285, 2.78639554977417, 1
    317, 4707, 3.545654296875, 3
    607, 1321, 3.0521390438079834, 3
    90, 1457, 3.104583740234375, 2
    459, 5938, 4.009897232055664, 5
    111, 15, 3.7621543407440186, 4
    379, 4256, 3.4654829502105713, 4
    216, 1594, 2.5995771884918213, 4
    180, 509, 3.187267541885376, 2
    447, 6293, 2.8469157218933105, 4
    605, 6207, 3.2837586402893066, 3
    557, 1290, 3.732119560241699, 4
    459, 5836, 4.010190010070801, 3
    416, 3136, 4.645247459411621, 5
    494, 7626, 3.6182267665863037, 5
    304, 1480, 3.210155487060547, 5
    481, 509, 3.452655553817749, 4
    379, 6897, 3.573092460632324, 4
    266, 2670, 4.154393196105957, 5
    143, 217, 3.4471206665039062, 4
    221, 3635, 3.3727846145629883, 4
    508, 6968, 2.7292356491088867, 4
    539, 910, 3.9922502040863037, 5
    598, 2027, 2.5497350692749023, 4
    437, 1727, 2.851961612701416, 2
    476, 692, 3.5040512084960938, 3
    550, 2370, 3.640385389328003, 4
    342, 6621, 3.677363634109497, 4
    524, 0, 3.5332794189453125, 4
    456, 2458, 3.340641736984253, 2
    478, 3052, 3.4551138877868652, 2
    602, 384, 3.2808337211608887, 5
    353, 4419, 3.053388833999634, 3
    186, 5708, 3.488229751586914, 2
    445, 244, 3.1059534549713135, 2
    200, 514, 4.119604110717773, 5
    18, 956, 3.0296685695648193, 2
    589, 2250, 3.2979605197906494, 2
    61, 2224, 4.424605846405029, 5
    180, 412, 3.2442877292633057, 4
    88, 8217, 3.190520763397217, 2
    161, 46, 4.23698616027832, 4
    297, 4519, 2.3572893142700195, 3
    413, 5990, 3.147594451904297, 4
    381, 7814, 3.497709274291992, 1
    158, 9256, 3.200376510620117, 4
    607, 2803, 2.7576303482055664, 4
    598, 9661, 2.4229652881622314, 3
    176, 6, 3.1198060512542725, 1
    245, 3979, 4.220835208892822, 5
    136, 731, 3.7338790893554688, 5
    602, 2660, 3.207186698913574, 1
    220, 4421, 3.8154475688934326, 3
    152, 9603, 2.3410210609436035, 1
    312, 325, 3.3966403007507324, 2
    607, 1480, 2.5036559104919434, 2
    248, 4421, 3.6240224838256836, 4
    231, 2037, 3.356675386428833, 3
    287, 3668, 3.36155366897583, 5
    497, 32, 4.262110710144043, 3
    281, 2511, 3.725001811981201, 4
    329, 835, 3.346240997314453, 4
    35, 3047, 2.7506301403045654, 2
    489, 6693, 3.3391880989074707, 4
    265, 2231, 3.5900399684906006, 5
    306, 684, 2.5618913173675537, 3
    473, 4691, 3.054239273071289, 3
    426, 3793, 2.643562078475952, 3
    386, 5290, 2.898162603378296, 3
    324, 1210, 3.645890474319458, 4
    138, 3794, 1.865760326385498, 1
    306, 2284, 2.294090747833252, 3
    488, 6533, 2.7502827644348145, 3
    607, 2912, 3.0864169597625732, 3
    473, 3783, 3.0383081436157227, 3
    524, 4551, 3.5272772312164307, 4
    31, 592, 3.5104565620422363, 4
    62, 2992, 3.8081252574920654, 3
    56, 702, 3.519911766052246, 4
    379, 5999, 4.124089241027832, 5
    135, 126, 2.99575138092041, 3
    266, 1960, 3.813767433166504, 4
    42, 363, 3.8473262786865234, 5
    0, 914, 4.579649925231934, 4
    176, 5417, 2.9269895553588867, 5
    0, 1502, 4.860281944274902, 4
    544, 1370, 2.919159173965454, 4
    381, 6911, 3.311633825302124, 5
    562, 5217, 2.6588802337646484, 3
    306, 968, 2.652230978012085, 4
    65, 2756, 3.7109553813934326, 4
    238, 19, 3.2482025623321533, 3
    325, 6032, 3.4364352226257324, 4
    180, 130, 2.9908447265625, 2
    231, 6400, 3.3262898921966553, 2
    508, 7571, 2.6130428314208984, 3
    591, 592, 3.4114105701446533, 5
    185, 3517, 4.157099723815918, 5
    569, 2370, 3.633389711380005, 4
    589, 4939, 3.059356451034546, 3
    138, 7047, 1.922778606414795, 1
    607, 1545, 2.6695964336395264, 1
    41, 633, 3.8754327297210693, 5
    327, 6513, 2.7915894985198975, 2
    602, 2592, 3.590470790863037, 3
    598, 1625, 2.2378220558166504, 2
    41, 943, 3.7103819847106934, 5
    431, 3143, 3.5553765296936035, 4
    374, 3136, 4.546265602111816, 5
    379, 7969, 3.3431100845336914, 2
    423, 919, 3.659191370010376, 4
    598, 86, 2.368582248687744, 3
    287, 6440, 2.9442102909088135, 4
    201, 1075, 3.7688732147216797, 3
    526, 418, 3.996070384979248, 5
    605, 542, 3.245645523071289, 4
    316, 4291, 3.0560672283172607, 4
    473, 2687, 3.169576644897461, 4
    50, 1037, 3.8120007514953613, 4
    311, 2067, 3.304300308227539, 3
    18, 2093, 2.799147367477417, 2
    306, 1138, 2.38287353515625, 1
    104, 9120, 3.5153322219848633, 5
    413, 3923, 2.821974277496338, 3
    246, 815, 3.4905195236206055, 4
    248, 8234, 3.3856112957000732, 5
    589, 1287, 3.067633867263794, 2
    265, 31, 3.926222562789917, 4
    513, 4993, 3.3117802143096924, 3
    63, 867, 3.415862560272217, 4
    413, 3553, 3.172227144241333, 5
    94, 1822, 3.6965365409851074, 4
    148, 224, 3.5741984844207764, 4
    413, 1481, 3.1049411296844482, 2
    607, 5250, 3.0956482887268066, 3
    245, 2096, 4.053587913513184, 3
    127, 1066, 4.798447608947754, 3
    353, 1291, 3.187972068786621, 3
    289, 260, 4.04708194732666, 4
    313, 235, 2.6606104373931885, 4
    68, 1978, 3.9732437133789062, 4
    105, 7626, 3.544186592102051, 5
    490, 6926, 3.659518241882324, 4
    413, 6418, 3.2289552688598633, 3
    605, 6012, 3.223881244659424, 3
    605, 2742, 3.2845306396484375, 4
    212, 6112, 3.6336283683776855, 3
    545, 1447, 2.9588403701782227, 4
    431, 6905, 3.4354054927825928, 4
    124, 6613, 3.548060178756714, 2
    447, 4620, 2.8291757106781006, 4
    231, 7070, 3.2739644050598145, 3
    494, 4799, 3.2499380111694336, 4
    300, 3136, 3.7770133018493652, 4
    32, 1881, 4.095828056335449, 2
    209, 2037, 3.8073482513427734, 4
    605, 3028, 3.506791114807129, 3
    88, 7742, 3.2879538536071777, 5
    371, 1034, 3.299842119216919, 2
    554, 1438, 3.6031503677368164, 5
    56, 922, 3.683725357055664, 4
    308, 5124, 3.4415431022644043, 4
    248, 3358, 3.4124627113342285, 3
    181, 4135, 3.2200539112091064, 4
    562, 514, 2.836066722869873, 4
    413, 6180, 3.2163467407226562, 3
    321, 3617, 3.490644693374634, 4
    447, 8821, 2.648319959640503, 5
    344, 614, 4.010666370391846, 5
    317, 8856, 2.99218487739563, 3
    238, 613, 3.859996795654297, 4
    609, 7448, 3.7212274074554443, 5
    97, 898, 4.121161460876465, 5
    3, 883, 3.6636133193969727, 4
    321, 2670, 3.117675542831421, 4
    317, 4819, 3.5756564140319824, 4
    482, 619, 3.008641242980957, 2
    366, 1785, 3.945632219314575, 3
    330, 5927, 3.309690475463867, 2
    214, 1575, 3.649388313293457, 4
    110, 8673, 3.0639450550079346, 5
    413, 810, 3.16715931892395, 5
    173, 275, 3.5756850242614746, 5
    444, 1938, 3.9230053424835205, 5
    417, 314, 4.138664722442627, 4
    241, 257, 3.8917901515960693, 5
    104, 9519, 3.6732637882232666, 5
    61, 7637, 3.6988677978515625, 3
    563, 7368, 3.1139228343963623, 3
    447, 263, 2.5446269512176514, 3
    452, 3205, 3.581536054611206, 4
    494, 3567, 3.4226293563842773, 2
    79, 6182, 3.93526029586792, 4
    194, 690, 3.412578582763672, 5
    67, 797, 2.668085813522339, 2
    473, 25, 3.1224989891052246, 3
    368, 613, 3.0898866653442383, 4
    40, 8294, 2.8361072540283203, 4
    312, 963, 3.683987617492676, 4
    494, 2806, 3.321718215942383, 3
    433, 1397, 2.8304338455200195, 2
    173, 217, 3.7135682106018066, 5
    410, 21, 2.986252784729004, 3
    312, 447, 3.0483129024505615, 3
    9, 7468, 2.851767063140869, 2
    304, 895, 4.219141960144043, 5
    541, 4421, 3.387516498565674, 4
    155, 1088, 3.561923027038574, 3
    464, 690, 4.142297744750977, 5
    18, 2328, 2.6042861938476562, 3
    201, 2072, 3.730724573135376, 2
    279, 325, 3.849914312362671, 2
    427, 2899, 2.2757742404937744, 3
    476, 4219, 3.09903883934021, 3
    114, 1070, 3.4055073261260986, 2
    473, 365, 3.036492109298706, 3
    413, 3036, 3.275813579559326, 2
    488, 1157, 2.992696523666382, 4
    605, 3473, 3.4491665363311768, 2
    473, 1643, 2.912452220916748, 3
    447, 620, 2.9081621170043945, 4
    437, 4109, 3.149587869644165, 2
    198, 4345, 3.030568838119507, 2
    533, 4135, 3.4419054985046387, 3
    90, 1829, 3.258084535598755, 2
    322, 8411, 3.1400938034057617, 2
    58, 520, 4.470494270324707, 5
    579, 191, 3.2419536113739014, 3
    81, 3189, 3.5565237998962402, 4
    519, 979, 3.6151657104492188, 4
    31, 315, 3.7720847129821777, 3
    325, 6341, 3.7580227851867676, 5
    67, 2843, 2.7804877758026123, 3
    324, 271, 2.9396533966064453, 2
    474, 5363, 4.566483497619629, 4
    552, 38, 3.7446255683898926, 4
    124, 8434, 3.255187511444092, 4
    386, 2583, 3.5638933181762695, 2
    194, 3242, 3.0262796878814697, 3
    72, 7628, 3.4815359115600586, 4
    144, 509, 3.269493579864502, 3
    437, 4066, 3.1840994358062744, 2
    532, 2782, 3.7331762313842773, 5
    221, 257, 3.477911949157715, 4
    607, 2, 2.8242995738983154, 2
    88, 7646, 2.8659682273864746, 2
    215, 3005, 3.354684352874756, 2
    596, 1225, 3.922553062438965, 4
    307, 4600, 2.3681302070617676, 3
    598, 503, 2.164611577987671, 2
    413, 5291, 3.527433395385742, 5
    42, 325, 4.43928337097168, 5
    83, 993, 3.557835340499878, 4
    306, 379, 1.5102529525756836, 1
    352, 337, 3.087453603744507, 3
    246, 6045, 3.5114684104919434, 2
    186, 3613, 3.7118630409240723, 4
    207, 2164, 2.890347719192505, 3
    554, 118, 3.349257469177246, 4
    386, 1471, 3.304698944091797, 4
    231, 4935, 3.248290538787842, 4
    94, 4664, 3.5673184394836426, 4
    413, 672, 3.2519545555114746, 3
    273, 334, 2.7730393409729004, 3
    198, 4473, 3.1088650226593018, 3
    183, 9635, 3.7053234577178955, 4
    186, 3285, 3.3683764934539795, 4
    544, 1825, 3.1110916137695312, 4
    609, 4131, 3.7335331439971924, 5
    424, 3002, 3.293015956878662, 3
    496, 659, 4.006258010864258, 4
    602, 55, 3.460432291030884, 1
    220, 895, 4.326528549194336, 5
    520, 31, 4.003690242767334, 3
    607, 594, 2.803217887878418, 3
    317, 8575, 2.9394147396087646, 4
    342, 901, 3.702454090118408, 3
    211, 2888, 3.217008590698242, 4
    407, 1291, 3.439103126525879, 4
    497, 257, 4.583365440368652, 4
    5, 426, 3.1205544471740723, 3
    599, 2264, 2.9161837100982666, 4
    576, 914, 3.412208318710327, 4
    579, 5879, 2.9483203887939453, 5
    421, 2963, 2.8825981616973877, 3
    317, 6805, 2.99554705619812, 4
    413, 3609, 3.6286063194274902, 4
    368, 1186, 2.9060802459716797, 3
    302, 914, 3.7217936515808105, 5
    390, 260, 3.8740391731262207, 4
    598, 89, 2.3764586448669434, 2
    431, 4328, 3.299333333969116, 4
    17, 549, 3.046454906463623, 3
    435, 378, 3.111574649810791, 3
    386, 1496, 3.210749864578247, 3
    273, 5755, 2.7069075107574463, 3
    287, 2481, 3.3326058387756348, 3
    524, 5712, 3.4096574783325195, 4
    407, 7870, 3.5628981590270996, 2
    598, 2432, 2.312910318374634, 3
    67, 7043, 2.90191388130188, 4
    384, 333, 3.115873098373413, 3
    604, 968, 3.4392952919006348, 4
    566, 7802, 1.8240671157836914, 0
    273, 7, 2.8854598999023438, 3
    458, 3563, 4.187591075897217, 4
    397, 1616, 3.7914209365844727, 5
    306, 2462, 2.4611778259277344, 1
    602, 2150, 3.002955913543701, 4
    295, 3827, 2.5790798664093018, 1
    273, 3485, 2.4399027824401855, 2
    112, 472, 3.5147202014923096, 3
    41, 2524, 3.3863556385040283, 4
    338, 6341, 3.7766120433807373, 4
    19, 3494, 3.042964220046997, 3
    500, 54, 3.4096405506134033, 3
    381, 8212, 3.3422305583953857, 3
    110, 8683, 3.2664754390716553, 4
    607, 3026, 2.6875360012054443, 3
    263, 1054, 3.73982834815979, 4
    303, 1882, 4.101134777069092, 3
    216, 2042, 2.4544432163238525, 3
    90, 4596, 3.1547327041625977, 5
    390, 1540, 3.774277925491333, 4
    95, 266, 3.7085814476013184, 1
    494, 3347, 3.5263302326202393, 4
    452, 3152, 3.887709379196167, 5
    4, 225, 3.8417654037475586, 4
    473, 1540, 3.2032103538513184, 4
    263, 5026, 3.4624414443969727, 3
    18, 1569, 2.835181951522827, 3
    181, 0, 3.5960235595703125, 4
    461, 7154, 3.1188154220581055, 4
    599, 225, 3.208217144012451, 3
    83, 861, 3.10754656791687, 3
    602, 1592, 3.276977062225342, 4
    219, 3347, 3.750154733657837, 4
    110, 3898, 3.1366188526153564, 3
    18, 1739, 2.5850372314453125, 3
    390, 1230, 3.7107036113739014, 5
    473, 4040, 2.998171329498291, 4
    158, 2370, 3.486302375793457, 0
    554, 1857, 3.5907323360443115, 4
    571, 297, 4.032683372497559, 3
    488, 3635, 3.1836512088775635, 4
    369, 3617, 3.805161952972412, 4
    56, 511, 3.4542315006256104, 4
    563, 7035, 3.2703566551208496, 4
    605, 3294, 3.310825824737549, 4
    18, 1251, 2.6777491569519043, 3
    317, 5050, 3.5770998001098633, 4
    598, 4112, 2.667147397994995, 2
    407, 3212, 3.3713364601135254, 4
    131, 436, 2.623684883117676, 2
    317, 6294, 3.7325823307037354, 3
    293, 1474, 2.5864622592926025, 4
    380, 2941, 3.4201266765594482, 5
    219, 2729, 4.042298316955566, 4
    178, 297, 3.3760225772857666, 3
    608, 257, 3.5613179206848145, 4
    254, 2612, 2.2472383975982666, 1
    444, 266, 3.206892251968384, 4
    551, 276, 2.75434947013855, 2
    216, 643, 2.3678767681121826, 3
    215, 2887, 3.5613489151000977, 2
    473, 3021, 2.9620015621185303, 4
    81, 1163, 2.8368067741394043, 3
    563, 1444, 3.4362096786499023, 4
    27, 6693, 3.1708824634552, 3
    521, 6996, 3.297053337097168, 1
    317, 6256, 3.587822914123535, 3
    273, 5030, 2.7377262115478516, 3
    459, 3002, 3.6706185340881348, 4
    75, 6659, 3.4201629161834717, 4
    65, 800, 3.606675386428833, 4
    220, 46, 4.332880020141602, 4
    253, 1515, 2.871558427810669, 4
    474, 8358, 4.412213325500488, 4
    473, 2454, 2.7816734313964844, 2
    308, 695, 3.299347400665283, 4
    451, 3228, 3.9371511936187744, 4
    545, 131, 2.70880126953125, 2
    447, 818, 3.352111339569092, 5
    455, 622, 2.856879472732544, 4
    598, 8387, 2.1307897567749023, 2
    189, 3827, 3.1226158142089844, 3
    354, 697, 3.1701676845550537, 4
    266, 901, 4.133618354797363, 5
    56, 973, 3.3584017753601074, 5
    83, 1001, 3.6003050804138184, 4
    51, 7355, 4.433141231536865, 5
    563, 9068, 3.1388566493988037, 3
    413, 4134, 3.3324193954467773, 3
    219, 4090, 3.3776659965515137, 1
    598, 4601, 2.2471861839294434, 3
    56, 2107, 3.0832841396331787, 3
    559, 4153, 3.2959327697753906, 4
    433, 0, 3.5397789478302, 4
    273, 4402, 2.978456497192383, 3
    488, 2957, 2.50423002243042, 4
    248, 6240, 3.1997721195220947, 2
    17, 1217, 3.541029453277588, 4
    176, 1409, 3.171180248260498, 4
    225, 404, 2.858597755432129, 3
    473, 307, 3.4026401042938232, 3
    216, 2123, 2.808772563934326, 2
    67, 5291, 2.923597812652588, 4
    589, 6192, 3.172712564468384, 2
    598, 4929, 2.1369757652282715, 3
    253, 1217, 3.782958507537842, 4
    62, 615, 3.329495668411255, 3
    609, 6901, 3.6513514518737793, 2
    452, 3002, 3.619990825653076, 5
    338, 1724, 4.005650520324707, 5
    186, 514, 3.447780132293701, 1
    134, 2440, 3.6631500720977783, 3
    211, 809, 3.5021166801452637, 3
    333, 3133, 2.9529361724853516, 2
    41, 3430, 3.74826717376709, 4
    317, 7103, 3.372079372406006, 3
    176, 6447, 3.2641429901123047, 3
    131, 6132, 2.774024486541748, 2
    253, 302, 3.0345988273620605, 3
    273, 2762, 2.647723913192749, 3
    478, 3875, 2.9535884857177734, 1
    56, 1053, 3.152738332748413, 2
    306, 1601, 2.169541358947754, 3
    248, 7700, 3.161963939666748, 3
    166, 933, 3.612043857574463, 3
    482, 921, 3.897165536880493, 3
    10, 1398, 3.2337958812713623, 4
    598, 5543, 2.420295476913452, 3
    231, 3614, 3.6233227252960205, 3
    216, 1558, 3.063124656677246, 2
    232, 4135, 2.951507568359375, 3
    461, 985, 3.3123178482055664, 4
    267, 1102, 2.4421937465667725, 1
    16, 956, 4.011814594268799, 4
    379, 5784, 3.620025634765625, 1
    413, 278, 3.0739448070526123, 4
    68, 314, 4.8807878494262695, 4
    545, 2917, 3.1661171913146973, 4
    296, 1045, 2.9286048412323, 4
    303, 1216, 3.374192237854004, 4
    447, 897, 3.3706421852111816, 5
    501, 1996, 2.731912136077881, 3
    560, 929, 3.384568452835083, 4
    170, 486, 4.740142822265625, 5
    121, 7827, 4.484278678894043, 4
    604, 615, 3.0331690311431885, 3
    436, 989, 3.817729949951172, 4
    447, 857, 2.9734771251678467, 3
    607, 1970, 2.3835065364837646, 3
    312, 522, 3.038315773010254, 5
    317, 1242, 3.732712507247925, 4
    558, 514, 3.2459373474121094, 3
    379, 5686, 3.9379477500915527, 4
    17, 334, 3.0938174724578857, 2
    502, 1290, 3.143470525741577, 2
    209, 9603, 3.866917371749878, 5
    594, 832, 3.2329161167144775, 4
    473, 4926, 3.535616874694824, 4
    18, 1812, 2.6569747924804688, 3
    545, 2933, 2.8389852046966553, 5
    606, 818, 4.238142013549805, 3
    0, 920, 4.468170166015625, 5
    110, 6416, 3.2513267993927, 4
    47, 3868, 3.792597770690918, 4
    110, 8219, 3.0246400833129883, 2
    595, 7379, 3.0155253410339355, 4
    308, 4827, 3.4043936729431152, 4
    523, 11, 3.2267894744873047, 1
    386, 5901, 3.467749834060669, 4
    560, 5365, 2.961402177810669, 2
    550, 8407, 3.3343238830566406, 3
    201, 835, 4.00067663192749, 4
    605, 6864, 3.422743797302246, 3
    265, 2037, 3.6640806198120117, 5
    218, 3898, 2.814218282699585, 3
    87, 1198, 3.103630304336548, 2
    609, 2964, 2.994074821472168, 2
    287, 3569, 3.2009639739990234, 3
    475, 406, 3.5739152431488037, 3
    67, 1177, 2.531890392303467, 3
    274, 1712, 4.029116630554199, 3
    483, 1, 3.4832277297973633, 2
    581, 7756, 3.487367630004883, 4
    61, 4791, 4.165918350219727, 5
    526, 2271, 3.4560680389404297, 4
    287, 988, 3.1495096683502197, 2
    393, 314, 3.751216173171997, 4
    297, 2884, 1.744710922241211, 2
    413, 40, 2.9544806480407715, 2
    17, 8059, 2.825352668762207, 3
    521, 2941, 3.338344097137451, 3
    31, 485, 3.786144256591797, 3
    267, 1660, 3.367006540298462, 3
    316, 1230, 3.4412527084350586, 3
    379, 2036, 3.6073436737060547, 3
    218, 2218, 2.9109201431274414, 3
    173, 418, 3.9628703594207764, 5
    200, 1059, 3.882068634033203, 2
    589, 3226, 2.6088247299194336, 2
    90, 287, 3.356846332550049, 2
    421, 2224, 4.036163330078125, 4
    520, 92, 3.6267666816711426, 3
    88, 7196, 3.0140585899353027, 1
    566, 314, 2.7184455394744873, 3
    605, 502, 3.679044008255005, 4
    289, 1531, 4.203693866729736, 5
    551, 1104, 2.782482385635376, 3
    592, 2144, 3.5361685752868652, 4
    413, 6343, 2.7713193893432617, 1
    29, 97, 4.138844966888428, 5
    383, 3201, 2.988861083984375, 4
    598, 9512, 1.940841555595398, 2
    447, 7707, 2.9347548484802246, 2
    488, 4792, 2.47951602935791, 0
    387, 3740, 3.4499411582946777, 4
    379, 7260, 3.327251434326172, 5
    609, 2273, 3.1666557788848877, 4
    138, 7195, 2.104205846786499, 1
    18, 915, 3.152602195739746, 3
    325, 9053, 3.8986778259277344, 4
    62, 8407, 3.5920376777648926, 3
    572, 7073, 3.6146910190582275, 5
    465, 5156, 3.8605570793151855, 4
    3, 2979, 3.743739604949951, 4
    379, 4155, 3.9336321353912354, 4
    216, 173, 2.5812251567840576, 3
    61, 6469, 3.3519275188446045, 1
    160, 1740, 3.6604247093200684, 3
    107, 4187, 3.598062038421631, 4
    333, 1073, 3.5080161094665527, 3
    451, 1471, 4.616232872009277, 5
    209, 9704, 3.454890251159668, 1
    451, 2832, 4.61484432220459, 4
    214, 4355, 3.4972310066223145, 3
    243, 334, 3.3379297256469727, 4
    367, 221, 2.6516175270080566, 3
    372, 398, 4.029810905456543, 3
    413, 3405, 3.0960593223571777, 1
    421, 1534, 3.329120635986328, 4
    479, 6283, 2.75888729095459, 4
    605, 4956, 3.684554100036621, 4
    181, 2832, 3.5125794410705566, 2
    483, 1290, 3.474562406539917, 3
    176, 226, 3.3834357261657715, 4
    509, 960, 2.9308412075042725, 4
    598, 9328, 2.4380910396575928, 1
    110, 6469, 2.879260540008545, 2
    49, 9205, 2.701629638671875, 2
    233, 1130, 3.55110239982605, 4
    474, 7675, 4.302664756774902, 4
    559, 2144, 3.596277952194214, 4
    509, 1959, 3.080739736557007, 4
    166, 898, 3.9514479637145996, 4
    104, 6729, 3.8996286392211914, 3
    479, 4090, 2.731938123703003, 2
    248, 8339, 3.4515790939331055, 4
    303, 1153, 3.6949548721313477, 2
    73, 6227, 4.220673084259033, 4
    139, 4256, 3.088402032852173, 3
    508, 8354, 3.0059316158294678, 3
    218, 1692, 2.8717055320739746, 3
    345, 5549, 3.839576244354248, 4
    565, 51, 3.48643159866333, 3
    605, 2999, 3.8812124729156494, 4
    447, 412, 3.014686346054077, 4
    233, 546, 3.765824794769287, 4
    599, 2133, 2.807971715927124, 1
    8, 3633, 3.5926146507263184, 5
    116, 314, 4.096971035003662, 4
    419, 1725, 3.3535866737365723, 4
    510, 7863, 3.402642011642456, 4
    478, 1852, 3.2813849449157715, 2
    308, 734, 3.2110443115234375, 2
    83, 957, 3.6536056995391846, 4
    259, 5015, 3.2805075645446777, 5
    447, 6999, 2.5232760906219482, 2
    451, 314, 5.0553131103515625, 5
    473, 2953, 3.0297279357910156, 3
    181, 989, 3.5808138847351074, 3
    61, 8484, 3.44648814201355, 4
    598, 4624, 2.317544937133789, 2
    379, 4631, 3.782331943511963, 3
    513, 3859, 3.0115394592285156, 2
    317, 6305, 3.3156609535217285, 3
    589, 1961, 3.0100035667419434, 3
    72, 3623, 3.542269229888916, 5
    43, 654, 2.6827540397644043, 3
    572, 793, 3.959989070892334, 5
    437, 1823, 3.0466413497924805, 4
    473, 3858, 2.5994322299957275, 1
    181, 3185, 3.211132764816284, 3
    271, 6416, 3.2744946479797363, 3
    605, 4672, 3.570913314819336, 3
    32, 890, 3.8608264923095703, 4
    363, 0, 4.261884689331055, 5
    609, 8229, 2.9529759883880615, 4
    537, 2650, 4.281890392303467, 4
    253, 5355, 3.5717310905456543, 4
    92, 1260, 3.702563762664795, 4
    606, 2347, 3.9551496505737305, 4
    42, 307, 4.581301212310791, 3
    56, 510, 4.0938520431518555, 5
    596, 929, 4.1910271644592285, 5
    95, 659, 4.5447540283203125, 4
    218, 3827, 2.599497079849243, 2
    50, 149, 3.108288526535034, 4
    248, 7050, 3.7837002277374268, 2
    67, 6506, 2.592355728149414, 3
    186, 6027, 2.9385757446289062, 4
    124, 6970, 3.5078203678131104, 2
    197, 144, 3.2483198642730713, 4
    413, 2341, 3.483985424041748, 4
    595, 1785, 3.1804873943328857, 3
    181, 2125, 3.0945091247558594, 3
    553, 1558, 3.8300230503082275, 4
    152, 6854, 1.9573768377304077, 3
    451, 2994, 4.569876194000244, 5
    473, 5220, 3.406346321105957, 3
    31, 191, 3.784433364868164, 4
    602, 2013, 3.083174467086792, 4
    386, 2452, 3.2500252723693848, 2
    598, 4592, 2.6286449432373047, 2
    521, 1072, 3.318999767303467, 4
    333, 592, 2.9967923164367676, 3
    164, 898, 3.9157965183258057, 5
    371, 811, 3.1399505138397217, 3
    131, 818, 3.2073006629943848, 3
    216, 1823, 2.733365535736084, 3
    413, 4260, 3.3269190788269043, 2
    39, 216, 3.5645716190338135, 4
    5, 222, 3.2904694080352783, 3
    451, 2419, 4.2655863761901855, 5
    317, 5927, 3.427557945251465, 3
    216, 2055, 2.659397840499878, 4
    248, 7819, 3.2131214141845703, 3
    482, 7646, 3.024766445159912, 3
    407, 8673, 3.518112897872925, 3
    317, 8396, 2.4637272357940674, 2
    563, 8287, 3.4804129600524902, 3
    553, 1258, 3.8852901458740234, 5
    598, 2936, 2.1550486087799072, 1
    533, 8147, 3.433917999267578, 3
    432, 827, 3.1456549167633057, 4
    431, 3930, 3.0536563396453857, 4
    413, 461, 3.8728437423706055, 4
    447, 59, 2.950216293334961, 2
    524, 3609, 3.5523509979248047, 4
    566, 4876, 1.8724470138549805, 2
    312, 2152, 3.0960114002227783, 5
    412, 2543, 3.496058225631714, 4
    120, 4, 2.789856195449829, 3
    413, 3574, 3.1284778118133545, 4
    291, 337, 3.2193150520324707, 4
    605, 2599, 3.283749580383301, 3
    317, 8885, 3.2776529788970947, 4
    482, 2037, 3.532492160797119, 3
    457, 319, 3.6895859241485596, 3
    309, 202, 3.3626723289489746, 0
    27, 5294, 2.7601630687713623, 3
    306, 6453, 1.7472134828567505, 3
    589, 2308, 3.340536117553711, 4
    273, 1121, 2.828357219696045, 3
    209, 6453, 3.034231662750244, 4
    311, 835, 3.730903148651123, 5
    38, 2077, 4.2999267578125, 5
    609, 4927, 3.1615207195281982, 3
    350, 6868, 3.7671594619750977, 3
    609, 8377, 3.607653856277466, 4
    386, 4415, 2.831019401550293, 4
    368, 1515, 2.3951539993286133, 3
    231, 222, 3.1673216819763184, 3
    11, 1869, 3.8434581756591797, 5
    386, 2443, 2.6940488815307617, 1
    215, 2473, 3.5980448722839355, 4
    367, 1235, 2.6908271312713623, 3
    464, 1031, 3.689866542816162, 5
    297, 2940, 2.187272071838379, 2
    476, 6014, 3.4805314540863037, 4
    551, 3557, 3.275653600692749, 5
    559, 5324, 3.4327518939971924, 4
    333, 6534, 3.1812562942504883, 3
    503, 334, 3.404256820678711, 4
    554, 2085, 3.3566884994506836, 5
    224, 810, 3.417288303375244, 4
    268, 645, 3.3195571899414062, 4
    602, 2557, 2.9269955158233643, 3
    245, 136, 4.027796268463135, 4
    88, 6520, 3.360642671585083, 2
    273, 483, 3.14481520652771, 3
    345, 254, 3.7963576316833496, 4
    593, 3785, 3.2379953861236572, 3
    231, 6096, 2.8605751991271973, 3
    605, 2618, 3.1669256687164307, 4
    379, 3086, 3.566870927810669, 3
    182, 2666, 3.05772066116333, 2
    173, 35, 3.6324915885925293, 4
    524, 3185, 3.148388385772705, 4
    183, 9719, 2.919426918029785, 4
    21, 6388, 2.7564077377319336, 1
    83, 966, 3.309046983718872, 3
    195, 914, 3.2811007499694824, 4
    379, 9341, 3.4307384490966797, 3
    63, 914, 3.643151044845581, 4
    103, 3303, 3.5291748046875, 4
    476, 190, 3.752924680709839, 4
    331, 4787, 3.4092860221862793, 4
    473, 2515, 2.860246181488037, 3
    499, 2660, 3.2030813694000244, 4
    379, 1043, 4.007531642913818, 5
    589, 375, 3.358548164367676, 4
    590, 1938, 3.5320730209350586, 5
    238, 2026, 3.3973207473754883, 2
    473, 4992, 3.0099599361419678, 4
    609, 9339, 3.073179244995117, 4
    197, 92, 3.593416452407837, 3
    142, 1601, 3.369173288345337, 5
    598, 672, 2.367064952850342, 1
    248, 8111, 3.3958990573883057, 4
    508, 7070, 2.9954006671905518, 3
    65, 474, 4.153124809265137, 5
    592, 921, 3.6433427333831787, 4
    461, 4794, 3.3663408756256104, 4
    159, 989, 3.0882723331451416, 3
    273, 3700, 2.9713759422302246, 4
    225, 4421, 3.305941581726074, 4
    187, 1833, 4.09080696105957, 5
    329, 3781, 3.0971693992614746, 4
    598, 454, 1.869157075881958, 2
    344, 1913, 3.318415403366089, 4
    273, 7022, 3.1120550632476807, 3
    461, 2274, 3.1238174438476562, 2
    97, 8681, 3.7770183086395264, 5
    331, 507, 3.5426111221313477, 3
    265, 1930, 3.4042861461639404, 4
    598, 2885, 2.2486157417297363, 2
    598, 854, 2.0373177528381348, 2
    508, 1153, 2.9535059928894043, 3
    68, 910, 4.702793121337891, 5
    17, 4842, 3.3193607330322266, 4
    479, 1370, 2.8053767681121826, 2
    429, 2302, 2.82153058052063, 1
    357, 7666, 2.93430233001709, 0
    371, 900, 3.6986231803894043, 3
    561, 5955, 3.8800530433654785, 3
    231, 5987, 2.9043095111846924, 2
    303, 1725, 3.4501256942749023, 3
    508, 3740, 3.052663803100586, 3
    264, 1053, 3.190114974975586, 3
    48, 901, 3.7436037063598633, 4
    561, 6992, 3.677889823913574, 3
    494, 8287, 3.7491304874420166, 5
    514, 8677, 3.8925793170928955, 5
    404, 2734, 3.2425289154052734, 5
    437, 6045, 3.4120023250579834, 4
    473, 957, 3.4125564098358154, 4
    194, 1266, 3.4336445331573486, 4
    296, 510, 3.4177565574645996, 4
    273, 197, 2.544154644012451, 4
    317, 6058, 3.6460413932800293, 3
    316, 3539, 3.5801501274108887, 3
    317, 8665, 3.4700076580047607, 3
    384, 998, 3.5210628509521484, 3
    65, 1397, 3.2842910289764404, 3
    356, 4136, 3.5640740394592285, 2
    607, 3097, 2.6074559688568115, 3
    369, 1972, 2.939484119415283, 4
    367, 2580, 2.3801722526550293, 3
    276, 549, 3.398519992828369, 4
    121, 8304, 4.605113983154297, 4
    5, 472, 3.257230520248413, 3
    313, 193, 2.827442169189453, 1
    10, 1275, 3.549349069595337, 5
    451, 1242, 4.667963027954102, 5
    242, 364, 3.445432662963867, 4
    264, 2404, 3.5129024982452393, 3
    304, 920, 3.772322416305542, 5
    306, 4561, 2.726600170135498, 4
    379, 8262, 3.988459587097168, 3
    473, 4782, 3.267129421234131, 3
    354, 682, 3.5782065391540527, 5
    131, 2955, 3.0723507404327393, 3
    83, 1066, 3.889277935028076, 4
    231, 615, 3.0824496746063232, 3
    44, 1873, 3.1600213050842285, 4
    579, 3759, 2.7680606842041016, 3
    473, 337, 3.0792102813720703, 4
    128, 897, 4.064345359802246, 4
    166, 3217, 3.9005088806152344, 3
    554, 2436, 3.5236854553222656, 4
    67, 1492, 2.5272929668426514, 3
    0, 325, 4.381186485290527, 4
    482, 2135, 3.276094436645508, 4
    331, 835, 3.300985813140869, 3
    94, 1242, 4.039930820465088, 5
    447, 2986, 2.571389675140381, 3
    255, 5719, 3.9008126258850098, 4
    447, 95, 2.699974536895752, 4
    563, 8274, 3.1985669136047363, 2
    559, 4640, 3.1439452171325684, 3
    400, 8529, 3.417571544647217, 3
    220, 964, 3.9299139976501465, 4
    533, 4909, 3.3495001792907715, 4
    544, 2006, 3.1260175704956055, 4
    18, 2505, 2.4634084701538086, 2
    598, 1958, 2.258498191833496, 2
    38, 1543, 3.658841609954834, 3
    530, 1860, 3.3916385173797607, 3
    519, 1793, 3.4937894344329834, 3
    328, 659, 3.823336124420166, 2
    248, 6592, 3.432537794113159, 3
    473, 5421, 2.7595126628875732, 3
    413, 4930, 3.1757214069366455, 3
    541, 835, 3.4255316257476807, 4
    413, 1612, 3.4068894386291504, 3
    22, 947, 3.341298818588257, 3
    67, 2887, 2.7223799228668213, 2
    433, 197, 2.928010940551758, 4
    83, 1073, 3.8437318801879883, 4
    


```python
with torch.no_grad():
  precisions = dict()
  recalls = dict()

  k = 100
  threshold = 3.5

  for uid, user_ratings in user_est_true.items():

    # Sort user ratings by estimated value
    user_ratings.sort(key = lambda x : x[0], reverse = True)

    n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

    n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

    n_rel_and_rec_k = sum(
        ((true_r >= threshold) and (est >= threshold))
        for (est, true_r) in user_ratings[:k]
    )
    print(f'uid {uid}, n_rel {n_rel}, n_rec_k {n_rec_k}, n_rel_and_rec_k {n_rel_and_rec_k}')

    # Precision@K: Proportion of recommended items that are relevant
    # When n_rec_k is 0, Precision is undefined. We here set it to 0.

    precisions[uid] = (n_rel_and_rec_k / n_rec_k) if n_rec_k != 0 else 0

    # Recall@K: Proportion of relevant items that are recommended
    # When n_rel is 0, Recall is undefined. We here set it to 0.

    recalls[uid] = (n_rel_and_rec_k / n_rel) if n_rel != 0 else 0
```

    uid 474, n_rel 20, n_rec_k 21, n_rel_and_rec_k 20
    uid 441, n_rel 0, n_rec_k 0, n_rel_and_rec_k 0
    uid 101, n_rel 0, n_rec_k 0, n_rel_and_rec_k 0
    uid 589, n_rel 25, n_rec_k 11, n_rel_and_rec_k 8
    uid 273, n_rel 28, n_rec_k 2, n_rel_and_rec_k 2
    uid 391, n_rel 1, n_rec_k 0, n_rel_and_rec_k 0
    uid 413, n_rel 121, n_rec_k 53, n_rel_and_rec_k 41
    uid 286, n_rel 9, n_rec_k 0, n_rel_and_rec_k 0
    uid 233, n_rel 13, n_rec_k 15, n_rel_and_rec_k 9
    uid 600, n_rel 12, n_rec_k 12, n_rel_and_rec_k 12
    uid 482, n_rel 49, n_rec_k 23, n_rel_and_rec_k 16
    uid 90, n_rel 14, n_rec_k 12, n_rel_and_rec_k 7
    uid 297, n_rel 11, n_rec_k 0, n_rel_and_rec_k 0
    uid 83, n_rel 18, n_rec_k 13, n_rel_and_rec_k 12
    uid 487, n_rel 7, n_rec_k 11, n_rel_and_rec_k 5
    uid 331, n_rel 14, n_rec_k 12, n_rel_and_rec_k 9
    uid 131, n_rel 3, n_rec_k 0, n_rel_and_rec_k 0
    uid 198, n_rel 8, n_rec_k 0, n_rel_and_rec_k 0
    uid 281, n_rel 19, n_rec_k 18, n_rel_and_rec_k 14
    uid 598, n_rel 22, n_rec_k 0, n_rel_and_rec_k 0
    uid 40, n_rel 6, n_rec_k 1, n_rel_and_rec_k 1
    uid 367, n_rel 8, n_rec_k 0, n_rel_and_rec_k 0
    uid 306, n_rel 18, n_rec_k 0, n_rel_and_rec_k 0
    uid 427, n_rel 4, n_rec_k 0, n_rel_and_rec_k 0
    uid 494, n_rel 21, n_rec_k 13, n_rel_and_rec_k 12
    uid 599, n_rel 18, n_rec_k 0, n_rel_and_rec_k 0
    uid 201, n_rel 36, n_rec_k 51, n_rel_and_rec_k 35
    uid 67, n_rel 42, n_rec_k 0, n_rel_and_rec_k 0
    uid 216, n_rel 10, n_rec_k 1, n_rel_and_rec_k 1
    uid 145, n_rel 0, n_rec_k 0, n_rel_and_rec_k 0
    uid 63, n_rel 34, n_rec_k 23, n_rel_and_rec_k 15
    uid 351, n_rel 13, n_rec_k 19, n_rel_and_rec_k 11
    uid 50, n_rel 20, n_rec_k 22, n_rel_and_rec_k 12
    uid 104, n_rel 59, n_rec_k 64, n_rel_and_rec_k 53
    uid 56, n_rel 27, n_rec_k 14, n_rel_and_rec_k 12
    uid 607, n_rel 27, n_rec_k 1, n_rel_and_rec_k 0
    uid 291, n_rel 16, n_rec_k 5, n_rel_and_rec_k 3
    uid 37, n_rel 2, n_rec_k 1, n_rel_and_rec_k 0
    uid 447, n_rel 56, n_rec_k 2, n_rel_and_rec_k 2
    uid 103, n_rel 12, n_rec_k 16, n_rel_and_rec_k 8
    uid 289, n_rel 20, n_rec_k 23, n_rel_and_rec_k 20
    uid 177, n_rel 3, n_rec_k 5, n_rel_and_rec_k 3
    uid 394, n_rel 0, n_rec_k 0, n_rel_and_rec_k 0
    uid 386, n_rel 31, n_rec_k 10, n_rel_and_rec_k 6
    uid 554, n_rel 26, n_rec_k 39, n_rel_and_rec_k 16
    uid 566, n_rel 2, n_rec_k 0, n_rel_and_rec_k 0
    uid 115, n_rel 6, n_rec_k 2, n_rel_and_rec_k 1
    uid 307, n_rel 3, n_rec_k 0, n_rel_and_rec_k 0
    uid 278, n_rel 11, n_rec_k 7, n_rel_and_rec_k 5
    uid 158, n_rel 9, n_rec_k 1, n_rel_and_rec_k 1
    uid 513, n_rel 22, n_rec_k 2, n_rel_and_rec_k 2
    uid 252, n_rel 5, n_rec_k 2, n_rel_and_rec_k 2
    uid 347, n_rel 8, n_rec_k 8, n_rel_and_rec_k 8
    uid 70, n_rel 1, n_rec_k 0, n_rel_and_rec_k 0
    uid 559, n_rel 22, n_rec_k 4, n_rel_and_rec_k 4
    uid 19, n_rel 8, n_rec_k 16, n_rel_and_rec_k 8
    uid 433, n_rel 14, n_rec_k 11, n_rel_and_rec_k 11
    uid 468, n_rel 29, n_rec_k 34, n_rel_and_rec_k 24
    uid 134, n_rel 16, n_rec_k 15, n_rel_and_rec_k 10
    uid 369, n_rel 4, n_rec_k 2, n_rel_and_rec_k 2
    uid 437, n_rel 24, n_rec_k 8, n_rel_and_rec_k 7
    uid 112, n_rel 8, n_rec_k 9, n_rel_and_rec_k 7
    uid 221, n_rel 6, n_rec_k 1, n_rel_and_rec_k 0
    uid 114, n_rel 6, n_rec_k 3, n_rel_and_rec_k 2
    uid 452, n_rel 23, n_rec_k 21, n_rel_and_rec_k 18
    uid 407, n_rel 11, n_rec_k 12, n_rel_and_rec_k 9
    uid 95, n_rel 3, n_rec_k 6, n_rel_and_rec_k 3
    uid 176, n_rel 33, n_rec_k 9, n_rel_and_rec_k 5
    uid 304, n_rel 56, n_rec_k 56, n_rel_and_rec_k 43
    uid 265, n_rel 14, n_rec_k 12, n_rel_and_rec_k 11
    uid 446, n_rel 4, n_rec_k 5, n_rel_and_rec_k 3
    uid 605, n_rel 65, n_rec_k 58, n_rel_and_rec_k 33
    uid 274, n_rel 21, n_rec_k 28, n_rel_and_rec_k 21
    uid 385, n_rel 1, n_rec_k 0, n_rel_and_rec_k 0
    uid 379, n_rel 66, n_rec_k 100, n_rel_and_rec_k 54
    uid 293, n_rel 9, n_rec_k 0, n_rel_and_rec_k 0
    uid 522, n_rel 7, n_rec_k 7, n_rel_and_rec_k 7
    uid 524, n_rel 18, n_rec_k 13, n_rel_and_rec_k 6
    uid 159, n_rel 20, n_rec_k 0, n_rel_and_rec_k 0
    uid 596, n_rel 24, n_rec_k 33, n_rel_and_rec_k 24
    uid 9, n_rel 2, n_rec_k 0, n_rel_and_rec_k 0
    uid 390, n_rel 31, n_rec_k 42, n_rel_and_rec_k 31
    uid 272, n_rel 3, n_rec_k 4, n_rel_and_rec_k 2
    uid 508, n_rel 9, n_rec_k 1, n_rel_and_rec_k 0
    uid 68, n_rel 5, n_rec_k 8, n_rel_and_rec_k 5
    uid 194, n_rel 11, n_rec_k 6, n_rel_and_rec_k 5
    uid 264, n_rel 6, n_rec_k 4, n_rel_and_rec_k 3
    uid 185, n_rel 21, n_rec_k 24, n_rel_and_rec_k 21
    uid 327, n_rel 9, n_rec_k 5, n_rel_and_rec_k 3
    uid 225, n_rel 18, n_rec_k 6, n_rel_and_rec_k 6
    uid 436, n_rel 7, n_rec_k 8, n_rel_and_rec_k 5
    uid 380, n_rel 23, n_rec_k 11, n_rel_and_rec_k 8
    uid 110, n_rel 27, n_rec_k 4, n_rel_and_rec_k 2
    uid 419, n_rel 7, n_rec_k 11, n_rel_and_rec_k 5
    uid 94, n_rel 14, n_rec_k 14, n_rel_and_rec_k 12
    uid 41, n_rel 26, n_rec_k 26, n_rel_and_rec_k 19
    uid 609, n_rel 70, n_rec_k 23, n_rel_and_rec_k 20
    uid 303, n_rel 19, n_rec_k 17, n_rel_and_rec_k 10
    uid 540, n_rel 6, n_rec_k 6, n_rel_and_rec_k 5
    uid 138, n_rel 0, n_rec_k 0, n_rel_and_rec_k 0
    uid 259, n_rel 10, n_rec_k 9, n_rel_and_rec_k 6
    uid 409, n_rel 10, n_rec_k 12, n_rel_and_rec_k 9
    uid 400, n_rel 3, n_rec_k 0, n_rel_and_rec_k 0
    uid 18, n_rel 7, n_rec_k 0, n_rel_and_rec_k 0
    uid 461, n_rel 24, n_rec_k 5, n_rel_and_rec_k 4
    uid 473, n_rel 74, n_rec_k 11, n_rel_and_rec_k 9
    uid 516, n_rel 8, n_rec_k 0, n_rel_and_rec_k 0
    uid 381, n_rel 10, n_rec_k 8, n_rel_and_rec_k 7
    uid 166, n_rel 15, n_rec_k 17, n_rel_and_rec_k 11
    uid 478, n_rel 8, n_rec_k 3, n_rel_and_rec_k 2
    uid 74, n_rel 2, n_rec_k 0, n_rel_and_rec_k 0
    uid 226, n_rel 9, n_rec_k 13, n_rel_and_rec_k 9
    uid 361, n_rel 10, n_rec_k 11, n_rel_and_rec_k 9
    uid 578, n_rel 7, n_rec_k 4, n_rel_and_rec_k 4
    uid 181, n_rel 58, n_rec_k 21, n_rel_and_rec_k 13
    uid 152, n_rel 4, n_rec_k 0, n_rel_and_rec_k 0
    uid 248, n_rel 56, n_rec_k 43, n_rel_and_rec_k 34
    uid 483, n_rel 19, n_rec_k 23, n_rel_and_rec_k 13
    uid 183, n_rel 11, n_rec_k 4, n_rel_and_rec_k 4
    uid 155, n_rel 21, n_rec_k 29, n_rel_and_rec_k 18
    uid 410, n_rel 7, n_rec_k 2, n_rel_and_rec_k 2
    uid 21, n_rel 2, n_rec_k 0, n_rel_and_rec_k 0
    uid 72, n_rel 14, n_rec_k 10, n_rel_and_rec_k 7
    uid 476, n_rel 30, n_rec_k 16, n_rel_and_rec_k 12
    uid 376, n_rel 8, n_rec_k 10, n_rel_and_rec_k 7
    uid 102, n_rel 20, n_rec_k 15, n_rel_and_rec_k 14
    uid 108, n_rel 1, n_rec_k 0, n_rel_and_rec_k 0
    uid 511, n_rel 0, n_rec_k 0, n_rel_and_rec_k 0
    uid 128, n_rel 11, n_rec_k 9, n_rel_and_rec_k 7
    uid 488, n_rel 14, n_rec_k 0, n_rel_and_rec_k 0
    uid 22, n_rel 3, n_rec_k 8, n_rel_and_rec_k 2
    uid 231, n_rel 24, n_rec_k 7, n_rel_and_rec_k 6
    uid 316, n_rel 14, n_rec_k 18, n_rel_and_rec_k 12
    uid 287, n_rel 20, n_rec_k 5, n_rel_and_rec_k 3
    uid 150, n_rel 5, n_rec_k 2, n_rel_and_rec_k 1
    uid 505, n_rel 1, n_rec_k 2, n_rel_and_rec_k 1
    uid 414, n_rel 8, n_rec_k 7, n_rel_and_rec_k 6
    uid 245, n_rel 14, n_rec_k 18, n_rel_and_rec_k 14
    uid 140, n_rel 4, n_rec_k 2, n_rel_and_rec_k 2
    uid 49, n_rel 5, n_rec_k 0, n_rel_and_rec_k 0
    uid 384, n_rel 7, n_rec_k 9, n_rel_and_rec_k 4
    uid 345, n_rel 12, n_rec_k 15, n_rel_and_rec_k 7
    uid 220, n_rel 24, n_rec_k 27, n_rel_and_rec_k 21
    uid 553, n_rel 9, n_rec_k 15, n_rel_and_rec_k 9
    uid 481, n_rel 11, n_rec_k 5, n_rel_and_rec_k 4
    uid 174, n_rel 2, n_rec_k 0, n_rel_and_rec_k 0
    uid 26, n_rel 7, n_rec_k 8, n_rel_and_rec_k 5
    uid 246, n_rel 9, n_rec_k 12, n_rel_and_rec_k 7
    uid 13, n_rel 1, n_rec_k 1, n_rel_and_rec_k 1
    uid 60, n_rel 5, n_rec_k 6, n_rel_and_rec_k 5
    uid 355, n_rel 20, n_rec_k 27, n_rel_and_rec_k 19
    uid 255, n_rel 19, n_rec_k 17, n_rel_and_rec_k 13
    uid 525, n_rel 2, n_rec_k 3, n_rel_and_rec_k 2
    uid 311, n_rel 17, n_rec_k 15, n_rel_and_rec_k 14
    uid 550, n_rel 9, n_rec_k 2, n_rel_and_rec_k 2
    uid 341, n_rel 1, n_rec_k 0, n_rel_and_rec_k 0
    uid 398, n_rel 4, n_rec_k 6, n_rel_and_rec_k 4
    uid 356, n_rel 23, n_rec_k 29, n_rel_and_rec_k 18
    uid 118, n_rel 16, n_rec_k 21, n_rel_and_rec_k 15
    uid 418, n_rel 5, n_rec_k 8, n_rel_and_rec_k 3
    uid 324, n_rel 20, n_rec_k 11, n_rel_and_rec_k 8
    uid 317, n_rel 38, n_rec_k 32, n_rel_and_rec_k 14
    uid 435, n_rel 4, n_rec_k 2, n_rel_and_rec_k 1
    uid 469, n_rel 4, n_rec_k 4, n_rel_and_rec_k 3
    uid 109, n_rel 1, n_rec_k 2, n_rel_and_rec_k 0
    uid 106, n_rel 3, n_rec_k 0, n_rel_and_rec_k 0
    uid 85, n_rel 3, n_rec_k 2, n_rel_and_rec_k 2
    uid 238, n_rel 20, n_rec_k 21, n_rel_and_rec_k 18
    uid 262, n_rel 14, n_rec_k 22, n_rel_and_rec_k 12
    uid 604, n_rel 6, n_rec_k 2, n_rel_and_rec_k 1
    uid 541, n_rel 7, n_rec_k 6, n_rel_and_rec_k 2
    uid 479, n_rel 29, n_rec_k 6, n_rel_and_rec_k 4
    uid 509, n_rel 2, n_rec_k 0, n_rel_and_rec_k 0
    uid 551, n_rel 10, n_rec_k 1, n_rel_and_rec_k 1
    uid 442, n_rel 6, n_rec_k 6, n_rel_and_rec_k 5
    uid 279, n_rel 9, n_rec_k 14, n_rel_and_rec_k 8
    uid 576, n_rel 14, n_rec_k 5, n_rel_and_rec_k 3
    uid 136, n_rel 9, n_rec_k 12, n_rel_and_rec_k 8
    uid 563, n_rel 9, n_rec_k 0, n_rel_and_rec_k 0
    uid 35, n_rel 0, n_rec_k 0, n_rel_and_rec_k 0
    uid 224, n_rel 3, n_rec_k 4, n_rel_and_rec_k 2
    uid 168, n_rel 24, n_rec_k 21, n_rel_and_rec_k 21
    uid 27, n_rel 12, n_rec_k 0, n_rel_and_rec_k 0
    uid 285, n_rel 4, n_rec_k 3, n_rel_and_rec_k 3
    uid 586, n_rel 10, n_rec_k 12, n_rel_and_rec_k 10
    uid 321, n_rel 6, n_rec_k 0, n_rel_and_rec_k 0
    uid 65, n_rel 23, n_rec_k 22, n_rel_and_rec_k 16
    uid 282, n_rel 4, n_rec_k 4, n_rel_and_rec_k 4
    uid 305, n_rel 4, n_rec_k 0, n_rel_and_rec_k 0
    uid 338, n_rel 22, n_rec_k 33, n_rel_and_rec_k 21
    uid 99, n_rel 13, n_rec_k 8, n_rel_and_rec_k 6
    uid 570, n_rel 3, n_rec_k 0, n_rel_and_rec_k 0
    uid 326, n_rel 4, n_rec_k 7, n_rel_and_rec_k 3
    uid 560, n_rel 10, n_rec_k 4, n_rel_and_rec_k 2
    uid 595, n_rel 15, n_rec_k 4, n_rel_and_rec_k 3
    uid 204, n_rel 2, n_rec_k 2, n_rel_and_rec_k 2
    uid 519, n_rel 13, n_rec_k 15, n_rel_and_rec_k 12
    uid 353, n_rel 11, n_rec_k 12, n_rel_and_rec_k 8
    uid 130, n_rel 3, n_rec_k 1, n_rel_and_rec_k 0
    uid 312, n_rel 18, n_rec_k 7, n_rel_and_rec_k 7
    uid 295, n_rel 3, n_rec_k 1, n_rel_and_rec_k 1
    uid 352, n_rel 4, n_rec_k 3, n_rel_and_rec_k 3
    uid 54, n_rel 1, n_rec_k 0, n_rel_and_rec_k 0
    uid 308, n_rel 10, n_rec_k 3, n_rel_and_rec_k 2
    uid 6, n_rel 5, n_rec_k 0, n_rel_and_rec_k 0
    uid 31, n_rel 7, n_rec_k 9, n_rel_and_rec_k 6
    uid 378, n_rel 2, n_rec_k 3, n_rel_and_rec_k 1
    uid 465, n_rel 10, n_rec_k 8, n_rel_and_rec_k 8
    uid 555, n_rel 4, n_rec_k 3, n_rel_and_rec_k 3
    uid 180, n_rel 1, n_rec_k 0, n_rel_and_rec_k 0
    uid 121, n_rel 27, n_rec_k 27, n_rel_and_rec_k 27
    uid 593, n_rel 21, n_rec_k 12, n_rel_and_rec_k 11
    uid 499, n_rel 2, n_rec_k 2, n_rel_and_rec_k 1
    uid 148, n_rel 5, n_rec_k 1, n_rel_and_rec_k 1
    uid 199, n_rel 21, n_rec_k 23, n_rel_and_rec_k 19
    uid 526, n_rel 13, n_rec_k 12, n_rel_and_rec_k 9
    uid 5, n_rel 16, n_rec_k 7, n_rel_and_rec_k 5
    uid 46, n_rel 5, n_rec_k 0, n_rel_and_rec_k 0
    uid 151, n_rel 6, n_rec_k 9, n_rel_and_rec_k 5
    uid 218, n_rel 13, n_rec_k 1, n_rel_and_rec_k 0
    uid 197, n_rel 12, n_rec_k 12, n_rel_and_rec_k 6
    uid 486, n_rel 2, n_rec_k 2, n_rel_and_rec_k 2
    uid 510, n_rel 4, n_rec_k 3, n_rel_and_rec_k 3
    uid 408, n_rel 2, n_rec_k 6, n_rel_and_rec_k 2
    uid 584, n_rel 5, n_rec_k 5, n_rel_and_rec_k 5
    uid 116, n_rel 4, n_rec_k 10, n_rel_and_rec_k 4
    uid 162, n_rel 1, n_rec_k 0, n_rel_and_rec_k 0
    uid 232, n_rel 1, n_rec_k 3, n_rel_and_rec_k 1
    uid 368, n_rel 5, n_rec_k 1, n_rel_and_rec_k 0
    uid 283, n_rel 4, n_rec_k 3, n_rel_and_rec_k 2
    uid 125, n_rel 2, n_rec_k 0, n_rel_and_rec_k 0
    uid 167, n_rel 7, n_rec_k 8, n_rel_and_rec_k 7
    uid 0, n_rel 24, n_rec_k 27, n_rel_and_rec_k 24
    uid 203, n_rel 6, n_rec_k 6, n_rel_and_rec_k 5
    uid 602, n_rel 60, n_rec_k 24, n_rel_and_rec_k 19
    uid 270, n_rel 0, n_rec_k 1, n_rel_and_rec_k 0
    uid 20, n_rel 11, n_rec_k 0, n_rel_and_rec_k 0
    uid 124, n_rel 21, n_rec_k 27, n_rel_and_rec_k 14
    uid 44, n_rel 20, n_rec_k 27, n_rel_and_rec_k 15
    uid 328, n_rel 1, n_rec_k 2, n_rel_and_rec_k 0
    uid 572, n_rel 24, n_rec_k 21, n_rel_and_rec_k 16
    uid 81, n_rel 6, n_rec_k 3, n_rel_and_rec_k 2
    uid 165, n_rel 19, n_rec_k 12, n_rel_and_rec_k 12
    uid 43, n_rel 4, n_rec_k 0, n_rel_and_rec_k 0
    uid 360, n_rel 2, n_rec_k 0, n_rel_and_rec_k 0
    uid 16, n_rel 11, n_rec_k 13, n_rel_and_rec_k 11
    uid 579, n_rel 22, n_rec_k 2, n_rel_and_rec_k 2
    uid 17, n_rel 29, n_rec_k 8, n_rel_and_rec_k 6
    uid 88, n_rel 25, n_rec_k 3, n_rel_and_rec_k 1
    uid 290, n_rel 4, n_rec_k 4, n_rel_and_rec_k 4
    uid 498, n_rel 3, n_rec_k 3, n_rel_and_rec_k 3
    uid 236, n_rel 2, n_rec_k 0, n_rel_and_rec_k 0
    uid 549, n_rel 1, n_rec_k 2, n_rel_and_rec_k 1
    uid 561, n_rel 15, n_rec_k 24, n_rel_and_rec_k 15
    uid 61, n_rel 25, n_rec_k 23, n_rel_and_rec_k 20
    uid 366, n_rel 20, n_rec_k 24, n_rel_and_rec_k 19
    uid 585, n_rel 11, n_rec_k 12, n_rel_and_rec_k 11
    uid 533, n_rel 19, n_rec_k 15, n_rel_and_rec_k 9
    uid 209, n_rel 14, n_rec_k 12, n_rel_and_rec_k 10
    uid 591, n_rel 6, n_rec_k 3, n_rel_and_rec_k 1
    uid 139, n_rel 19, n_rec_k 19, n_rel_and_rec_k 10
    uid 364, n_rel 4, n_rec_k 0, n_rel_and_rec_k 0
    uid 79, n_rel 10, n_rec_k 11, n_rel_and_rec_k 10
    uid 211, n_rel 12, n_rec_k 11, n_rel_and_rec_k 2
    uid 539, n_rel 3, n_rec_k 3, n_rel_and_rec_k 3
    uid 343, n_rel 5, n_rec_k 3, n_rel_and_rec_k 3
    uid 253, n_rel 10, n_rec_k 10, n_rel_and_rec_k 9
    uid 431, n_rel 16, n_rec_k 4, n_rel_and_rec_k 4
    uid 38, n_rel 6, n_rec_k 10, n_rel_and_rec_k 6
    uid 336, n_rel 9, n_rec_k 10, n_rel_and_rec_k 9
    uid 358, n_rel 5, n_rec_k 2, n_rel_and_rec_k 2
    uid 325, n_rel 16, n_rec_k 14, n_rel_and_rec_k 12
    uid 387, n_rel 1, n_rec_k 0, n_rel_and_rec_k 0
    uid 322, n_rel 2, n_rec_k 0, n_rel_and_rec_k 0
    uid 87, n_rel 5, n_rec_k 3, n_rel_and_rec_k 3
    uid 451, n_rel 27, n_rec_k 27, n_rel_and_rec_k 27
    uid 371, n_rel 9, n_rec_k 6, n_rel_and_rec_k 3
    uid 496, n_rel 3, n_rec_k 4, n_rel_and_rec_k 2
    uid 318, n_rel 6, n_rec_k 6, n_rel_and_rec_k 6
    uid 213, n_rel 0, n_rec_k 0, n_rel_and_rec_k 0
    uid 485, n_rel 3, n_rec_k 5, n_rel_and_rec_k 3
    uid 73, n_rel 15, n_rec_k 18, n_rel_and_rec_k 15
    uid 42, n_rel 8, n_rec_k 9, n_rel_and_rec_k 8
    uid 57, n_rel 3, n_rec_k 4, n_rel_and_rec_k 3
    uid 39, n_rel 10, n_rec_k 8, n_rel_and_rec_k 7
    uid 169, n_rel 1, n_rec_k 1, n_rel_and_rec_k 1
    uid 62, n_rel 16, n_rec_k 22, n_rel_and_rec_k 12
    uid 296, n_rel 3, n_rec_k 0, n_rel_and_rec_k 0
    uid 606, n_rel 12, n_rec_k 16, n_rel_and_rec_k 12
    uid 75, n_rel 6, n_rec_k 0, n_rel_and_rec_k 0
    uid 14, n_rel 9, n_rec_k 7, n_rel_and_rec_k 5
    uid 288, n_rel 1, n_rec_k 0, n_rel_and_rec_k 0
    uid 477, n_rel 1, n_rec_k 0, n_rel_and_rec_k 0
    uid 492, n_rel 3, n_rec_k 4, n_rel_and_rec_k 2
    uid 428, n_rel 5, n_rec_k 6, n_rel_and_rec_k 5
    uid 424, n_rel 9, n_rec_k 9, n_rel_and_rec_k 5
    uid 36, n_rel 2, n_rec_k 1, n_rel_and_rec_k 1
    uid 489, n_rel 3, n_rec_k 1, n_rel_and_rec_k 0
    uid 300, n_rel 2, n_rec_k 2, n_rel_and_rec_k 2
    uid 315, n_rel 0, n_rec_k 0, n_rel_and_rec_k 0
    uid 313, n_rel 3, n_rec_k 0, n_rel_and_rec_k 0
    uid 239, n_rel 7, n_rec_k 11, n_rel_and_rec_k 7
    uid 256, n_rel 1, n_rec_k 0, n_rel_and_rec_k 0
    uid 562, n_rel 2, n_rec_k 0, n_rel_and_rec_k 0
    uid 161, n_rel 3, n_rec_k 4, n_rel_and_rec_k 3
    uid 135, n_rel 4, n_rec_k 5, n_rel_and_rec_k 1
    uid 457, n_rel 6, n_rec_k 7, n_rel_and_rec_k 6
    uid 448, n_rel 3, n_rec_k 0, n_rel_and_rec_k 0
    uid 143, n_rel 10, n_rec_k 13, n_rel_and_rec_k 9
    uid 77, n_rel 4, n_rec_k 0, n_rel_and_rec_k 0
    uid 569, n_rel 6, n_rec_k 5, n_rel_and_rec_k 2
    uid 190, n_rel 6, n_rec_k 9, n_rel_and_rec_k 6
    uid 571, n_rel 15, n_rec_k 18, n_rel_and_rec_k 15
    uid 521, n_rel 13, n_rec_k 11, n_rel_and_rec_k 6
    uid 417, n_rel 4, n_rec_k 6, n_rel_and_rec_k 4
    uid 558, n_rel 2, n_rec_k 2, n_rel_and_rec_k 1
    uid 92, n_rel 9, n_rec_k 9, n_rel_and_rec_k 9
    uid 3, n_rel 15, n_rec_k 20, n_rel_and_rec_k 11
    uid 45, n_rel 3, n_rec_k 3, n_rel_and_rec_k 3
    uid 266, n_rel 6, n_rec_k 7, n_rel_and_rec_k 6
    uid 219, n_rel 15, n_rec_k 21, n_rel_and_rec_k 15
    uid 454, n_rel 2, n_rec_k 2, n_rel_and_rec_k 2
    uid 228, n_rel 8, n_rec_k 5, n_rel_and_rec_k 5
    uid 432, n_rel 2, n_rec_k 0, n_rel_and_rec_k 0
    uid 227, n_rel 2, n_rec_k 1, n_rel_and_rec_k 1
    uid 142, n_rel 5, n_rec_k 4, n_rel_and_rec_k 2
    uid 523, n_rel 10, n_rec_k 1, n_rel_and_rec_k 1
    uid 254, n_rel 0, n_rec_k 0, n_rel_and_rec_k 0
    uid 339, n_rel 2, n_rec_k 2, n_rel_and_rec_k 2
    uid 372, n_rel 2, n_rec_k 3, n_rel_and_rec_k 2
    uid 186, n_rel 18, n_rec_k 16, n_rel_and_rec_k 14
    uid 120, n_rel 5, n_rec_k 4, n_rel_and_rec_k 1
    uid 535, n_rel 0, n_rec_k 1, n_rel_and_rec_k 0
    uid 404, n_rel 13, n_rec_k 9, n_rel_and_rec_k 7
    uid 122, n_rel 4, n_rec_k 6, n_rel_and_rec_k 4
    uid 377, n_rel 3, n_rec_k 1, n_rel_and_rec_k 1
    uid 32, n_rel 8, n_rec_k 14, n_rel_and_rec_k 8
    uid 392, n_rel 10, n_rec_k 12, n_rel_and_rec_k 8
    uid 545, n_rel 7, n_rec_k 0, n_rel_and_rec_k 0
    uid 33, n_rel 3, n_rec_k 1, n_rel_and_rec_k 0
    uid 55, n_rel 0, n_rec_k 4, n_rel_and_rec_k 0
    uid 309, n_rel 2, n_rec_k 1, n_rel_and_rec_k 1
    uid 354, n_rel 4, n_rec_k 1, n_rel_and_rec_k 1
    uid 493, n_rel 2, n_rec_k 1, n_rel_and_rec_k 1
    uid 208, n_rel 3, n_rec_k 4, n_rel_and_rec_k 3
    uid 144, n_rel 1, n_rec_k 1, n_rel_and_rec_k 1
    uid 588, n_rel 4, n_rec_k 5, n_rel_and_rec_k 3
    uid 11, n_rel 3, n_rec_k 3, n_rel_and_rec_k 3
    uid 401, n_rel 4, n_rec_k 3, n_rel_and_rec_k 2
    uid 464, n_rel 12, n_rec_k 13, n_rel_and_rec_k 12
    uid 15, n_rel 3, n_rec_k 9, n_rel_and_rec_k 2
    uid 445, n_rel 1, n_rec_k 0, n_rel_and_rec_k 0
    uid 506, n_rel 0, n_rec_k 0, n_rel_and_rec_k 0
    uid 484, n_rel 1, n_rec_k 4, n_rel_and_rec_k 1
    uid 153, n_rel 3, n_rec_k 4, n_rel_and_rec_k 3
    uid 504, n_rel 1, n_rec_k 1, n_rel_and_rec_k 1
    uid 429, n_rel 4, n_rec_k 2, n_rel_and_rec_k 2
    uid 472, n_rel 0, n_rec_k 2, n_rel_and_rec_k 0
    uid 187, n_rel 6, n_rec_k 6, n_rel_and_rec_k 6
    uid 425, n_rel 7, n_rec_k 5, n_rel_and_rec_k 3
    uid 260, n_rel 4, n_rec_k 4, n_rel_and_rec_k 4
    uid 229, n_rel 1, n_rec_k 0, n_rel_and_rec_k 0
    uid 149, n_rel 0, n_rec_k 0, n_rel_and_rec_k 0
    uid 544, n_rel 3, n_rec_k 0, n_rel_and_rec_k 0
    uid 147, n_rel 4, n_rec_k 2, n_rel_and_rec_k 1
    uid 240, n_rel 7, n_rec_k 3, n_rel_and_rec_k 2
    uid 375, n_rel 6, n_rec_k 8, n_rel_and_rec_k 6
    uid 412, n_rel 10, n_rec_k 8, n_rel_and_rec_k 8
    uid 178, n_rel 2, n_rec_k 1, n_rel_and_rec_k 1
    uid 416, n_rel 6, n_rec_k 6, n_rel_and_rec_k 6
    uid 257, n_rel 3, n_rec_k 4, n_rel_and_rec_k 3
    uid 263, n_rel 4, n_rec_k 10, n_rel_and_rec_k 4
    uid 98, n_rel 3, n_rec_k 8, n_rel_and_rec_k 3
    uid 350, n_rel 4, n_rec_k 5, n_rel_and_rec_k 2
    uid 399, n_rel 8, n_rec_k 8, n_rel_and_rec_k 8
    uid 298, n_rel 1, n_rec_k 2, n_rel_and_rec_k 1
    uid 491, n_rel 7, n_rec_k 9, n_rel_and_rec_k 5
    uid 192, n_rel 2, n_rec_k 1, n_rel_and_rec_k 0
    uid 242, n_rel 3, n_rec_k 3, n_rel_and_rec_k 2
    uid 163, n_rel 3, n_rec_k 2, n_rel_and_rec_k 2
    uid 51, n_rel 10, n_rec_k 12, n_rel_and_rec_k 10
    uid 456, n_rel 1, n_rec_k 6, n_rel_and_rec_k 1
    uid 503, n_rel 4, n_rec_k 4, n_rel_and_rec_k 3
    uid 450, n_rel 4, n_rec_k 5, n_rel_and_rec_k 4
    uid 280, n_rel 2, n_rec_k 0, n_rel_and_rec_k 0
    uid 581, n_rel 5, n_rec_k 3, n_rel_and_rec_k 3
    uid 330, n_rel 8, n_rec_k 6, n_rel_and_rec_k 5
    uid 393, n_rel 2, n_rec_k 2, n_rel_and_rec_k 2
    uid 590, n_rel 2, n_rec_k 1, n_rel_and_rec_k 1
    uid 536, n_rel 6, n_rec_k 7, n_rel_and_rec_k 6
    uid 382, n_rel 3, n_rec_k 3, n_rel_and_rec_k 3
    uid 64, n_rel 2, n_rec_k 1, n_rel_and_rec_k 1
    uid 567, n_rel 3, n_rec_k 4, n_rel_and_rec_k 3
    uid 463, n_rel 8, n_rec_k 7, n_rel_and_rec_k 4
    uid 403, n_rel 1, n_rec_k 3, n_rel_and_rec_k 1
    uid 532, n_rel 7, n_rec_k 5, n_rel_and_rec_k 5
    uid 214, n_rel 6, n_rec_k 6, n_rel_and_rec_k 4
    uid 191, n_rel 1, n_rec_k 2, n_rel_and_rec_k 1
    uid 443, n_rel 2, n_rec_k 1, n_rel_and_rec_k 1
    uid 29, n_rel 4, n_rec_k 5, n_rel_and_rec_k 4
    uid 107, n_rel 8, n_rec_k 9, n_rel_and_rec_k 8
    uid 96, n_rel 4, n_rec_k 4, n_rel_and_rec_k 3
    uid 342, n_rel 5, n_rec_k 7, n_rel_and_rec_k 5
    uid 52, n_rel 5, n_rec_k 5, n_rel_and_rec_k 5
    uid 301, n_rel 3, n_rec_k 4, n_rel_and_rec_k 2
    uid 200, n_rel 10, n_rec_k 12, n_rel_and_rec_k 10
    uid 146, n_rel 3, n_rec_k 0, n_rel_and_rec_k 0
    uid 111, n_rel 7, n_rec_k 4, n_rel_and_rec_k 4
    uid 243, n_rel 4, n_rec_k 2, n_rel_and_rec_k 2
    uid 534, n_rel 0, n_rec_k 0, n_rel_and_rec_k 0
    uid 195, n_rel 5, n_rec_k 0, n_rel_and_rec_k 0
    uid 582, n_rel 4, n_rec_k 0, n_rel_and_rec_k 0
    uid 207, n_rel 1, n_rec_k 0, n_rel_and_rec_k 0
    uid 438, n_rel 3, n_rec_k 3, n_rel_and_rec_k 3
    uid 608, n_rel 2, n_rec_k 1, n_rel_and_rec_k 1
    uid 302, n_rel 2, n_rec_k 2, n_rel_and_rec_k 2
    uid 520, n_rel 3, n_rec_k 4, n_rel_and_rec_k 2
    uid 105, n_rel 3, n_rec_k 3, n_rel_and_rec_k 3
    uid 460, n_rel 3, n_rec_k 0, n_rel_and_rec_k 0
    uid 82, n_rel 7, n_rec_k 0, n_rel_and_rec_k 0
    uid 406, n_rel 2, n_rec_k 2, n_rel_and_rec_k 1
    uid 480, n_rel 0, n_rec_k 0, n_rel_and_rec_k 0
    uid 514, n_rel 2, n_rec_k 2, n_rel_and_rec_k 2
    uid 329, n_rel 14, n_rec_k 5, n_rel_and_rec_k 4
    uid 389, n_rel 8, n_rec_k 4, n_rel_and_rec_k 4
    uid 160, n_rel 3, n_rec_k 1, n_rel_and_rec_k 0
    uid 459, n_rel 7, n_rec_k 8, n_rel_and_rec_k 7
    uid 189, n_rel 4, n_rec_k 6, n_rel_and_rec_k 4
    uid 542, n_rel 6, n_rec_k 6, n_rel_and_rec_k 6
    uid 467, n_rel 1, n_rec_k 1, n_rel_and_rec_k 1
    uid 397, n_rel 4, n_rec_k 4, n_rel_and_rec_k 4
    uid 97, n_rel 6, n_rec_k 4, n_rel_and_rec_k 3
    uid 80, n_rel 1, n_rec_k 1, n_rel_and_rec_k 0
    uid 421, n_rel 7, n_rec_k 6, n_rel_and_rec_k 5
    uid 117, n_rel 0, n_rec_k 0, n_rel_and_rec_k 0
    uid 84, n_rel 2, n_rec_k 0, n_rel_and_rec_k 0
    uid 4, n_rel 4, n_rec_k 4, n_rel_and_rec_k 3
    uid 592, n_rel 5, n_rec_k 4, n_rel_and_rec_k 3
    uid 337, n_rel 4, n_rec_k 0, n_rel_and_rec_k 0
    uid 267, n_rel 4, n_rec_k 2, n_rel_and_rec_k 1
    uid 426, n_rel 3, n_rec_k 0, n_rel_and_rec_k 0
    uid 601, n_rel 7, n_rec_k 3, n_rel_and_rec_k 2
    uid 58, n_rel 6, n_rec_k 6, n_rel_and_rec_k 6
    uid 93, n_rel 1, n_rec_k 0, n_rel_and_rec_k 0
    uid 164, n_rel 2, n_rec_k 1, n_rel_and_rec_k 1
    uid 552, n_rel 5, n_rec_k 8, n_rel_and_rec_k 5
    uid 215, n_rel 11, n_rec_k 13, n_rel_and_rec_k 7
    uid 370, n_rel 4, n_rec_k 4, n_rel_and_rec_k 4
    uid 127, n_rel 3, n_rec_k 4, n_rel_and_rec_k 3
    uid 132, n_rel 2, n_rec_k 1, n_rel_and_rec_k 1
    uid 537, n_rel 4, n_rec_k 4, n_rel_and_rec_k 4
    uid 100, n_rel 2, n_rec_k 1, n_rel_and_rec_k 0
    uid 568, n_rel 3, n_rec_k 3, n_rel_and_rec_k 2
    uid 527, n_rel 4, n_rec_k 5, n_rel_and_rec_k 4
    uid 25, n_rel 0, n_rec_k 1, n_rel_and_rec_k 0
    uid 557, n_rel 3, n_rec_k 4, n_rel_and_rec_k 3
    uid 423, n_rel 5, n_rec_k 7, n_rel_and_rec_k 4
    uid 53, n_rel 0, n_rec_k 0, n_rel_and_rec_k 0
    uid 388, n_rel 1, n_rec_k 1, n_rel_and_rec_k 0
    uid 529, n_rel 3, n_rec_k 2, n_rel_and_rec_k 2
    uid 502, n_rel 1, n_rec_k 3, n_rel_and_rec_k 0
    uid 78, n_rel 4, n_rec_k 6, n_rel_and_rec_k 4
    uid 547, n_rel 1, n_rec_k 2, n_rel_and_rec_k 1
    uid 241, n_rel 1, n_rec_k 4, n_rel_and_rec_k 1
    uid 261, n_rel 1, n_rec_k 0, n_rel_and_rec_k 0
    uid 320, n_rel 0, n_rec_k 0, n_rel_and_rec_k 0
    uid 583, n_rel 7, n_rec_k 5, n_rel_and_rec_k 4
    uid 310, n_rel 1, n_rec_k 0, n_rel_and_rec_k 0
    uid 230, n_rel 2, n_rec_k 2, n_rel_and_rec_k 2
    uid 415, n_rel 2, n_rec_k 0, n_rel_and_rec_k 0
    uid 344, n_rel 5, n_rec_k 4, n_rel_and_rec_k 3
    uid 69, n_rel 2, n_rec_k 3, n_rel_and_rec_k 2
    uid 422, n_rel 2, n_rec_k 1, n_rel_and_rec_k 1
    uid 23, n_rel 3, n_rec_k 3, n_rel_and_rec_k 2
    uid 396, n_rel 1, n_rec_k 1, n_rel_and_rec_k 1
    uid 603, n_rel 5, n_rec_k 3, n_rel_and_rec_k 3
    uid 89, n_rel 5, n_rec_k 5, n_rel_and_rec_k 5
    uid 275, n_rel 1, n_rec_k 3, n_rel_and_rec_k 1
    uid 66, n_rel 2, n_rec_k 2, n_rel_and_rec_k 1
    uid 271, n_rel 1, n_rec_k 3, n_rel_and_rec_k 1
    uid 113, n_rel 0, n_rec_k 2, n_rel_and_rec_k 0
    uid 8, n_rel 5, n_rec_k 2, n_rel_and_rec_k 2
    uid 234, n_rel 3, n_rec_k 3, n_rel_and_rec_k 2
    uid 126, n_rel 1, n_rec_k 1, n_rel_and_rec_k 0
    uid 28, n_rel 7, n_rec_k 7, n_rel_and_rec_k 6
    uid 171, n_rel 4, n_rec_k 1, n_rel_and_rec_k 1
    uid 501, n_rel 2, n_rec_k 1, n_rel_and_rec_k 1
    uid 182, n_rel 1, n_rec_k 1, n_rel_and_rec_k 1
    uid 455, n_rel 2, n_rec_k 0, n_rel_and_rec_k 0
    uid 411, n_rel 7, n_rec_k 8, n_rel_and_rec_k 7
    uid 334, n_rel 2, n_rec_k 2, n_rel_and_rec_k 2
    uid 129, n_rel 2, n_rec_k 3, n_rel_and_rec_k 2
    uid 528, n_rel 0, n_rec_k 0, n_rel_and_rec_k 0
    uid 395, n_rel 1, n_rec_k 0, n_rel_and_rec_k 0
    uid 299, n_rel 3, n_rec_k 1, n_rel_and_rec_k 1
    uid 76, n_rel 3, n_rec_k 4, n_rel_and_rec_k 2
    uid 500, n_rel 2, n_rec_k 1, n_rel_and_rec_k 0
    uid 173, n_rel 5, n_rec_k 5, n_rel_and_rec_k 5
    uid 196, n_rel 1, n_rec_k 1, n_rel_and_rec_k 0
    uid 490, n_rel 5, n_rec_k 6, n_rel_and_rec_k 4
    uid 47, n_rel 4, n_rec_k 6, n_rel_and_rec_k 4
    uid 357, n_rel 0, n_rec_k 0, n_rel_and_rec_k 0
    uid 244, n_rel 1, n_rec_k 0, n_rel_and_rec_k 0
    uid 119, n_rel 2, n_rec_k 1, n_rel_and_rec_k 1
    uid 573, n_rel 1, n_rec_k 1, n_rel_and_rec_k 0
    uid 210, n_rel 5, n_rec_k 4, n_rel_and_rec_k 3
    uid 123, n_rel 5, n_rec_k 4, n_rel_and_rec_k 4
    uid 223, n_rel 3, n_rec_k 4, n_rel_and_rec_k 3
    uid 333, n_rel 2, n_rec_k 3, n_rel_and_rec_k 0
    uid 247, n_rel 3, n_rec_k 2, n_rel_and_rec_k 2
    uid 188, n_rel 3, n_rec_k 3, n_rel_and_rec_k 3
    uid 515, n_rel 1, n_rec_k 1, n_rel_and_rec_k 1
    uid 258, n_rel 1, n_rec_k 1, n_rel_and_rec_k 1
    uid 507, n_rel 2, n_rec_k 0, n_rel_and_rec_k 0
    uid 34, n_rel 1, n_rec_k 0, n_rel_and_rec_k 0
    uid 276, n_rel 1, n_rec_k 1, n_rel_and_rec_k 0
    uid 475, n_rel 3, n_rec_k 3, n_rel_and_rec_k 2
    uid 141, n_rel 1, n_rec_k 1, n_rel_and_rec_k 1
    uid 319, n_rel 1, n_rec_k 1, n_rel_and_rec_k 1
    uid 365, n_rel 2, n_rec_k 2, n_rel_and_rec_k 2
    uid 577, n_rel 2, n_rec_k 0, n_rel_and_rec_k 0
    uid 439, n_rel 1, n_rec_k 3, n_rel_and_rec_k 1
    uid 249, n_rel 0, n_rec_k 0, n_rel_and_rec_k 0
    uid 538, n_rel 1, n_rec_k 1, n_rel_and_rec_k 0
    uid 222, n_rel 1, n_rec_k 1, n_rel_and_rec_k 0
    uid 48, n_rel 2, n_rec_k 2, n_rel_and_rec_k 2
    uid 458, n_rel 2, n_rec_k 2, n_rel_and_rec_k 2
    uid 531, n_rel 4, n_rec_k 6, n_rel_and_rec_k 4
    uid 154, n_rel 1, n_rec_k 3, n_rel_and_rec_k 1
    uid 346, n_rel 1, n_rec_k 2, n_rel_and_rec_k 0
    uid 335, n_rel 3, n_rec_k 3, n_rel_and_rec_k 3
    uid 420, n_rel 6, n_rec_k 5, n_rel_and_rec_k 5
    uid 24, n_rel 3, n_rec_k 3, n_rel_and_rec_k 3
    uid 202, n_rel 3, n_rec_k 3, n_rel_and_rec_k 2
    uid 12, n_rel 4, n_rec_k 1, n_rel_and_rec_k 1
    uid 59, n_rel 1, n_rec_k 1, n_rel_and_rec_k 1
    uid 193, n_rel 0, n_rec_k 1, n_rel_and_rec_k 0
    uid 466, n_rel 0, n_rec_k 1, n_rel_and_rec_k 0
    uid 269, n_rel 3, n_rec_k 0, n_rel_and_rec_k 0
    uid 294, n_rel 1, n_rec_k 1, n_rel_and_rec_k 1
    uid 363, n_rel 3, n_rec_k 3, n_rel_and_rec_k 3
    uid 237, n_rel 1, n_rec_k 3, n_rel_and_rec_k 1
    uid 1, n_rel 0, n_rec_k 1, n_rel_and_rec_k 0
    uid 30, n_rel 2, n_rec_k 2, n_rel_and_rec_k 2
    uid 574, n_rel 0, n_rec_k 1, n_rel_and_rec_k 0
    uid 402, n_rel 1, n_rec_k 0, n_rel_and_rec_k 0
    uid 497, n_rel 3, n_rec_k 5, n_rel_and_rec_k 3
    uid 587, n_rel 0, n_rec_k 0, n_rel_and_rec_k 0
    uid 444, n_rel 4, n_rec_k 6, n_rel_and_rec_k 3
    uid 471, n_rel 2, n_rec_k 1, n_rel_and_rec_k 1
    uid 434, n_rel 4, n_rec_k 3, n_rel_and_rec_k 3
    uid 170, n_rel 2, n_rec_k 3, n_rel_and_rec_k 2
    uid 383, n_rel 2, n_rec_k 0, n_rel_and_rec_k 0
    uid 175, n_rel 3, n_rec_k 2, n_rel_and_rec_k 2
    uid 594, n_rel 2, n_rec_k 0, n_rel_and_rec_k 0
    uid 206, n_rel 0, n_rec_k 0, n_rel_and_rec_k 0
    uid 556, n_rel 1, n_rec_k 1, n_rel_and_rec_k 1
    uid 91, n_rel 2, n_rec_k 1, n_rel_and_rec_k 1
    uid 512, n_rel 2, n_rec_k 1, n_rel_and_rec_k 1
    uid 10, n_rel 4, n_rec_k 2, n_rel_and_rec_k 2
    uid 495, n_rel 3, n_rec_k 1, n_rel_and_rec_k 1
    uid 440, n_rel 3, n_rec_k 3, n_rel_and_rec_k 3
    uid 284, n_rel 1, n_rec_k 3, n_rel_and_rec_k 1
    uid 362, n_rel 0, n_rec_k 0, n_rel_and_rec_k 0
    uid 405, n_rel 0, n_rec_k 1, n_rel_and_rec_k 0
    uid 530, n_rel 1, n_rec_k 1, n_rel_and_rec_k 1
    uid 349, n_rel 0, n_rec_k 0, n_rel_and_rec_k 0
    uid 565, n_rel 2, n_rec_k 1, n_rel_and_rec_k 1
    uid 470, n_rel 1, n_rec_k 1, n_rel_and_rec_k 1
    uid 71, n_rel 2, n_rec_k 2, n_rel_and_rec_k 2
    uid 217, n_rel 0, n_rec_k 0, n_rel_and_rec_k 0
    uid 2, n_rel 1, n_rec_k 1, n_rel_and_rec_k 1
    uid 453, n_rel 2, n_rec_k 2, n_rel_and_rec_k 0
    uid 430, n_rel 2, n_rec_k 0, n_rel_and_rec_k 0
    uid 212, n_rel 1, n_rec_k 3, n_rel_and_rec_k 1
    uid 575, n_rel 2, n_rec_k 0, n_rel_and_rec_k 0
    uid 580, n_rel 1, n_rec_k 1, n_rel_and_rec_k 1
    uid 7, n_rel 2, n_rec_k 1, n_rel_and_rec_k 1
    uid 179, n_rel 0, n_rec_k 1, n_rel_and_rec_k 0
    uid 462, n_rel 2, n_rec_k 2, n_rel_and_rec_k 1
    uid 184, n_rel 0, n_rec_k 0, n_rel_and_rec_k 0
    uid 277, n_rel 2, n_rec_k 1, n_rel_and_rec_k 1
    uid 137, n_rel 1, n_rec_k 0, n_rel_and_rec_k 0
    uid 133, n_rel 1, n_rec_k 1, n_rel_and_rec_k 1
    uid 86, n_rel 3, n_rec_k 3, n_rel_and_rec_k 3
    uid 543, n_rel 1, n_rec_k 1, n_rel_and_rec_k 0
    uid 597, n_rel 2, n_rec_k 1, n_rel_and_rec_k 0
    uid 157, n_rel 1, n_rec_k 0, n_rel_and_rec_k 0
    uid 323, n_rel 0, n_rec_k 0, n_rel_and_rec_k 0
    uid 546, n_rel 0, n_rec_k 1, n_rel_and_rec_k 0
    uid 518, n_rel 1, n_rec_k 0, n_rel_and_rec_k 0
    uid 449, n_rel 2, n_rec_k 3, n_rel_and_rec_k 2
    uid 314, n_rel 1, n_rec_k 1, n_rel_and_rec_k 1
    uid 517, n_rel 0, n_rec_k 0, n_rel_and_rec_k 0
    uid 292, n_rel 0, n_rec_k 0, n_rel_and_rec_k 0
    uid 340, n_rel 1, n_rec_k 1, n_rel_and_rec_k 1
    uid 268, n_rel 1, n_rec_k 0, n_rel_and_rec_k 0
    uid 359, n_rel 0, n_rec_k 1, n_rel_and_rec_k 0
    uid 348, n_rel 1, n_rec_k 1, n_rel_and_rec_k 1
    uid 374, n_rel 1, n_rec_k 1, n_rel_and_rec_k 1
    


```python
# Precision and recall can then be averaged over all users
# Precision(정밀도): 추천된 아이템 중 사용자가 실제로 관심을 가진 아이템의 비율
# Recall(재현율): 실제로 관련 있는 항목 중에서 모델이 성공적으로 추천한 항목의 비율
print(f'precision @ {k}: {sum(prec for prec in precisions.values()) / len(precisions)}')
print(f'recall @ {k} : {sum(rec for rec in recalls.values()) / len(recalls)}')
```

    precision @ 100: 0.5606722011551397
    recall @ 100 : 0.5217945086846218
    
