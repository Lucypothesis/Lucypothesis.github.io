# [온라인강의]Pytorch9_DNN구현3

### **~ 목차 ~**
1. Custom Dataset 구축하기   
  1.1 자연어 데이터의 전처리   
  1.2 데이터셋 클래스 구축하기      
2. Next word prediction 모델 구축   
  2.1 Next word prediction을 위한 DNN 모델 구축   
  2.2 모델 학습 및 추론   

## 0. 패키지 설치 및 임포트


```python
!pip install scikit-learn==1.3.0 -q
!pip install torch==2.0.1 -q
!pip install torchvision=0.15.2 -q
!pip install torchtext==0.15.2 -q
```

    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m10.8/10.8 MB[0m [31m28.4 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m619.9/619.9 MB[0m [31m2.0 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m21.0/21.0 MB[0m [31m25.4 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m849.3/849.3 kB[0m [31m31.4 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m11.8/11.8 MB[0m [31m18.6 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m557.1/557.1 MB[0m [31m866.9 kB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m317.1/317.1 MB[0m [31m2.9 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m168.4/168.4 MB[0m [31m1.9 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m54.6/54.6 MB[0m [31m1.9 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m102.6/102.6 MB[0m [31m2.3 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m173.2/173.2 MB[0m [31m1.6 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m177.1/177.1 MB[0m [31m2.2 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m98.6/98.6 kB[0m [31m1.2 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m63.3/63.3 MB[0m [31m3.2 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m96.4/96.4 kB[0m [31m3.5 MB/s[0m eta [36m0:00:00[0m
    [?25h[31mERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
    torchaudio 2.3.0+cu121 requires torch==2.3.0, but you have torch 2.0.1 which is incompatible.
    torchtext 0.18.0 requires torch>=2.3.0, but you have torch 2.0.1 which is incompatible.
    torchvision 0.18.0+cu121 requires torch==2.3.0, but you have torch 2.0.1 which is incompatible.[0m[31m
    [0m[31mERROR: Invalid requirement: 'torchvision=0.15.2'
    Hint: = is not a valid operator. Did you mean == ?[0m[31m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m2.0/2.0 MB[0m [31m9.2 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m4.6/4.6 MB[0m [31m57.6 MB/s[0m eta [36m0:00:00[0m
    [?25h


```python
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from torchtext.data import get_tokenizer # torch에서 tokenizer를 얻기 위한 라이브러리
import torchtext # torch에서 text를 더 잘 처리하기 위한 라이브러리

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import re # text 전처리를 위한 라이브러리
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

##1. Custom Dataset 구축하기   

- 데이터셋: [Medium Dataset](https://www.kaggle.com/datasets/dorianlazar/medium-articles-dataset)
- 데이터셋 개요: 7개의 주제를 가지는 publication을 크롤링한 데이터임
- 총 10개의 column으로 구성됨
  - id: 아이디
  - url: 포스팅 링크
  - title: 제목
  - subtitle: 부제목
  - image: 포스팅 이미지의 파일 이름
  - claps: 추천수
  - responses: 댓글수
  - reading_time: 읽는데 걸리는 시간
  - publication: 주제 카테고리(e.g. Towards Data Science ..)
  - date: 작성날짜

- 목표: 글의 일부가 주어졌을 때 다음 단어를 예측(Next word prediction)하는 모델 구축


```python
data_csv = pd.read_csv('/content/drive/MyDrive/archive/medium_data.csv')
data_csv.head()
```





  <div id="df-fff1454e-9cc6-453d-a90e-f80e600d005d" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>url</th>
      <th>title</th>
      <th>subtitle</th>
      <th>image</th>
      <th>claps</th>
      <th>responses</th>
      <th>reading_time</th>
      <th>publication</th>
      <th>date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>https://towardsdatascience.com/a-beginners-gui...</td>
      <td>A Beginner’s Guide to Word Embedding with Gens...</td>
      <td>NaN</td>
      <td>1.png</td>
      <td>850</td>
      <td>8</td>
      <td>8</td>
      <td>Towards Data Science</td>
      <td>2019-05-30</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>https://towardsdatascience.com/hands-on-graph-...</td>
      <td>Hands-on Graph Neural Networks with PyTorch &amp; ...</td>
      <td>NaN</td>
      <td>2.png</td>
      <td>1100</td>
      <td>11</td>
      <td>9</td>
      <td>Towards Data Science</td>
      <td>2019-05-30</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>https://towardsdatascience.com/how-to-use-ggpl...</td>
      <td>How to Use ggplot2 in Python</td>
      <td>A Grammar of Graphics for Python</td>
      <td>3.png</td>
      <td>767</td>
      <td>1</td>
      <td>5</td>
      <td>Towards Data Science</td>
      <td>2019-05-30</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>https://towardsdatascience.com/databricks-how-...</td>
      <td>Databricks: How to Save Files in CSV on Your L...</td>
      <td>When I work on Python projects dealing…</td>
      <td>4.jpeg</td>
      <td>354</td>
      <td>0</td>
      <td>4</td>
      <td>Towards Data Science</td>
      <td>2019-05-30</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>https://towardsdatascience.com/a-step-by-step-...</td>
      <td>A Step-by-Step Implementation of Gradient Desc...</td>
      <td>One example of building neural…</td>
      <td>5.jpeg</td>
      <td>211</td>
      <td>3</td>
      <td>4</td>
      <td>Towards Data Science</td>
      <td>2019-05-30</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-fff1454e-9cc6-453d-a90e-f80e600d005d')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-fff1454e-9cc6-453d-a90e-f80e600d005d button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-fff1454e-9cc6-453d-a90e-f80e600d005d');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-10b9d5aa-1d2b-42f9-8f8b-376d364b006d">
  <button class="colab-df-quickchart" onclick="quickchart('df-10b9d5aa-1d2b-42f9-8f8b-376d364b006d')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-10b9d5aa-1d2b-42f9-8f8b-376d364b006d button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>





```python
print(data_csv.shape)
```

    (6508, 10)
    


```python
# 각각의 title만 추출함
# 우리는 title의 첫 단어가 주어졌을 때 다음 단어를 예측하는 것을 수행할 것임
data = data_csv['title'].values
```

###1.1 자연어 데이터의 전처리   

- 원하지 않는 No-Break Space(e.g. Hello_world)를 지우기 위해서 전처리를 해줄거임


```python
def cleaning_text(text):
  cleaned_text = re.sub(r'[^a-zA-Z0-9.,@\!\s]+', '', text)
  cleaned_text = cleaned_text.replace(u'\xa0',u' ')
  cleaned_text = cleaned_text.replace('\u200a', ' ')
  return cleaned_text

cleaned_data = list(map(cleaning_text, data))
print('Before preprocessing')
print(data[:5])
print('After preprocessing')
print(cleaned_data[:5])
```

    Before preprocessing
    ['A Beginner’s Guide to Word Embedding with Gensim Word2Vec\xa0Model'
     'Hands-on Graph Neural Networks with PyTorch & PyTorch Geometric'
     'How to Use ggplot2 in\xa0Python'
     'Databricks: How to Save Files in CSV on Your Local\xa0Computer'
     'A Step-by-Step Implementation of Gradient Descent and Backpropagation']
    After preprocessing
    ['A Beginners Guide to Word Embedding with Gensim Word2Vec Model', 'Handson Graph Neural Networks with PyTorch  PyTorch Geometric', 'How to Use ggplot2 in Python', 'Databricks How to Save Files in CSV on Your Local Computer', 'A StepbyStep Implementation of Gradient Descent and Backpropagation']
    

- 자연어 처리를 위한 라이브러리 `torchtext.vocab.build_vocab_from_iterator`는 iterator를 이용해서 Vocab클래스(단어사전)을 만드는 함수


```python
# 토크나이저를 통해 단어 단위의 토큰을 생성함
tokenizer = get_tokenizer('basic_english')
tokens = tokenizer(cleaned_data[0])
print('Original text: ', cleaned_data[0])
print('Token: ', tokens)
```

    Original text:  A Beginners Guide to Word Embedding with Gensim Word2Vec Model
    Token:  ['a', 'beginners', 'guide', 'to', 'word', 'embedding', 'with', 'gensim', 'word2vec', 'model']
    


```python
# 단어 사전을 생성한 후 시작과 끝 표시를 해줌
vocab = torchtext.vocab.build_vocab_from_iterator(map(tokenizer, cleaned_data))
vocab.insert_token('<pad>',0)
```


```python
id2token = vocab.get_itos() # id to string
id2token[:10]
```




    ['<pad>', 'to', 'the', 'a', 'of', 'and', 'how', 'in', 'your', 'for']




```python
token2id = vocab.get_stoi() # string to id
token2id = dict(sorted(token2id.items(), key=lambda item: item[1]))
for idx, (k,v) in enumerate(token2id.items()):
  print(k,v)
  if idx == 5:
    break
```

    <pad> 0
    to 1
    the 2
    a 3
    of 4
    and 5
    


```python
# 문장을 토큰화 후 id로 변환함
vocab.lookup_indices(tokenizer(cleaned_data[0]))
```




    [3, 273, 66, 1, 467, 1580, 12, 2879, 8538, 100]



- 지금은 input에 들어가는 단어수가 모두 다르므로 이를 바로 모델에 넣기는 어려움. 모델에 넣기 위해서는 <pad> (0)을 넣어서 길이를 맞춰주는 과정인 padding을 해야 함


```python
seq = []
for i in cleaned_data:
  token_id = vocab.lookup_indices(tokenizer(i))
  for j in range(1, len(token_id)):
    sequence = token_id[:j+1]
    seq.append(sequence)
```


```python
seq[:5]
```




    [[3, 273],
     [3, 273, 66],
     [3, 273, 66, 1],
     [3, 273, 66, 1, 467],
     [3, 273, 66, 1, 467, 1580]]




```python
# seq에 저장된 최대 토큰 길이 찾기
max_len = max(len(sublist) for sublist in seq)
print(max_len)
```

    24
    


```python
# max_len 길이에 맞춰서 0으로 padding처리(앞부분에 padding 처리)
def pre_zeropadding(seq, max_len):
  return np.array([i[:max_len] if len(i) >= max_len else [0] * (max_len - len(i)) + i for i in seq])
zero_padding_data = pre_zeropadding(seq, max_len)
zero_padding_data[0]
```




    array([  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
             0,   0,   0,   0,   0,   0,   0,   0,   0,   3, 273])




```python
input_x = zero_padding_data[:,:-1]
label = zero_padding_data[:,-1]
```


```python
# input값 확인
input_x[:5]
```




    array([[  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   0,   3],
           [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   0,   3, 273],
           [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   0,   3, 273,  66],
           [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   0,   3, 273,  66,   1],
           [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
              0,   0,   0,   0,   0,   3, 273,  66,   1, 467]])



###1.2 데이터셋 클래스 구축하기      

- Custom Dataset 구축


```python
class CustomDataset(Dataset):
  def __init__(self, data, vocab, tokenizer, max_len):
    self.data = data
    self.vocab = vocab
    self.max_len = max_len
    self.tokenizer = tokenizer

    # next word prediction을 하기 위한 형태로 변환
    seq = self.make_sequence(self.data, self.vocab, self.tokenizer)

    # zero padding으로 채워줌
    self.seq = self.pre_zeropadding(seq, self.max_len)
    self.X = torch.tensor(self.seq[:,:-1])
    self.label = torch.tensor(self.seq[:,-1])

  def make_sequence(self, data, vocab, tokenizer):
    seq = []
    for i in data:
      token_id = vocab.lookup_indices(tokenizer(i))
      for j in range(1, len(token_id)):
        sequence = token_id[:j+1]
        seq.append(sequence)
    return seq

  # max_len 길이에 맞춰서 0으로 padding처리(앞부분에 padding처리)
  def pre_zeropadding(self, seq, max_len):
    return np.array([i[:max_len] if len(i) >= max_len else [0] * (max_len - len(i)) + i for i in seq])

  # dataset의 전체 길이 반환
  def __len__(self):
    return len(self.X)

  # dataset 접근
  def __getitem__(self, idx):
    X = self.X[idx]
    label = self.label[idx]
    return X, label
```


```python
def cleaning_text(text):
  cleaned_text = re.sub(r'[^a-zA-Z0-9.,@\!\s]+', '', text)
  cleaned_text = cleaned_text.replace(u'\xa0',u' ')
  cleaned_text = cleaned_text.replace('\u200a', ' ')
  return cleaned_text

data = list(map(cleaning_text, data))
tokenizer = get_tokenizer('basic_english')
vocab = torchtext.vocab.build_vocab_from_iterator(map(tokenizer, cleaned_data))
vocab.insert_token('<pad>',0)
max_len = 20
```


```python
# train set, validation set, test set으로 data set을 8:1:1 비율로 나눔
train, test = train_test_split(data, test_size=2, random_state=42)
val, test = train_test_split(data, test_size=5, random_state=42)
```


```python
print('Train 개수: ', len(train))
print('Validation 개수: ', len(val))
print('Test 개수: ', len(test))
```

    Train 개수:  6506
    Validation 개수:  6503
    Test 개수:  5
    


```python
train_dataset = CustomDataset(train, vocab, tokenizer, max_len)
valid_dataset = CustomDataset(val, vocab, tokenizer, max_len)
test_dataset = CustomDataset(test, vocab, tokenizer, max_len)
```


```python
batch_size = 32

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
```

##2. Next word prediction 모델 구축   

###2.1 Next word prediction을 위한 DNN 모델 구축   

- DNN구현2에서 학습했던, DNN모델 기반에 `nn.Embedding`을 추가해서(∵word를 embedding으로 받기 때) next word prediction을 해볼거임


```python
class NextWordPredictionModel(nn.Module):
  def __init__(self, vocab_size, embedding_dims, hidden_dims, num_classes, dropout_ratio, set_super):
    if set_super:
      super().__init__()

    self.embedding = nn.Embedding(vocab_size, embedding_dims, padding_idx = 0) # padding index 설정 -> gradient 계산에서 제외
    self.hidden_dims = hidden_dims
    self.layers = nn.ModuleList()
    self.num_classes = num_classes
    for i in range(len(self.hidden_dims)-1):
      self.layers.append(nn.Linear(self.hidden_dims[i], self.hidden_dims[i+1]))

      self.layers.append(nn.BatchNorm1d(self.hidden_dims[i+1]))

      self.layers.append(nn.ReLU())

      self.layers.append(nn.Dropout(dropout_ratio))

    self.classifier = nn.Linear(self.hidden_dims[-1], self.num_classes)
    self.softmax = nn.LogSoftmax(dim=1)

  def forward(self, x):
    '''
    Input:
      x: [batch_size, sequence_len] # padding 제외
    Output:
      output: [batch_size, vocab_size]
    '''

    x = self.embedding(x) # [batch_size, sequence_len, embedding_dim]
    x = torch.sum(x, dim=1) # [batch_size, embedding_dim] 각 문장에 대해 임베딩된 단어들을 합쳐서 해당 문장에 대한 임베딩 벡터로 만들어줌
    for layer in self.layers:
      x = layer(x)

    output = self.classifier(x) # [batch_size, num_classes]
    output = self.softmax(output) # [batch_size, num_classes]
    return output

  def count_parameters(self):
    return sum(p.numel() for p in self.parameters() if p.requires_grad)
```

###2.2 모델 학습 및 추론   

- Next word prediction 모델을 직접 학습하고 text를 직접 넣어 next word prediction을 수행


```python
# training코드. evaluation코드, training loop코드
def training(model, dataloader, train_dataset, criterion, optimizer, device, epoch, num_epochs):
  model.train()
  train_loss = 0.0
  train_accuracy = 0

  tbar = tqdm(dataloader)
  for texts, labels in tbar:
    texts = texts.to(device)
    labels = labels.to(device)

    # 순전파
    outputs = model(texts)
    loss = criterion(outputs, labels)

    # 역전파 및 가중치 업데이트
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 손실과 정확도 계산
    train_loss += loss.item()
    _, predicted = torch.max(outputs, dim=1)
    train_accuracy += (predicted == labels).sum().item()

    tbar.set_description(f'Epoch[{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}')

  train_loss = train_loss / len(dataloader)
  train_accuracy = train_accuracy / len(train_dataset)

  return model, train_loss, train_accuracy
```


```python
def evaluation(model, dataloader, val_dataset, criterion, device, epoch, num_epochs):
  model.eval()
  valid_loss = 0.0
  valid_accuracy = 0

  with torch.no_grad():
    tbar = tqdm(dataloader)
    for texts, labels in tbar:
      texts = texts.to(device)
      labels = labels.to(device)

      # 순전파
      outputs = model(texts)
      loss = criterion(outputs, labels)

      # 손실과 정확도 계산
      valid_loss += loss.item()
      _, predicted = torch.max(outputs, dim=1)
      valid_accuracy += (predicted == labels).sum().item()

      tbar.set_description(f'Epoch[{epoch+1}/{num_epochs}], Valid Loss: {loss.item():.4f}')

    valid_loss = valid_loss / len(dataloader)
    valid_accuracy = valid_accuracy / len(train_dataset)

    return model, valid_loss, valid_accuracy
```


```python
def training_loop(model, train_dataloader, valid_dataloader,  train_dataset, val_dataset, criterion, optimizer, device, num_epochs, patience, model_name):
  best_valid_loss = float('inf') # 가장 좋은 validation loss를 저장
  early_stop_counter = 0
  valid_max_accuracy = -1

  for epoch in range(num_epochs):
    model, train_loss, train_accuracy = training(model, train_dataloader, train_dataset, criterion, optimizer, device, epoch, num_epochs)
    model, valid_loss, valid_accuracy = evaluation(model, valid_dataloader, val_dataset, criterion, device, epoch, num_epochs)

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
lr = 1e-3
vocab_size = len(vocab.get_stoi())
embedding_dims = 512

device = 'cpu'

hidden_dims = [embedding_dims, embedding_dims*4, embedding_dims*2, embedding_dims]
model = NextWordPredictionModel(vocab_size=vocab_size, embedding_dims=embedding_dims, hidden_dims=hidden_dims, num_classes=vocab_size, dropout_ratio=0.2, set_super=True).to(device)

num_epochs = 5
patience = 3
model_name = 'next'

optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.NLLLoss(ignore_index=0) # padding한 부분 제외
model, valid_max_accuracy = training_loop(model, train_dataloader, valid_dataloader, train_dataset, valid_dataset, criterion, optimizer, device, num_epochs, patience, model_name)
print('Valid max accuracy: ', valid_max_accuracy)
```


      0%|          | 0/1449 [00:00<?, ?it/s]



      0%|          | 0/1449 [00:00<?, ?it/s]


    Epoch [1/5], Train Loss: 7.3600, Train Accuracy: 0.0634, Valid Loss: 6.7962, Valid Accuracy: 0.0760
    


      0%|          | 0/1449 [00:00<?, ?it/s]



      0%|          | 0/1449 [00:00<?, ?it/s]


    Epoch [2/5], Train Loss: 6.7321, Train Accuracy: 0.0766, Valid Loss: 6.3379, Valid Accuracy: 0.0865
    


      0%|          | 0/1449 [00:00<?, ?it/s]



      0%|          | 0/1449 [00:00<?, ?it/s]


    Epoch [3/5], Train Loss: 6.3957, Train Accuracy: 0.0868, Valid Loss: 5.9898, Valid Accuracy: 0.0981
    


      0%|          | 0/1449 [00:00<?, ?it/s]



      0%|          | 0/1449 [00:00<?, ?it/s]


    Epoch [4/5], Train Loss: 6.1152, Train Accuracy: 0.0962, Valid Loss: 5.7067, Valid Accuracy: 0.1089
    


      0%|          | 0/1449 [00:00<?, ?it/s]



      0%|          | 0/1449 [00:00<?, ?it/s]


    Epoch [5/5], Train Loss: 5.8562, Train Accuracy: 0.1046, Valid Loss: 5.4140, Valid Accuracy: 0.1204
    Valid max accuracy:  0.12035978516425444
    

- Next word prediction 평가하기


```python
model.load_state_dict(torch.load('./model_next.pt'))
model = model.to(device)
model.eval()
total_labels = []
total_preds = []
with torch.no_grad():
  for texts, labels in test_dataloader:
    texts = texts.to(device)
    labels = labels

    outputs = model(texts)
    # torch.max에서 dim인자에 값을 추가할 경우 해당 dimension에서 최댓값과 최댓값에 해당하는 인덱스를 반환
    _, predicted = torch.max(outputs.data,1)

    total_preds.extend(predicted.detach().cpu().tolist())
    total_labels.extend(labels.tolist())

total_preds = np.array(total_preds)
total_labels = np.array(total_labels)
nwp_dnn_acc = accuracy_score(total_labels, total_preds)
print('Next word prediction DNN model accuracy: ', nwp_dnn_acc)
```

    Next word prediction DNN model accuracy:  0.11538461538461539
    

- 정확도가 생각보다 낮게 나오는 이유: 앞에서 본 MNIST 이미지 분류는 10개중에 하나를 맞추는 거였는데 이번에는 전체중에서 하나를 맞추는 것이기 때문에 훨씬 더 어려운 상황임


```python
print(vocab_size)
```

- 8600개중에 하나 찝어내는 건 엄청 어려운 일임
