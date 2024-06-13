---
layout: post
title: "240603(월) H&M EDA FIRST LOOK 정리"
subtitle: "[Tips]"
date: 2024-06-03 19:22
background: 
tag: [Tips, Github io, Notion]
---

링크:
https://www.kaggle.com/code/vanguarde/h-m-eda-first-look

# 1. 라이브러리 임포트 & csv 파일 열기
 1) tqdm: 진행 바(progress bar)를 통해 얼마나 진행되고 있는지 확인할 수 있음


```python
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm
```

2) feature 설명
- article_id: 각 제품의 고유 식별자.
- product_code, prod_name: 각 제품의 고유 식별자와 그 이름 (이 둘은 동일하지 않음).
- product_type_no, product_type_name, product_group_name: product_code의 그룹과 그 이름.
- graphical_appearance_no, graphical_appearance_name: 그래픽의 그룹과 그 이름.
- colour_group_code, colour_group_name: 색상의 그룹과 그 이름.
- perceived_colour_value_id, perceived_colour_value_name, perceived_colour_master_id, perceived_colour_master_name: 추가 색상 정보.
- perceived_colour_value_id: 인지된 색상 값의 ID.
- perceived_colour_value_name: 인지된 색상 값의 이름.
- perceived_colour_master_id: 인지된 색상 마스터의 ID.
- perceived_colour_master_name: 인지된 색상 마스터의 이름.
- department_no, department_name: 각 부서의 고유 식별자와 그 이름.
- index_code, index_name: 각 인덱스의 고유 식별자와 그 이름.
- index_group_no, index_group_name: 인덱스의 그룹과 그 이름.
- section_no, section_name: 각 섹션의 고유 식별자와 그 이름.
- garment_group_no, garment_group_name: 각 의류 그룹의 고유 식별자와 그 이름.
- detail_desc: 제품에 대한 세부 설명.


```python
articles = pd.read_csv("/content/drive/MyDrive/csv_datas/articles.csv")
customers = pd.read_csv("/content/drive/MyDrive/csv_datas/customers.csv")
transactions = pd.read_csv("/content/drive/MyDrive/csv_datas/transactions_train.csv")
```

# 2. 제품(Articles)


```python
articles.head(3)
```





  <div id="df-9f074716-a4b5-485d-aa41-636b1adffebd" class="colab-df-container">
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
      <th>article_id</th>
      <th>product_code</th>
      <th>prod_name</th>
      <th>product_type_no</th>
      <th>product_type_name</th>
      <th>product_group_name</th>
      <th>graphical_appearance_no</th>
      <th>graphical_appearance_name</th>
      <th>colour_group_code</th>
      <th>colour_group_name</th>
      <th>...</th>
      <th>department_name</th>
      <th>index_code</th>
      <th>index_name</th>
      <th>index_group_no</th>
      <th>index_group_name</th>
      <th>section_no</th>
      <th>section_name</th>
      <th>garment_group_no</th>
      <th>garment_group_name</th>
      <th>detail_desc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>108775015</td>
      <td>108775</td>
      <td>Strap top</td>
      <td>253</td>
      <td>Vest top</td>
      <td>Garment Upper body</td>
      <td>1010016</td>
      <td>Solid</td>
      <td>9</td>
      <td>Black</td>
      <td>...</td>
      <td>Jersey Basic</td>
      <td>A</td>
      <td>Ladieswear</td>
      <td>1</td>
      <td>Ladieswear</td>
      <td>16</td>
      <td>Womens Everyday Basics</td>
      <td>1002</td>
      <td>Jersey Basic</td>
      <td>Jersey top with narrow shoulder straps.</td>
    </tr>
    <tr>
      <th>1</th>
      <td>108775044</td>
      <td>108775</td>
      <td>Strap top</td>
      <td>253</td>
      <td>Vest top</td>
      <td>Garment Upper body</td>
      <td>1010016</td>
      <td>Solid</td>
      <td>10</td>
      <td>White</td>
      <td>...</td>
      <td>Jersey Basic</td>
      <td>A</td>
      <td>Ladieswear</td>
      <td>1</td>
      <td>Ladieswear</td>
      <td>16</td>
      <td>Womens Everyday Basics</td>
      <td>1002</td>
      <td>Jersey Basic</td>
      <td>Jersey top with narrow shoulder straps.</td>
    </tr>
    <tr>
      <th>2</th>
      <td>108775051</td>
      <td>108775</td>
      <td>Strap top (1)</td>
      <td>253</td>
      <td>Vest top</td>
      <td>Garment Upper body</td>
      <td>1010017</td>
      <td>Stripe</td>
      <td>11</td>
      <td>Off White</td>
      <td>...</td>
      <td>Jersey Basic</td>
      <td>A</td>
      <td>Ladieswear</td>
      <td>1</td>
      <td>Ladieswear</td>
      <td>16</td>
      <td>Womens Everyday Basics</td>
      <td>1002</td>
      <td>Jersey Basic</td>
      <td>Jersey top with narrow shoulder straps.</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 25 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-9f074716-a4b5-485d-aa41-636b1adffebd')"
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
        document.querySelector('#df-9f074716-a4b5-485d-aa41-636b1adffebd button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-9f074716-a4b5-485d-aa41-636b1adffebd');
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


<div id="df-4e23b380-124a-4182-b92c-614d802aac47">
  <button class="colab-df-quickchart" onclick="quickchart('df-4e23b380-124a-4182-b92c-614d802aac47')"
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
        document.querySelector('#df-4e23b380-124a-4182-b92c-614d802aac47 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>




Ladieswear가 대부분을 차지하고 Sport가 젤 적게 차지함을 알 수 있음


```python
f, ax = plt.subplots(figsize=(15,7))
ax = sns.histplot(data=articles, y='index_name', color='orange')
ax.set_xlabel('count by index name')
ax.set_ylabel('index_name')
plt.show()
```


    
![png](240603_HM_EDA_FIRST_LOOK_files/240603_HM_EDA_FIRST_LOOK_8_0.png)
    


- hue = 'index_group_name'  
  : 각 garment_group_name 내에서 index_group_name 값에 따라 다른 색상으로 히스토그램 막대를 구분함
- multiple = 'stack'   
  : 히스토그램 막대를 쌓아올리는(stacked) 형태로 그림


```python
f, ax = plt.subplots(figsize=(15,7))
ax = sns.histplot(data = articles, y = 'garment_group_name', color = 'orange', hue = 'index_group_name', multiple = 'stack')
ax.set_xlabel('count by garment group')
ax.set_ylabel('garment group')
plt.show()
```


    
![png](240603_HM_EDA_FIRST_LOOK_files/240603_HM_EDA_FIRST_LOOK_10_0.png)
    


- index_name: 각 인덱스의 이름
- index_group_name: 인덱스 그룹의 이름   
 => Children/Baby하고 Ladieswear는 하위 그룹이 있음을 알 수 있음


```python
articles.groupby(['index_group_name', 'index_name']).count()['article_id']
```




    index_group_name  index_name                    
    Baby/Children     Baby Sizes 50-98                   8875
                      Children Accessories, Swimwear     4615
                      Children Sizes 134-170             9214
                      Children Sizes 92-140             12007
    Divided           Divided                           15149
    Ladieswear        Ladies Accessories                 6961
                      Ladieswear                        26001
                      Lingeries/Tights                   6775
    Menswear          Menswear                          12553
    Sport             Sport                              3392
    Name: article_id, dtype: int64



- product_group_name: product의 그룹 이름
- product_type_name: product의 종류 이름
=> Accessories 종류가 엄청 다양함. 가장 많은 종류는 가방, 귀걸이 모자
=> 그러나 Garment Lower body에 있는 Trousers 항목이 가장 많음(11169개)


```python
pd.options.display.max_rows = None
articles.groupby(['product_group_name', 'product_type_name']).count()['article_id']
```




    product_group_name     product_type_name       
    Accessories            Accessories set                 7
                           Alice band                      6
                           Baby Bib                        3
                           Bag                          1280
                           Beanie                         56
                           Belt                          458
                           Bracelet                      180
                           Braces                          3
                           Bucket hat                      7
                           Cap                            13
                           Cap/peaked                    573
                           Dog Wear                       20
                           Earring                      1159
                           Earrings                       11
                           Eyeglasses                      2
                           Felt hat                       10
                           Giftbox                        15
                           Gloves                        367
                           Hair clip                     244
                           Hair string                   238
                           Hair ties                      24
                           Hair/alice band               854
                           Hairband                        2
                           Hat/beanie                   1349
                           Hat/brim                      396
                           Headband                        1
                           Necklace                      581
                           Other accessories            1034
                           Ring                          240
                           Scarf                        1013
                           Soft Toys                      46
                           Straw hat                       6
                           Sunglasses                    621
                           Tie                           141
                           Umbrella                       26
                           Wallet                         77
                           Watch                          73
                           Waterbottle                    22
    Bags                   Backpack                        6
                           Bumbag                          1
                           Cross-body bag                  5
                           Shoulder bag                    2
                           Tote bag                        2
                           Weekend/Gym bag                 9
    Cosmetic               Chem. cosmetics                 3
                           Fine cosmetics                 46
    Fun                    Toy                             2
    Furniture              Side table                     13
    Garment Full body      Costumes                       90
                           Dress                       10362
                           Dungarees                     309
                           Garment Set                  1320
                           Jumpsuit/Playsuit            1147
                           Outdoor overall                64
    Garment Lower body     Leggings/Tights              1878
                           Outdoor trousers              130
                           Shorts                       3939
                           Skirt                        2696
                           Trousers                    11169
    Garment Upper body     Blazer                       1110
                           Blouse                       3979
                           Bodysuit                      913
                           Cardigan                     1550
                           Coat                          460
                           Hoodie                       2356
                           Jacket                       3940
                           Outdoor Waistcoat             154
                           Polo shirt                    449
                           Shirt                        3405
                           Sweater                      9302
                           T-shirt                      7904
                           Tailored Waistcoat             73
                           Top                          4155
                           Vest top                     2991
    Garment and Shoe care  Clothing mist                   1
                           Sewing kit                      1
                           Stain remover spray             2
                           Washing bag                     1
                           Wood balls                      1
                           Zipper head                     3
    Interior textile       Blanket                         1
                           Cushion                         1
                           Towel                           1
    Items                  Dog wear                        7
                           Keychain                        1
                           Mobile case                     4
                           Umbrella                        3
                           Wireless earphone case          2
    Nightwear              Night gown                    171
                           Pyjama bottom                 220
                           Pyjama jumpsuit/playsuit      388
                           Pyjama set                   1120
    Shoes                  Ballerinas                    372
                           Bootie                         31
                           Boots                        1028
                           Flat shoe                     165
                           Flat shoes                     10
                           Flip flop                     125
                           Heeled sandals                202
                           Heels                          22
                           Moccasins                       4
                           Other shoe                    395
                           Pre-walkers                     1
                           Pumps                         188
                           Sandals                       757
                           Slippers                      249
                           Sneakers                     1621
                           Wedge                         113
    Socks & Tights         Leg warmers                     7
                           Socks                        1889
                           Underwear Tights              546
    Stationery             Marker pen                      5
    Swimwear               Bikini top                    850
                           Sarong                         66
                           Swimsuit                      662
                           Swimwear bottom              1307
                           Swimwear set                  192
                           Swimwear top                   50
    Underwear              Bra                          2212
                           Bra extender                    1
                           Kids Underwear top             96
                           Long John                      30
                           Nipple covers                  19
                           Robe                          136
                           Underdress                     20
                           Underwear body                174
                           Underwear bottom             2748
                           Underwear corset                7
                           Underwear set                  47
    Underwear/nightwear    Sleep Bag                       6
                           Sleeping sack                  48
    Unknown                Unknown                       121
    Name: article_id, dtype: int64




```python
for col in articles.columns:
  if not 'no' in col and not 'code' in col and not 'id' in col:
    un_n = articles[col].nunique() # nunique(): 해당 열의 고유값의 개수를 계산함
    print(f'n of unique {col}: {un_n}')
```

    n of unique prod_name: 45875
    n of unique product_type_name: 131
    n of unique product_group_name: 19
    n of unique graphical_appearance_name: 30
    n of unique colour_group_name: 50
    n of unique perceived_colour_value_name: 8
    n of unique perceived_colour_master_name: 20
    n of unique department_name: 250
    n of unique index_name: 10
    n of unique index_group_name: 5
    n of unique section_name: 56
    n of unique garment_group_name: 21
    n of unique detail_desc: 43404
    

# 3. 고객(Customers)

1) feature 설명
- customer_id: 각 고객의 고유 식별자
- FN: 1 또는 누락됨
- Active: 1 또는 누락됨
- club_member_status: 클럽 내 상태
- fashion_news_frequency: H&M이 고객에게 소식을 얼마나 자주 보낼 수 있는지
- age: 현재 나이
- postal_code: 고객의 우편번호


```python
pd.options.display.max_rows = 50 # pd.options: Pandas 출력옵션 /  display.max_rows: 한번에 표시되는 최대 row 수를 50개로 설정
customers.head()
```





  <div id="df-4d29700e-023e-4082-9c22-d3cb5f3f40c7" class="colab-df-container">
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
      <th>customer_id</th>
      <th>FN</th>
      <th>Active</th>
      <th>club_member_status</th>
      <th>fashion_news_frequency</th>
      <th>age</th>
      <th>postal_code</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>00000dbacae5abe5e23885899a1fa44253a17956c6d1c3...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>ACTIVE</td>
      <td>NONE</td>
      <td>49.0</td>
      <td>52043ee2162cf5aa7ee79974281641c6f11a68d276429a...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0000423b00ade91418cceaf3b26c6af3dd342b51fd051e...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>ACTIVE</td>
      <td>NONE</td>
      <td>25.0</td>
      <td>2973abc54daa8a5f8ccfe9362140c63247c5eee03f1d93...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>000058a12d5b43e67d225668fa1f8d618c13dc232df0ca...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>ACTIVE</td>
      <td>NONE</td>
      <td>24.0</td>
      <td>64f17e6a330a85798e4998f62d0930d14db8db1c054af6...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>00005ca1c9ed5f5146b52ac8639a40ca9d57aeff4d1bd2...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>ACTIVE</td>
      <td>NONE</td>
      <td>54.0</td>
      <td>5d36574f52495e81f019b680c843c443bd343d5ca5b1c2...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>00006413d8573cd20ed7128e53b7b13819fe5cfc2d801f...</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>ACTIVE</td>
      <td>Regularly</td>
      <td>52.0</td>
      <td>25fa5ddee9aac01b35208d01736e57942317d756b32ddd...</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-4d29700e-023e-4082-9c22-d3cb5f3f40c7')"
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
        document.querySelector('#df-4d29700e-023e-4082-9c22-d3cb5f3f40c7 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-4d29700e-023e-4082-9c22-d3cb5f3f40c7');
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


<div id="df-5223bcbc-acf9-494c-9387-08dfb7de5f31">
  <button class="colab-df-quickchart" onclick="quickchart('df-5223bcbc-acf9-494c-9387-08dfb7de5f31')"
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
        document.querySelector('#df-5223bcbc-acf9-494c-9387-08dfb7de5f31 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>




customers에는 중복된 값이 없음을 알 수 있음


```python
print(customers.shape[0])
print(customers['customer_id'].nunique())
```

    1371980
    1371980
    

우편번호 값 하나가 비정상적으로 큼(120303)   
가능성1. 주소가 누락된 값(nan)으로 인코딩됨   
가능성2. 대형 물류 센터나 픽업 장소일 수 있음


```python
data_postal = customers.groupby('postal_code', as_index = False).count().sort_values('customer_id', ascending = False)
data_postal.head()
```





  <div id="df-38bde9ad-d1fd-47a2-b344-70a847688822" class="colab-df-container">
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
      <th>postal_code</th>
      <th>customer_id</th>
      <th>FN</th>
      <th>Active</th>
      <th>club_member_status</th>
      <th>fashion_news_frequency</th>
      <th>age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>61034</th>
      <td>2c29ae653a9282cce4151bd87643c907644e09541abc28...</td>
      <td>120303</td>
      <td>42874</td>
      <td>39886</td>
      <td>118281</td>
      <td>114377</td>
      <td>118002</td>
    </tr>
    <tr>
      <th>281937</th>
      <td>cc4ed85e30f4977dae47662ddc468cd2eec11472de6fac...</td>
      <td>261</td>
      <td>109</td>
      <td>104</td>
      <td>261</td>
      <td>261</td>
      <td>260</td>
    </tr>
    <tr>
      <th>156090</th>
      <td>714976379549eb90aae4a71bca6c7402cc646ae7c40f6c...</td>
      <td>159</td>
      <td>90</td>
      <td>88</td>
      <td>159</td>
      <td>159</td>
      <td>158</td>
    </tr>
    <tr>
      <th>171208</th>
      <td>7c1fa3b0ec1d37ce2c3f34f63bd792f3b4494f324b6be5...</td>
      <td>157</td>
      <td>55</td>
      <td>54</td>
      <td>157</td>
      <td>156</td>
      <td>156</td>
    </tr>
    <tr>
      <th>126228</th>
      <td>5b7eb31eabebd3277de632b82267286d847fd5d44287ee...</td>
      <td>156</td>
      <td>42</td>
      <td>41</td>
      <td>156</td>
      <td>156</td>
      <td>155</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-38bde9ad-d1fd-47a2-b344-70a847688822')"
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
        document.querySelector('#df-38bde9ad-d1fd-47a2-b344-70a847688822 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-38bde9ad-d1fd-47a2-b344-70a847688822');
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


<div id="df-8a8baf13-b29f-4694-b5ee-52abd25d3813">
  <button class="colab-df-quickchart" onclick="quickchart('df-8a8baf13-b29f-4694-b5ee-52abd25d3813')"
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
        document.querySelector('#df-8a8baf13-b29f-4694-b5ee-52abd25d3813 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>




customer_id와 마찬가지로 age하고 club_member_status도 다 다름   
## ❓Q: 우편번호가 같은데 이렇게 많은 사람들이 있을 수 있나?


```python
customers[customers['postal_code'] == '2c29ae653a9282cce4151bd87643c907644e09541abc28ae87dea0d1f6603b1c']
```





  <div id="df-115522ad-8b1c-43ea-adbd-76e3a4a7779c" class="colab-df-container">
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
      <th>customer_id</th>
      <th>FN</th>
      <th>Active</th>
      <th>club_member_status</th>
      <th>fashion_news_frequency</th>
      <th>age</th>
      <th>postal_code</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>000064249685c11552da43ef22a5030f35a147f723d5b0...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>None</td>
      <td>NaN</td>
      <td>2c29ae653a9282cce4151bd87643c907644e09541abc28...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>00007e8d4e54114b5b2a9b51586325a8d0fa74ea23ef77...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>ACTIVE</td>
      <td>None</td>
      <td>20.0000</td>
      <td>2c29ae653a9282cce4151bd87643c907644e09541abc28...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>00008469a21b50b3d147c97135e25b4201a8c58997f787...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>ACTIVE</td>
      <td>None</td>
      <td>20.0000</td>
      <td>2c29ae653a9282cce4151bd87643c907644e09541abc28...</td>
    </tr>
    <tr>
      <th>11</th>
      <td>000097d91384a0c14893c09ed047a963c4fc6a5c021044...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>ACTIVE</td>
      <td>None</td>
      <td>31.0000</td>
      <td>2c29ae653a9282cce4151bd87643c907644e09541abc28...</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0000ae1bbb25e04bdc7e35f718e852adfb3fbb72ef38b3...</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>ACTIVE</td>
      <td>Regularly</td>
      <td>29.0000</td>
      <td>2c29ae653a9282cce4151bd87643c907644e09541abc28...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1371911</th>
      <td>fffc36edc9658a27fc810a5a752b26a917c1a0a6b4261a...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>ACTIVE</td>
      <td>None</td>
      <td>29.0000</td>
      <td>2c29ae653a9282cce4151bd87643c907644e09541abc28...</td>
    </tr>
    <tr>
      <th>1371914</th>
      <td>fffc582abe0ed452f145f2e86649d7d745a2b3a16942dd...</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>ACTIVE</td>
      <td>Regularly</td>
      <td>46.0000</td>
      <td>2c29ae653a9282cce4151bd87643c907644e09541abc28...</td>
    </tr>
    <tr>
      <th>1371917</th>
      <td>fffc8a9f1545e08fdf09f7dc3bd0736e1bdccceecca685...</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>ACTIVE</td>
      <td>Regularly</td>
      <td>48.0000</td>
      <td>2c29ae653a9282cce4151bd87643c907644e09541abc28...</td>
    </tr>
    <tr>
      <th>1371943</th>
      <td>fffdf87d73dfab4ff56920a14c89e6cc929fda2a0619c9...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>ACTIVE</td>
      <td>None</td>
      <td>23.0000</td>
      <td>2c29ae653a9282cce4151bd87643c907644e09541abc28...</td>
    </tr>
    <tr>
      <th>1371974</th>
      <td>ffffaff3905b803d1c7e153a1378a5151e1f34f236ba54...</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>ACTIVE</td>
      <td>Regularly</td>
      <td>21.0000</td>
      <td>2c29ae653a9282cce4151bd87643c907644e09541abc28...</td>
    </tr>
  </tbody>
</table>
<p>120303 rows × 7 columns</p>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-115522ad-8b1c-43ea-adbd-76e3a4a7779c')"
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
        document.querySelector('#df-115522ad-8b1c-43ea-adbd-76e3a4a7779c button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-115522ad-8b1c-43ea-adbd-76e3a4a7779c');
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


<div id="df-682240de-a96a-41f2-9094-789322772442">
  <button class="colab-df-quickchart" onclick="quickchart('df-682240de-a96a-41f2-9094-789322772442')"
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
        document.querySelector('#df-682240de-a96a-41f2-9094-789322772442 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>




21~23살이 가장 많은 걸 알 수 있음


```python
sns.set_style('darkgrid') # darkgrid: 어두운 그리드 배경
f, ax = plt.subplots(figsize = (10,5))
ax = sns.histplot(data = customers, x = 'age', bins = 50, color = 'orange') # bins=50: 데이터 범위를 50개로 나눠서 히스토그램 그린다는 뜻
ax.set_xlabel('Distribution of the customers age')
plt.show()
```


    
![png](240603_HM_EDA_FIRST_LOOK_files/240603_HM_EDA_FIRST_LOOK_26_0.png)
    


거의 모든 멤버의 club_member_status가 active이고 몇몇은 pre-create, 탈퇴고객(left club)은 소수임


```python
sns.set_style('darkgrid')
f, ax = plt.subplots(figsize = (10,5))
ax = sns.histplot(data = customers, x = 'club_member_status', color = 'orange')
ax.set_xlabel('Distribution of club member status')
plt.show()
```


    
![png](240603_HM_EDA_FIRST_LOOK_files/240603_HM_EDA_FIRST_LOOK_28_0.png)
    


fashion_news_frequency에 NONE 값과 nan 값이 있는데 다 'None'으로 통일시켜줄거임


```python
customers['fashion_news_frequency'].unique()
```




    array(['NONE', 'Regularly', nan, 'Monthly'], dtype=object)




```python
customers.loc[~customers['fashion_news_frequency'].isin(['Regularly', 'Monthly']), 'fashion_news_frequency'] = 'None'
customers['fashion_news_frequency'].unique()
```




    array(['None', 'Regularly', 'Monthly'], dtype=object)




```python
pie_data = customers[['customer_id', 'fashion_news_frequency']].groupby('fashion_news_frequency').count()
pie_data
```





  <div id="df-2a2cbb0d-837a-45cc-8c56-d4ef1b7ee5a9" class="colab-df-container">
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
      <th>customer_id</th>
    </tr>
    <tr>
      <th>fashion_news_frequency</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Monthly</th>
      <td>842</td>
    </tr>
    <tr>
      <th>None</th>
      <td>893722</td>
    </tr>
    <tr>
      <th>Regularly</th>
      <td>477416</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-2a2cbb0d-837a-45cc-8c56-d4ef1b7ee5a9')"
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
        document.querySelector('#df-2a2cbb0d-837a-45cc-8c56-d4ef1b7ee5a9 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-2a2cbb0d-837a-45cc-8c56-d4ef1b7ee5a9');
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


<div id="df-79b6d2b6-111e-41ef-af80-b9778ad2b76b">
  <button class="colab-df-quickchart" onclick="quickchart('df-79b6d2b6-111e-41ef-af80-b9778ad2b76b')"
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
        document.querySelector('#df-79b6d2b6-111e-41ef-af80-b9778ad2b76b button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

  <div id="id_ab8ebaae-21e1-46a1-aefc-1bdf25a43e16">
    <style>
      .colab-df-generate {
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

      .colab-df-generate:hover {
        background-color: #E2EBFA;
        box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
        fill: #174EA6;
      }

      [theme=dark] .colab-df-generate {
        background-color: #3B4455;
        fill: #D2E3FC;
      }

      [theme=dark] .colab-df-generate:hover {
        background-color: #434B5C;
        box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
        filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
        fill: #FFFFFF;
      }
    </style>
    <button class="colab-df-generate" onclick="generateWithVariable('pie_data')"
            title="Generate code using this dataframe."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M7,19H8.4L18.45,9,17,7.55,7,17.6ZM5,21V16.75L18.45,3.32a2,2,0,0,1,2.83,0l1.4,1.43a1.91,1.91,0,0,1,.58,1.4,1.91,1.91,0,0,1-.58,1.4L9.25,21ZM18.45,9,17,7.55Zm-12,3A5.31,5.31,0,0,0,4.9,8.1,5.31,5.31,0,0,0,1,6.5,5.31,5.31,0,0,0,4.9,4.9,5.31,5.31,0,0,0,6.5,1,5.31,5.31,0,0,0,8.1,4.9,5.31,5.31,0,0,0,12,6.5,5.46,5.46,0,0,0,6.5,12Z"/>
  </svg>
    </button>
    <script>
      (() => {
      const buttonEl =
        document.querySelector('#id_ab8ebaae-21e1-46a1-aefc-1bdf25a43e16 button.colab-df-generate');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      buttonEl.onclick = () => {
        google.colab.notebook.generateWithVariable('pie_data');
      }
      })();
    </script>
  </div>

    </div>
  </div>




customer는 뉴스 안 받는 걸 선호함(None이 젤 많음)


```python
sns.set_style('darkgrid')
f, ax = plt.subplots(figsize = (10,5))
colors = sns.color_palette('pastel')
ax.pie(pie_data.customer_id, labels = pie_data.index, colors = colors)
ax.set_facecolor('lightgrey')
ax.set_xlabel('Distribution of fashion news frequency')
plt.show()
```


    
![png](240603_HM_EDA_FIRST_LOOK_files/240603_HM_EDA_FIRST_LOOK_34_0.png)
    


# 4. 거래(Transactions)
1) feature 설명
- t_dat: 각 고객의 고유 식별자
- customer_id: 각 고객의 고유 식별자 (customers 테이블에 있음)
- article_id: 각 제품의 고유 식별자 (articles 테이블에 있음)
- price: 구매 가격
- sales_channel_id: 1 또는 2



```python
transactions.head()
```





  <div id="df-4af648ee-a4ec-4013-a404-49029973d62f" class="colab-df-container">
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
      <th>t_dat</th>
      <th>customer_id</th>
      <th>article_id</th>
      <th>price</th>
      <th>sales_channel_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2018-09-20</td>
      <td>000058a12d5b43e67d225668fa1f8d618c13dc232df0ca...</td>
      <td>663713001</td>
      <td>0.050831</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2018-09-20</td>
      <td>000058a12d5b43e67d225668fa1f8d618c13dc232df0ca...</td>
      <td>541518023</td>
      <td>0.030492</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2018-09-20</td>
      <td>00007d2de826758b65a93dd24ce629ed66842531df6699...</td>
      <td>505221004</td>
      <td>0.015237</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2018-09-20</td>
      <td>00007d2de826758b65a93dd24ce629ed66842531df6699...</td>
      <td>685687003</td>
      <td>0.016932</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2018-09-20</td>
      <td>00007d2de826758b65a93dd24ce629ed66842531df6699...</td>
      <td>685687004</td>
      <td>0.016932</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-4af648ee-a4ec-4013-a404-49029973d62f')"
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
        document.querySelector('#df-4af648ee-a4ec-4013-a404-49029973d62f button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-4af648ee-a4ec-4013-a404-49029973d62f');
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


<div id="df-300cd5e3-edbb-4000-9c91-3d05ff5a10c0">
  <button class="colab-df-quickchart" onclick="quickchart('df-300cd5e3-edbb-4000-9c91-3d05ff5a10c0')"
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
        document.querySelector('#df-300cd5e3-edbb-4000-9c91-3d05ff5a10c0 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>




price에 이상치가 있다는 걸 알 수 있다!


```python
pd.set_option('display.float_format', '{:.4f}'.format)
transactions.describe()['price']
```




    count   31788324.0000
    mean           0.0278
    std            0.0192
    min            0.0000
    25%            0.0158
    50%            0.0254
    75%            0.0339
    max            0.5915
    Name: price, dtype: float64




```python
sns.set_style('darkgrid')
f, ax = plt.subplots(figsize = (10,5))
ax = sns.boxplot(data = transactions, x = 'price', color = 'orange')
ax.set_xlabel('Price outliers')
plt.show()
```


    
![png](240603_HM_EDA_FIRST_LOOK_files/240603_HM_EDA_FIRST_LOOK_39_0.png)
    



```python
transactions_byid = transactions.groupby('customer_id').count()
```


```python
transactions_byid.sort_values(by = 'price', ascending = False)['price'][:10]
```




    customer_id
    be1981ab818cf4ef6765b2ecaea7a2cbf14ccd6e8a7ee985513d9e8e53c6d91b    1895
    b4db5e5259234574edfff958e170fe3a5e13b6f146752ca066abca3c156acc71    1441
    49beaacac0c7801c2ce2d189efe525fe80b5d37e46ed05b50a4cd88e34d0748f    1364
    a65f77281a528bf5c1e9f270141d601d116e1df33bf9df512f495ee06647a9cc    1361
    cd04ec2726dd58a8c753e0d6423e57716fd9ebcf2f14ed6012e7e5bea016b4d6    1237
    55d15396193dfd45836af3a6269a079efea339e875eff42cc0c228b002548a9d    1208
    c140410d72a41ee5e2e3ba3d7f5a860f337f1b5e41c27cf9bda5517c8774f8fa    1170
    8df45859ccd71ef1e48e2ee9d1c65d5728c31c46ae957d659fa4e5c3af6cc076    1169
    03d0011487606c37c1b1ed147fc72f285a50c05f00b9712e0fc3da400c864296    1157
    6cc121e5cc202d2bf344ffe795002bdbf87178054bcda2e57161f0ef810a4b55    1143
    Name: price, dtype: int64



전체 가격으로 비교하는 것보다 그룹끼리 비교하는 것이 좋음. Accessories하고 trousers 자체에서 가격 차이가 크게 날 수 있기 때문.   
비교를 위해 articles 데이터셋의 하위 집합을 가져와서 transactions 데이터셋과 병합할거임


```python
articles_for_merge = articles[['article_id', 'prod_name', 'product_type_name', 'product_group_name', 'index_name']]
```


```python
articles_for_merge = transactions[['customer_id', 'article_id', 'price', 't_dat']].merge(articles_for_merge, on = 'article_id', how = 'left')
```

그룹 이름별 가격의 이상치를 확인할 수 있음.   
Lower/Upper/Full body 사이에 가격 차이가 크게 남. 일반적인 의류에 비해 독특한 컬렉션일 수 있다는 걸 추측할 수 음
일부 높은 가격의 제품은 액세서리 그룹에 속함을 알 수 있음.


```python
sns.set_style('darkgrid')
f, ax = plt.subplots(figsize = (25, 18))
ax = sns.boxplot(data = articles_for_merge, x = 'price', y = 'product_group_name')
ax.set_xlabel('Price outliers', fontsize = 22)
ax.set_ylabel('Index names', fontsize = 22)
ax.xaxis.set_tick_params(labelsize = 22)
ax.yaxis.set_tick_params(labelsize = 22)
plt.show()
```


    
![png](240603_HM_EDA_FIRST_LOOK_files/240603_HM_EDA_FIRST_LOOK_46_0.png)
    


Accessories 그룹에 대한 박스플롯 가격을 확인해볼 수 있음   
가장 큰 이상치는 가방에서 발견됨. 그리고 scarf하고 기타 accessories는 나머지 garments와 가격이 크게 대조되는 제품이 있음


```python
sns.set_style("darkgrid")
f, ax = plt.subplots(figsize=(25,18))
_ = articles_for_merge[articles_for_merge['product_group_name'] == 'Accessories']
ax = sns.boxplot(data=_, x='price', y='product_type_name')
ax.set_xlabel('Price outliers', fontsize=22)
ax.set_ylabel('Index names', fontsize=22)
ax.xaxis.set_tick_params(labelsize=22)
ax.yaxis.set_tick_params(labelsize=22)
del _

plt.show()
```


    
![png](240603_HM_EDA_FIRST_LOOK_files/240603_HM_EDA_FIRST_LOOK_48_0.png)
    


Ladieswear의 평균 가격이 가장 높고 children이 가장 낮음


```python
articles_index = articles_for_merge[['index_name', 'price']].groupby('index_name').mean()
articles_index.sort_values(by = 'price', ascending=False)
```





  <div id="df-d7035562-8095-4f2d-b739-6936d07700f5" class="colab-df-container">
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
      <th>price</th>
    </tr>
    <tr>
      <th>index_name</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Ladieswear</th>
      <td>0.0328</td>
    </tr>
    <tr>
      <th>Sport</th>
      <td>0.0288</td>
    </tr>
    <tr>
      <th>Menswear</th>
      <td>0.0275</td>
    </tr>
    <tr>
      <th>Divided</th>
      <td>0.0265</td>
    </tr>
    <tr>
      <th>Ladies Accessories</th>
      <td>0.0240</td>
    </tr>
    <tr>
      <th>Children Sizes 134-170</th>
      <td>0.0219</td>
    </tr>
    <tr>
      <th>Lingeries/Tights</th>
      <td>0.0208</td>
    </tr>
    <tr>
      <th>Children Accessories, Swimwear</th>
      <td>0.0176</td>
    </tr>
    <tr>
      <th>Children Sizes 92-140</th>
      <td>0.0175</td>
    </tr>
    <tr>
      <th>Baby Sizes 50-98</th>
      <td>0.0175</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-d7035562-8095-4f2d-b739-6936d07700f5')"
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
        document.querySelector('#df-d7035562-8095-4f2d-b739-6936d07700f5 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-d7035562-8095-4f2d-b739-6936d07700f5');
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


<div id="df-4553a052-bb01-4106-be94-7677298959f3">
  <button class="colab-df-quickchart" onclick="quickchart('df-4553a052-bb01-4106-be94-7677298959f3')"
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
        document.querySelector('#df-4553a052-bb01-4106-be94-7677298959f3 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>




Shoes의 평균 가격이 가장 높고 Stationery(문구류)가 가장 낮음


```python
articles_index = articles_for_merge[['product_group_name', 'price']].groupby('product_group_name').mean()
articles_index.sort_values(by = 'price', ascending=False)
```





  <div id="df-1c84e703-a94b-46a3-bd95-326e579b5a60" class="colab-df-container">
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
      <th>price</th>
    </tr>
    <tr>
      <th>product_group_name</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Shoes</th>
      <td>0.0387</td>
    </tr>
    <tr>
      <th>Garment Full body</th>
      <td>0.0362</td>
    </tr>
    <tr>
      <th>Bags</th>
      <td>0.0333</td>
    </tr>
    <tr>
      <th>Garment Lower body</th>
      <td>0.0329</td>
    </tr>
    <tr>
      <th>Underwear/nightwear</th>
      <td>0.0279</td>
    </tr>
    <tr>
      <th>Garment Upper body</th>
      <td>0.0270</td>
    </tr>
    <tr>
      <th>Unknown</th>
      <td>0.0268</td>
    </tr>
    <tr>
      <th>Nightwear</th>
      <td>0.0254</td>
    </tr>
    <tr>
      <th>Swimwear</th>
      <td>0.0223</td>
    </tr>
    <tr>
      <th>Underwear</th>
      <td>0.0212</td>
    </tr>
    <tr>
      <th>Garment and Shoe care</th>
      <td>0.0175</td>
    </tr>
    <tr>
      <th>Interior textile</th>
      <td>0.0164</td>
    </tr>
    <tr>
      <th>Accessories</th>
      <td>0.0156</td>
    </tr>
    <tr>
      <th>Socks &amp; Tights</th>
      <td>0.0114</td>
    </tr>
    <tr>
      <th>Items</th>
      <td>0.0113</td>
    </tr>
    <tr>
      <th>Furniture</th>
      <td>0.0096</td>
    </tr>
    <tr>
      <th>Fun</th>
      <td>0.0089</td>
    </tr>
    <tr>
      <th>Cosmetic</th>
      <td>0.0058</td>
    </tr>
    <tr>
      <th>Stationery</th>
      <td>0.0032</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-1c84e703-a94b-46a3-bd95-326e579b5a60')"
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
        document.querySelector('#df-1c84e703-a94b-46a3-bd95-326e579b5a60 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-1c84e703-a94b-46a3-bd95-326e579b5a60');
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


<div id="df-7961bb6c-2711-49ff-828a-3fcbb296cd7f">
  <button class="colab-df-quickchart" onclick="quickchart('df-7961bb6c-2711-49ff-828a-3fcbb296cd7f')"
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
        document.querySelector('#df-7961bb6c-2711-49ff-828a-3fcbb296cd7f button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>




평균 가격 상위 5개 제품 그룹(Shoew, Garment Full body, Bags, Garment Lower body, Underwear/nightwear) 의 시간에 따른 평균 가격 변화도 봐볼 수 있음


```python
articles_for_merge['t_dat'] = pd.to_datetime(articles_for_merge['t_dat'])
```


```python
product_list = ['Shoes', 'Garment Full body', 'Bags', 'Garment Lower body', 'Underwear/nightwear']
colors = ['cadetblue', 'orange','mediumspringgreen', 'tomato', 'lightseagreen']
k = 0
f, ax = plt.subplots(3, 2, figsize = (20, 15))
for i in range(3):
  for j in range(2):
    try:
      product = product_list[k]
      articles_for_merge_product = articles_for_merge[articles_for_merge.product_group_name == product_list[k]]
      series_mean = articles_for_merge_product[['t_dat', 'price']].groupby(pd.Grouper(key = 't_dat', freq = 'M')).mean().fillna(0) # pd.Grouper: 데이터프레임을 특정 시간 간격으로 그룹화할 수 있
      series_std = articles_for_merge_product[['t_dat', 'price']].groupby(pd.Grouper(key = "t_dat", freq = 'M')).std().fillna(0)
      ax[i, j].plot(series_mean, linewidth=4, color=colors[k])
      ax[i, j].fill_between(series_mean.index, (series_mean.values-2*series_std.values).ravel(),
                        (series_mean.values+2*series_std.values).ravel(), color=colors[k], alpha=.1)
      ax[i, j].set_title(f'Mean {product_list[k]} price in time')
      ax[i, j].set_xlabel('month')
      ax[i, j].set_xlabel(f'{product_list[k]}')
      k += 1
    except IndexError:
      ax[i, j].set_visible(False)
plt.show()
```


    
![png](240603_HM_EDA_FIRST_LOOK_files/240603_HM_EDA_FIRST_LOOK_55_0.png)
    


# 5. 설명 및 가격이 포함된 이미지(Images with description and price)


```python
import matplotlib.image as mpimg # 이미지 파일을 읽고 표시
```


```python
max_price_ids = transactions[transactions.t_dat == transactions.t_dat.max()].sort_values('price', ascending = False).iloc[:5][['article_id', 'price']]
min_price_ids = transactions[transactions.t_dat == transactions.t_dat.min()].sort_values('price', ascending = True).iloc[:5][['article_id', 'price']]
```

가격 탑 5개 제품의 사진과 설명 (용량 없어서 생략)
