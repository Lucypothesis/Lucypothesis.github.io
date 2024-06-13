---
layout: post
title: "240611(화) H&M ML 예제코드"
subtitle: "[Tips]"
date: 2024-06-11 15:20
background: 
tag: [Tips, Github io, Notion]
---

```python
import time
import numpy as np
import pandas as pd

import multiprocessing as mp
from multiprocessing import Pool
from functools import partial
```


```python
base_path = '/content/drive/MyDrive/csv_datas/'
csv_train = f'{base_path}transactions_train.csv'
csv_sub = f'{base_path}sample_submission.csv'
csv_users = f'{base_path}customers.csv'
csv_items = f'{base_path}articles.csv'

df = pd.read_csv(csv_train, dtype={'article_id': str}, parse_dates=['t_dat'])
df_sub = pd.read_csv(csv_sub)
```

## 사용자와 상품 데이터를 관리하고 이를 활용하여 매핑을 생성함
- 매핑을 하는 이유: 데이터 처리과정에서 ID를 기반으로 다양한 데이터 소스간의 관계를 효과적으로 관리하고 참조할 수 있게 도와줌. 데이터의 일관성과 처리속도를 개선할 수 있음


```python
dfu = pd.read_csv(csv_users)
dfi = pd.read_csv(csv_items, dtype={'article_id': str})

# dfu와 dfi에서 모든 고유 customer_id와 article_id를 추출한다. 각 사용자와 상품의 고유 식별자 목록을 생성함
ALL_USERS = dfu['customer_id'].unique().tolist()
ALL_ITEMS = dfi['article_id'].unique().tolist()

# customer인덱스에서 고객ID로의 매핑을 형성함
user_to_customer_map = {user_id: customer_id for user_id, customer_id in enumerate(ALL_USERS)}
# 고객ID에서 customer인덱스로의 매핑을 생성함
customer_to_user_map = {customer_id: user_id for user_id, customer_id in enumerate(ALL_USERS)}

# 상품인덱스에서 상품ID로 매핑을 형성함
item_to_article_map = {item_id: article_id for item_id, article_id in enumerate(ALL_ITEMS)}
# 상품ID에서 상품인덱스로 매핑을 형성함
article_to_item_map = {article_id: item_id for item_id, article_id in enumerate(ALL_ITEMS)}

# 마지막으로 더이상 필요하지 않은 dfu, dfi를 삭제하여 메모리 사용을 최적화시킴
del dfu, dfi
```

## 데이터프레임에 열을 추가해서 고객ID와 상품ID를 각각 사용자ID와 품목ID로 변환함


```python
# customer_id 열에 있는 각 고객ID를 customer_to_user_map 사전을 통해 해당하는 사용자ID로 매핑함
df['user_id'] = df['customer_id'].map(customer_to_user_map)
# article_id 열에 있는 각 상품ID를 article_to_item_map 사전을 통해 해당하는 품목ID로 매핑함
df['item_id'] = df['article_id'].map(article_to_item_map)
```

# Building Model


```python
# 추천 생성할 때 고려할 유사한 사용자의 수
N_SIMILAR_USERS = 30

# 최소 3번 이상의 구매이력이 있는 사용자만을 대상으로 분석 진행
MINIMUM_PURCHASES = 3

# 데이터 필터링할 시작 날짜 지정
START_DATE = '2020-08-21'

# 이미 구매한 항목을 추천 목록에서 제외할지 여부를 결정하는 불리언 변수
DROP_PURCHASED_ITEMS = False

# 사용자 자신을 자신의 이웃 목록에서 제외할지 여부
DROP_USER_FROM_HIS_NEIGHBORHOOD = False

# 시스템을 테스트모드로 실행할지 여부
TEST_RUN = False

# 테스트 실행시 고려할 데이터의 크기 지정
TEST_SIZE = 1000
```


```python
# 중첩리스트(리스트의 리스트)를 입력으로 받아 그 내용을 단일 리스트로 만듦
def flatten(l):
    return [item for sublist in l for item in sublist]

# 두 사용자의 구매 목록을 나타내는 두 벡터(리스트)를 비교해서 유사도 계산함
def compare_vectors(v1, v2):
    intersection = len(set(v1) & set(v2))
    denominator = np.sqrt(len(v1) * len(v2))
    # 0에서 1사이의 유사도 값 반환함
    return intersection / denominator
```

## User-based collaborative filtering 추천시스템의 핵심 요소들임


```python
# 특정 사용자와 유사한 사용자들을 찾아 그들의 유사도 점수와 함께 반환함
# u: 사용자ID / v: 해당 사용자의 구매항목ID목록을 나타내는 벡터 / dfh: 모든 사용자의 거래 이력을 나타내는 데이터프레임
def get_similar_users(u, v, dfh):
    # 데이터프레임 dfh의 각 행(다른 사용자의 구매 목록)에 대해 compare_vectors를 적용해서 주어진 사용자와의 유사도 계산함
    similar_users = dfh.apply(lambda v_other: compare_vectors(v, v_other)).sort_values(ascending=False).head(N_SIMILAR_USERS + 1)

    if DROP_USER_FROM_HIS_NEIGHBORHOOD:
      # 입력 사용자 자신을 결과 목록에서 제외함
        similar_users = similar_users[similar_users.index != u]

    # 유사한 사용자의 ID목록과 그들의 유사도 점수 목록을 튜플로 반환함
    return similar_users.index.tolist(), similar_users.tolist()

# 주어진 사용자에게 추천할 상품 목록을 생성함
def get_items(u, v, dfh):

    # i, n은 전역변수임. global 키워드를 사용하면 함수 외부에서 정의된 전역 범위의 변수를 참조하고 수정할 수 있게
    global i, n
    # get_similar_users 함수를 호출하여 입력된 사용자 u에 대해
    users, scores = get_similar_users(u, v, dfh)
    # 유사도 점수가 높은 사용자목록(users)와 그 점수(scores)를 받아옴. 이 점수는 사용자 u와 다른 사용자들과의 구매 패턴 유사도를 반영함
    df_nn = pd.DataFrame({'user': users, 'score': scores})
    # 유사 사용자들이 구매한 아이템을 수집함. df_nn 데이터프레임을 생성하고, 각 유사사용자(user)의 구매 아이템 목록을 dfh 데이터프레임에서 추출해서 items 열에 저장함
    df_nn['items'] = df_nn.apply(lambda row: dfh.loc[row.user], axis=1)
    # 각 아이템에 대해 유사도 점수(score)을 가중치로 적용함. weighted_items 열에서는 각 유사 사용자가 구매한 아이템 각각에 대해 해당 사용자와의 유사도 점수를 곱하여 새로운 튜플(item, score)을 생성함
    df_nn['weighted_items'] = df_nn.apply(lambda row: [(item, row.score) for item in row['items']], axis=1)

    # flaten 함수를 사용해서 weighted_items 목록을 하나의 리스트로 평탄화하고, 이를 item과 score열을 포함하는 데이터프레임으로 변환함
    # groupby('item')과 sum()을 사용해 같은 아이템에 대한 점수를 합산하고, sort_values로 점수가 높은 순으로 아이템을 정렬함
    recs = pd.DataFrame(flatten(df_nn['weighted_items'].tolist()), columns=['item', 'score']).groupby('item')['score'].sum().sort_values(ascending=False)
    if DROP_PURCHASED_ITEMS:
        # 'u'가 이미 구매한 아이템들은 추천 목록에서 제외함. 사용자에게 새로운 아이템을 추천하기 위한 코드임
        recs = recs[~recs.index.isin(v)]
    # 이건 그냥 진행상태 출력하는 코드임
    i +=1
    if i % 200 == 0:
        pid = mp.current_process().pid
        print(f"[PID {pid:>2d}] Finished {i:3d} / {n:5d} - {i/n*100:3.0f}%")

    # 최종적으로 점수가 높은 상위 12개 항목을 반환함
    return recs.head(12).index.tolist()
```


```python
# 여러 사용자의 ID 목록을 받아 각 사용자에 대해 개인화된 추천 항목 목록을 생성함
# user_dids: 추천을 생성할 사용자ID의 리스트 / dfh: 모든 사용자의 거래 이력이 담긴 데이터프레임
def get_items_chunk(user_ids: np.array, dfh: pd.DataFrame):

    global i, n
    i = 0
    n = len(user_ids)

    # 현재 작업중인 프로세스ID(pid)와 함께 작업을 시작하는 사용자의 총 수를 로깅함. 작업의 시작을 알리고 진행상태를 모니터링 하는데 사용됨
    pid = mp.current_process().pid
    print(f"[PID {pid:>2d}] Started working with {n:5d} users")

    # dfh 데이터프레임에서 user_ids에 해당하는 사용자의 정보를 추출하고 인덱스를 재설정함
    # -> 이렇게 하면 각 행이 하나의 사용자를 나타내도록 구성되게 됨
    df_user_vectors = pd.DataFrame(dfh.loc[user_ids]).reset_index()
    # df_users_vectors의 각 행에 대해 get_items 함수를 적용하여 해당 사용자에게 추천할 항목 목록을 생성함
    # apply 함수는 lambda를 사용하여 각 행마다 get_items함수를 호출하는데 이때 user_id와 item_id를 인자로 넘겨줌
    # 추천 결과는 df_user_vectors의 'recs' 열에 저장됨
    df_user_vectors['recs'] = df_user_vectors.apply(lambda row: get_items(row.user_id, row.item_id, dfh), axis=1)

    # df_user_vectors 데이터프레임을 user_id를 인덱스로 하여 재구성하고, recs열만을 포함하는 pd.Series객체로 반환함
    # -> 여기에 각 사용자ID와 그에 해당하는 추천 아이템 목록을 포함함
    return df_user_vectors.set_index('user_id')['recs']
```


```python
# 사용자 리스트에 대한 추천 항목을 병렬 방식으로 생성하는 과정을 정의함. 함수의 실행시간을 단축시키는데 중점을 두고 있음
def get_recommendations(users: list, dfh: pd.DataFrame):
    """
    Returns:
        pd.DataFrame with index user_id and list of item_id (recommendations) as value
    """
    # 함수가 시작되는 시점의 시간을 time.time()을 사용해 기록함. 전체 함수의 실행 시간을 측정하기 위한 기준점이 됨
    time_start = time.time()

    # 입력된 users 리스트를 시스템의 CPU수만큼 분할하여 각 CPU에서 처리할 사용자 데이터의 청크를 생성함. 각 프로세스가 처리할 데이터양을 균등하게 분배해서 리소스 화룡을 최대화 하려는거임
    user_chunks = np.array_split(users, mp.cpu_count())

    # get_items_chunk 함수에 dfh를 고정 인자로 설정하여 새로운 함수 f를 생성함
    # -> 이렇게 하면 p.map 호출시 추가로 인자를 전달할 필요 없이 각 청크를 처리할 수 있음
    f = partial(get_items_chunk, dfh=dfh)
    # 사용가능한 CPU 수만큼의 프로세스 풀 생성함. 이 풀을 통해 병렬처리를 시작하는거임
    with Pool(mp.cpu_count()) as p:
        # 병렬처리 풀에서 f함수를 각 사용자 청크에 대해서 실행함
        # -> 각 사용자에 대한 추천 결과를 반환받음
        res = p.map(f, user_chunks)

    # 각 청크에서 반환된 결과(각 사용자의 추천아이템 목록)을 하나의 pd.DataFrame으로 병합함
    df_rec = pd.DataFrame(pd.concat(res))

    # 함수의 종료 시점과 시작시점의 차이를 계산하여 소요시간을 분단위로 나타냄
    elapsed = (time.time() - time_start) / 60
    # 실행시간과 처리된 사용자수를 알림
    print(f"Finished get_recommendations({len(users)}). It took {elapsed:5.2f} mins")
    return df_rec
```

## UUCF(User-User Collaborative Filtering) 모델을 실행하는데 사용됨
특정 날짜 이후의 거래 데이터를 기반으로 사용자에게 추천을 제공함


```python
def uucf(df, start_date=START_DATE):

    # start_date 이후의 거래데이터만을 필터링함
    df_small = df[df['t_dat'] > start_date]
    # 데이터의 행수를 출력하여 얼마나 많은 데이터가 보존되었는지 로깅함
    print(f"Kept data from {start_date} on. Total rows: {len(df_small)}")

    # H stands for "Transaction history"
    # dfh is a series of user_id => list of item_id (the list of purchases in order)
    # 사용자ID별로 구매한 고유한 아이템 목록 생성. item_id를 그룹화한 뒤 중복 제거하고 리스트로 변환함
    dfh = df_small.groupby("user_id")['item_id'].apply(lambda items: list(set(items)))
    # 최소구매기준(MINIMUM_PURCHASES) 이상의 아이템을 구매한 사용자만을 필터링함
    dfh = dfh[dfh.str.len() >= MINIMUM_PURCHASES]

    # 만약 테스트모드가 활성화되어있다면
    if TEST_RUN:
        # TEST_SIZE 만큼의 데이터를 사용하여 실행함
        print("WARNING: TEST_RUN is True. It will be a toy execution.")
        dfh = dfh.head(TEST_SIZE)

    # 거래이력데이터프레임(dfh)에서 인덱스(사용자ID)를 추출하여 리스트로 변환함. 이 리스트는 추천을 받을 사용자들의 목록임
    users = dfh.index.tolist()
    # 추출된 사용자 리스트의 길이(사용자수) 계산함
    n_users = len(users)
    print(f"Total users in the time frame with at least {MINIMUM_PURCHASES}: {n_users}")

    # 사용자목록(users)와 거래이력(dfh)를 활용하여 get_recommendations 함수를 호출해서 각 사용자별 추천아이템 목록을 포함하는 데이터프레임을 반홚받음
    df_rec = get_recommendations(users, dfh)

    # 고객ID 매핑
    # df_rec 데이ㅓ프레임의 인덱스(사용자ID)를 user_to_Customer_map을 사용하여 고객ID로 변환함
    # -> 사용자ID에서 보다 의미 있는 고객ID로 데이터를 변환하여 외부 시스템과의 호환성을 높이기 위함임
    df_rec['customer_id'] = df_rec.index.map(user_to_customer_map)

    # 예측결과 매핑
    # 추천된 아이템ID목록(recs)을 item_to_article_map을 사용하여 실제 상품ID로 변환함
    # -> 내부적으로 사용되는 아이템ID를 외부에서 인식할 수 있는 상품ID로 매핑하기 위함임
    df_rec['prediction'] = df_rec['recs'].map(lambda l: [item_to_article_map[i] for i in l])

    # Submission ready dataframe
    # 최종 데이터프레임에서 인덱스 리셋하고 필요한 열만 선택해서 새로운 데이터프레임 생성함. 각 고객ID별로 추천된 상품ID 목록을 포함함
    df_rec.reset_index(drop=True)[['customer_id', 'prediction']]

    # 결과 제출하기 위해 딱 준비된 데이터프레임
    return df_rec
```


```python
df_recs = uucf(df)
```

    Kept data from 2020-08-21 on. Total rows: 1190911
    Total users in the time frame with at least 3: 141572
    [PID 1568] Started working with 70786 users
    [PID 1569] Started working with 70786 users
    [PID 1568] Finished 200 / 70786 -   0%
    [PID 1569] Finished 200 / 70786 -   0%
    [PID 1568] Finished 400 / 70786 -   1%
    [PID 1569] Finished 400 / 70786 -   1%
    [PID 1568] Finished 600 / 70786 -   1%
    [PID 1569] Finished 600 / 70786 -   1%
    [PID 1568] Finished 800 / 70786 -   1%
    [PID 1569] Finished 800 / 70786 -   1%
    [PID 1568] Finished 1000 / 70786 -   1%
    [PID 1569] Finished 1000 / 70786 -   1%
    [PID 1568] Finished 1200 / 70786 -   2%
    [PID 1569] Finished 1200 / 70786 -   2%
    [PID 1568] Finished 1400 / 70786 -   2%
    [PID 1569] Finished 1400 / 70786 -   2%
    [PID 1568] Finished 1600 / 70786 -   2%
    [PID 1569] Finished 1600 / 70786 -   2%
    [PID 1568] Finished 1800 / 70786 -   3%
    [PID 1569] Finished 1800 / 70786 -   3%
    [PID 1568] Finished 2000 / 70786 -   3%
    [PID 1569] Finished 2000 / 70786 -   3%
    [PID 1568] Finished 2200 / 70786 -   3%
    [PID 1569] Finished 2200 / 70786 -   3%
    [PID 1568] Finished 2400 / 70786 -   3%
    [PID 1569] Finished 2400 / 70786 -   3%
    [PID 1568] Finished 2600 / 70786 -   4%
    [PID 1569] Finished 2600 / 70786 -   4%
    [PID 1568] Finished 2800 / 70786 -   4%
    [PID 1569] Finished 2800 / 70786 -   4%
    [PID 1568] Finished 3000 / 70786 -   4%
    [PID 1569] Finished 3000 / 70786 -   4%
    [PID 1568] Finished 3200 / 70786 -   5%
    [PID 1569] Finished 3200 / 70786 -   5%
    [PID 1568] Finished 3400 / 70786 -   5%
    [PID 1569] Finished 3400 / 70786 -   5%
    [PID 1568] Finished 3600 / 70786 -   5%
    [PID 1569] Finished 3600 / 70786 -   5%
    [PID 1568] Finished 3800 / 70786 -   5%
    [PID 1569] Finished 3800 / 70786 -   5%
    [PID 1568] Finished 4000 / 70786 -   6%
    [PID 1569] Finished 4000 / 70786 -   6%
    [PID 1568] Finished 4200 / 70786 -   6%
    [PID 1569] Finished 4200 / 70786 -   6%
    [PID 1568] Finished 4400 / 70786 -   6%
    [PID 1569] Finished 4400 / 70786 -   6%
    [PID 1568] Finished 4600 / 70786 -   6%
    [PID 1569] Finished 4600 / 70786 -   6%
    [PID 1568] Finished 4800 / 70786 -   7%
    [PID 1569] Finished 4800 / 70786 -   7%
    [PID 1568] Finished 5000 / 70786 -   7%
    [PID 1569] Finished 5000 / 70786 -   7%
    [PID 1568] Finished 5200 / 70786 -   7%
    [PID 1569] Finished 5200 / 70786 -   7%
    [PID 1568] Finished 5400 / 70786 -   8%
    [PID 1569] Finished 5400 / 70786 -   8%
    [PID 1568] Finished 5600 / 70786 -   8%
    [PID 1569] Finished 5600 / 70786 -   8%
    [PID 1568] Finished 5800 / 70786 -   8%
    [PID 1569] Finished 5800 / 70786 -   8%
    [PID 1568] Finished 6000 / 70786 -   8%
    [PID 1569] Finished 6000 / 70786 -   8%
    [PID 1568] Finished 6200 / 70786 -   9%
    [PID 1569] Finished 6200 / 70786 -   9%
    [PID 1568] Finished 6400 / 70786 -   9%
    [PID 1569] Finished 6400 / 70786 -   9%
    [PID 1568] Finished 6600 / 70786 -   9%
    [PID 1569] Finished 6600 / 70786 -   9%
    [PID 1568] Finished 6800 / 70786 -  10%
    [PID 1569] Finished 6800 / 70786 -  10%
    [PID 1568] Finished 7000 / 70786 -  10%
    [PID 1569] Finished 7000 / 70786 -  10%
    [PID 1568] Finished 7200 / 70786 -  10%
    [PID 1569] Finished 7200 / 70786 -  10%
    [PID 1568] Finished 7400 / 70786 -  10%
    [PID 1569] Finished 7400 / 70786 -  10%
    [PID 1568] Finished 7600 / 70786 -  11%
    [PID 1569] Finished 7600 / 70786 -  11%
    [PID 1568] Finished 7800 / 70786 -  11%
    [PID 1569] Finished 7800 / 70786 -  11%
    [PID 1568] Finished 8000 / 70786 -  11%
    [PID 1569] Finished 8000 / 70786 -  11%
    [PID 1568] Finished 8200 / 70786 -  12%
    [PID 1569] Finished 8200 / 70786 -  12%
    [PID 1568] Finished 8400 / 70786 -  12%
    [PID 1569] Finished 8400 / 70786 -  12%
    [PID 1568] Finished 8600 / 70786 -  12%
    [PID 1569] Finished 8600 / 70786 -  12%
    [PID 1568] Finished 8800 / 70786 -  12%
    [PID 1569] Finished 8800 / 70786 -  12%
    [PID 1568] Finished 9000 / 70786 -  13%
    [PID 1569] Finished 9000 / 70786 -  13%
    [PID 1568] Finished 9200 / 70786 -  13%
    [PID 1569] Finished 9200 / 70786 -  13%
    [PID 1568] Finished 9400 / 70786 -  13%
    [PID 1569] Finished 9400 / 70786 -  13%
    [PID 1568] Finished 9600 / 70786 -  14%
    [PID 1569] Finished 9600 / 70786 -  14%
    [PID 1568] Finished 9800 / 70786 -  14%
    [PID 1569] Finished 9800 / 70786 -  14%
    [PID 1568] Finished 10000 / 70786 -  14%
    [PID 1569] Finished 10000 / 70786 -  14%
    [PID 1568] Finished 10200 / 70786 -  14%
    [PID 1569] Finished 10200 / 70786 -  14%
    [PID 1568] Finished 10400 / 70786 -  15%
    [PID 1569] Finished 10400 / 70786 -  15%
    [PID 1568] Finished 10600 / 70786 -  15%
    [PID 1569] Finished 10600 / 70786 -  15%
    [PID 1568] Finished 10800 / 70786 -  15%
    [PID 1569] Finished 10800 / 70786 -  15%
    [PID 1568] Finished 11000 / 70786 -  16%
    [PID 1569] Finished 11000 / 70786 -  16%
    [PID 1568] Finished 11200 / 70786 -  16%
    [PID 1569] Finished 11200 / 70786 -  16%
    [PID 1568] Finished 11400 / 70786 -  16%
    [PID 1569] Finished 11400 / 70786 -  16%
    [PID 1568] Finished 11600 / 70786 -  16%
    [PID 1569] Finished 11600 / 70786 -  16%
    [PID 1568] Finished 11800 / 70786 -  17%
    [PID 1569] Finished 11800 / 70786 -  17%
    [PID 1568] Finished 12000 / 70786 -  17%
    [PID 1569] Finished 12000 / 70786 -  17%
    [PID 1568] Finished 12200 / 70786 -  17%
    [PID 1569] Finished 12200 / 70786 -  17%
    [PID 1568] Finished 12400 / 70786 -  18%
    [PID 1569] Finished 12400 / 70786 -  18%
    [PID 1568] Finished 12600 / 70786 -  18%
    [PID 1569] Finished 12600 / 70786 -  18%
    [PID 1568] Finished 12800 / 70786 -  18%
    [PID 1569] Finished 12800 / 70786 -  18%
    [PID 1568] Finished 13000 / 70786 -  18%
    [PID 1569] Finished 13000 / 70786 -  18%
    [PID 1568] Finished 13200 / 70786 -  19%
    [PID 1569] Finished 13200 / 70786 -  19%
    [PID 1568] Finished 13400 / 70786 -  19%
    

## Fill the remainder with another model


```python
csv_fill = '../input/h-m-content-based-12-most-popular-items-0-007/submission.csv'
df_fill = pd.read_csv(csv_fill)
```

## 추천시스템에서 생성된 결과를 보완하고 최종적으로 사용자에게 제공할 추천리스트를 완성하는 과정


```python
# 입력된 시쿼스(리스트 등)에서 중복을 제거하면서 원래의 순서를 유지하는 함수
def drop_duplicates(seq):
    # seen이라는 빈 집합을 사용하여 이미 처리한 요소를 기록함
    # 집합을 사용하는 이유: 요소의 존재 여부를 빠르게 검사할 수 있음. 시간복잡도 O(1)임
    seen = set()
    # 집합의 add 메서드를 seen_add 라는 변수에 바인딩함. 함수 호출 오버헤드를 줄일 수 있고 코드 내에서 직접 seen.add()를 호출하는 것보다 더 빠르게 실행됨
    seen_add = seen.add
    # x가 seen 집합에 없으면 그 요소를 결과 리스트에 포함시키고 seen 집합에 추가함
    # x in seen or seen_add(x) -> x in seen이 참이면 seen_add(x)는 실행되지 않음. 만약 x가 seen에 없다면 seen_add(x)가 실행되어 x를 seen에 추가하고 seen_add(x)는 항상 None을 반환하므로 not None은 참이 되는 원리임
    # -> 이 방식으로 각 요소가 최초로 등장했을 때만 결과 리스트에 추가되며 중복은 자동으로 제거됨
    return [x for x in seq if not (x in seen or seen_add(x))]
```


```python
# 추천시스템에서 보완 모델의 결과를 사용하여 기본 추천모델의 결과를 향상시키는데 사용됨. 다양한 소스에서 추천을 조합하여 다양하고 만족스러운 추천을 제공하는 것임
def fill_row(row):
    # uucf: 모델을 통해 생성된 추천 아이템 목록
    uucf = row['prediction_uucf']
    # 보완 모델을 통해 생성된 추천 아이템 목록
    fill = row['prediction_fill'].split()
    new_list = drop_duplicates(uucf + fill)[:12]
    return ' '.join(new_list)

# 추천시스템에서 사용자에게 최종적으로 제공할 추천 목록 조정하고 완성하는 함수
def fill(df_recs, df_fill):
    # 각 사용자의 추천 목록 길이를 계산하고 len열에 저장함
    # -> 나중에 충분한 수의 추천이 있는지 판단하는 기준으로 사용됨
    df_recs['len'] = df_recs['prediction'].str.len()
    # df_fill(보안 모델의 추천결과)과 df_recs(기존 모델의 추천결과)를 customer_id를 기준으로 왼쪽조인함. 각 데이터프레임의 열 이름이 겹치지 않도록 접미사(_fill과 _uucf)를 추가함
    df_recs = pd.merge(df_fill, df_recs, how='left', on='customer_id', suffixes=('_fill', '_uucf'))


    # UUCF 추천이 하나도 없는 경우: 보완모델의 추천(prediction_fill)을 그대로 사용함
    df_recs.loc[df_recs['prediction_uucf'].isnull(), 'prediction'] = df_recs['prediction_fill']

    # UUCF에서 12개의 추천을 모두 제공한 경우: 해당 추천(prediction_uucf)을 최종 결과로 사용함
    mask = df_recs['prediction_uucf'].notnull() & (df_recs['len'] == 12)
    # 설정한 mask를 사용하여 데이터프레임의 해당 행을 선택하고 prediction 열을 prediction_uucf 값으로 업데이트함
    df_recs.loc[mask, 'prediction'] = df_recs['prediction_uucf']


    # UUCF에서 12개 미만으로 추천한 경우: fill_row 함수를 적용하여 UUCF 추천과 보완모델 추천을 결합함
    fill_mask = df_recs['prediction_uucf'].notnull() & (df_recs['len'] < 12)
    # 조건에 맞는 행에 대해 fill_row 함수를 적용하고 결과를 prediction 열에 저장함
    df_recs.loc[fill_mask, 'prediction'] = df_recs[fill_mask].apply(fill_row, axis=1)

    # 불필요한 중간 열을 삭제하고 각 고객ID에 대해 최종적으로 결정된 추천아이템 목록을 반환함
    return df_recs.drop(['prediction_uucf', 'prediction_fill', 'len', 'recs'], axis=1)
```


```python
# UUCF 결과와 보완모델의 결과를 결합해서 최종적으로 추천 데이터프레임 만듦
df_sub = fill(df_recs, df_fill)
df_sub.head()
```


```python
df_sub.shape
```


```python
# 제출할 csv 파일 형
df_sub.to_csv("uucf.csv", index=False)
```
