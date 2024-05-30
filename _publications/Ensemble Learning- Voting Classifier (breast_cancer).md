---
layout: post
title: "240530(목) 03 앙상블 학습"
subtitle: "[Tips]"
date: 2024-05-30 23:54
background: 
tag: [Tips, Github io, Notion]
---

## 03 앙상블 학습

앙상블 학습이란! 여러 개의 Classifier를 생성하고 그 예측을 결합해서 더 정확하게 예측하는 기법! 여러 사람이 모이면 더 좋은 결과가 나온다는 집단 지성 느낌이라고 생각하면 된다.
이미지, 영상, 음성 등의 비정형 데이터 분류는 딥러닝이 뛰어난 성능을 보이고 있지만 대부분의 **정형 데이터 분류** 시에는 앙상블이 뛰어난 성능을 나타내고 있음.

대표적인 앙상블 알고리즘: RandomForest, GradientBoosting   

매력적인 모델: XGBoost, LightBGM(유사한 예측성능, 훨씬 빠른 수행 속도), Stacking(여러 가지 모델 결과를 기반으로 메타 모델 수립)   

앙상블 학습 유형: 1. **Voting** / 2. **Bagging** / 3. **Boosting**

Voting, Bagging 둘 다 여러 개의 분류기가 투표를 해서 최종 결과를 예측함. Voting은 다른 알고리즘 분류기를 결합하고 Bagging은 같은 알고리즘 분류기를 결합하지만 데이터 샘플링을 다르게 가져가서 학습하여 Voting을 수행함. Bagging의 대표적인 방식이 RandomForest임   

![Hard & Soft Voting](images/1.png)

Bootstrapping: Bagging 분류기에서 개별 Classifier에게 데이터를 샘플링해서 추출하는 방식

교차검증은 dataset 간에 중첩을 허용하지 않지만 Bagging은 중첩을 허용함

**Boosting**: 여러 개의 분류기가 순차적으로 학습을 하지만 예측이 틀린 데이터에 대해서는 올바르게 예측할 수 있도록 다음 분류기에 가중치를 boosting 하면서 학습을 진행함. 예측 성능이 뛰어남. 대표적인 boosting 모델은 XGBoost, LightBGM이 있음.

Voting 유형- 1. **Hard Voting** / 2. **Soft Voting**
Hard Voting은 다수결, Soft Voting은 확률 평균내서 구함. 일반적으로 Soft Voting이 쓰임. Hard Voting보다 예측 성능이 좋기 때문

![Hard & Soft Voting](images/2.png)

### sklearn은 Voting 방식의 앙상블을 구현한 VotingClassifier 클래스를 제공함. 아래는 위스콘신 유방암 데이터 세트를 예측 분석하는 코드임


```python
import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
```

    C:\Users\yeoni\anaconda3\lib\site-packages\scipy\__init__.py:138: UserWarning: A NumPy version >=1.16.5 and <1.23.0 is required for this version of SciPy (detected version 1.24.3)
      warnings.warn(f"A NumPy version >={np_minversion} and <{np_maxversion} is required for this version of "
    


```python
cancer = load_breast_cancer()
```


```python
data_df = pd.DataFrame(cancer.data, columns = cancer.feature_names)
data_df.head(3)
```




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
      <th>mean radius</th>
      <th>mean texture</th>
      <th>mean perimeter</th>
      <th>mean area</th>
      <th>mean smoothness</th>
      <th>mean compactness</th>
      <th>mean concavity</th>
      <th>mean concave points</th>
      <th>mean symmetry</th>
      <th>mean fractal dimension</th>
      <th>...</th>
      <th>worst radius</th>
      <th>worst texture</th>
      <th>worst perimeter</th>
      <th>worst area</th>
      <th>worst smoothness</th>
      <th>worst compactness</th>
      <th>worst concavity</th>
      <th>worst concave points</th>
      <th>worst symmetry</th>
      <th>worst fractal dimension</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>17.99</td>
      <td>10.38</td>
      <td>122.8</td>
      <td>1001.0</td>
      <td>0.11840</td>
      <td>0.27760</td>
      <td>0.3001</td>
      <td>0.14710</td>
      <td>0.2419</td>
      <td>0.07871</td>
      <td>...</td>
      <td>25.38</td>
      <td>17.33</td>
      <td>184.6</td>
      <td>2019.0</td>
      <td>0.1622</td>
      <td>0.6656</td>
      <td>0.7119</td>
      <td>0.2654</td>
      <td>0.4601</td>
      <td>0.11890</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20.57</td>
      <td>17.77</td>
      <td>132.9</td>
      <td>1326.0</td>
      <td>0.08474</td>
      <td>0.07864</td>
      <td>0.0869</td>
      <td>0.07017</td>
      <td>0.1812</td>
      <td>0.05667</td>
      <td>...</td>
      <td>24.99</td>
      <td>23.41</td>
      <td>158.8</td>
      <td>1956.0</td>
      <td>0.1238</td>
      <td>0.1866</td>
      <td>0.2416</td>
      <td>0.1860</td>
      <td>0.2750</td>
      <td>0.08902</td>
    </tr>
    <tr>
      <th>2</th>
      <td>19.69</td>
      <td>21.25</td>
      <td>130.0</td>
      <td>1203.0</td>
      <td>0.10960</td>
      <td>0.15990</td>
      <td>0.1974</td>
      <td>0.12790</td>
      <td>0.2069</td>
      <td>0.05999</td>
      <td>...</td>
      <td>23.57</td>
      <td>25.53</td>
      <td>152.5</td>
      <td>1709.0</td>
      <td>0.1444</td>
      <td>0.4245</td>
      <td>0.4504</td>
      <td>0.2430</td>
      <td>0.3613</td>
      <td>0.08758</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 30 columns</p>
</div>



### Logistic Regression과 KNN을 기반, Soft Voting 방식 사용


```python
# 개별 모델은 로지스틱 회귀와 KNN
lr_clf = LogisticRegression(solver = 'liblinear')
knn_clf = KNeighborsClassifier(n_neighbors = 8)

vo_clf = VotingClassifier(estimators = [('LR', lr_clf), ('KNN', knn_clf)], voting = 'soft')
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size = 0.2, random_state = 156)
```

- VotingClassifier는 주요 생성 인자로 estimators와 voting 값을 입력받음
    - estimators: 리스트 형식임. Voting에 사용될 여러 개의 Classifier 개체들을 튜플 형식으로 입력받음
    - voting: 'hard'일 때 hard voting, 'soft'일 때 soft voting임. 기본은 hard!


```python
# VotingClassifier 학습/예측/평가
vo_clf.fit(X_train, y_train)
pred = vo_clf.predict(X_test)
print('VotingClassifier 정확도: {0:.4f}'.format(accuracy_score(y_test, pred)))
```

    VotingClassifier 정확도: 0.9561
    


```python
# LogisticRegression, KNeighborsClassifier의 학습/예측/평가
classifiers = [lr_clf, knn_clf]
for classifier in classifiers:
    classifier.fit(X_train, y_train)
    pred = classifier.predict(X_test)
    class_name = classifier.__class__.__name__
    print('{0} 정확도: {1:.4f}'.format(class_name, accuracy_score(y_test, pred)))
```

    LogisticRegression 정확도: 0.9474
    KNeighborsClassifier 정확도: 0.9386
    

VotingClassifier의 정확도가 조금 더 높게 나왔다. 근데 voting으로 여러 개 결합한다고 해서 항상 조건 기반 분류기(여기서는 lr, knn)보다 예측 성능이 향상되는 건 아님. 데이터의 특성과 분포 등 다양한 요건에 따라 오히려 기반 분류기 중 가장 좋은 분류기의 성능이 voting 했을 때보다 나을 수 있음
하지만 그럼에도 불구하고 앙상블 방법을 쓰는 이유: 더 유연하게 대처할 수 있기 때문! 팀플할 때 능력치 좋은 팀원들로 구성된 팀과 다양한 경험을 가진 팀원들로 구성된 팀 중 후자가 팀플하기에는 더 좋을 수 있는 것과 비슷함.

아직 지식이 영글지 못했지만 책의 문단을 읽고 앙상블 기법이 와닿아서 그대로 적어봤다. (책 215쪽)

"ML 모델의 성능은 다양한 테스트 데이터에 의해 검증되므로 어떻게 높은 유연성을 가지고 현실에 대처할 수 있는가가 중요한 ML 모델의 평가요소가 된다. 이런 관점에서 편향-분산 트레이드오프는 ML 모델이 극복해야 할 중요 과제입니다. 보팅과 스태킹 등은 서로 다른 알고리즘을 기반으로 하고 있지만, 배깅과 부스팅은 대부분 결정 트리 알고리즘을 기반으로 합니다. 결정 트리 알고리즘은 쉽고 직관적인 분류 기준을 가지고 있지만 정확한 예측을 위해 학습 데이터의 예외 상황에 집착한 나머지 오히려 과적합이 발생해 실제 테스트 데이터에서 예측 성능이 떨어지는 현상이 발생하기 쉽다고 앞에서 말했습니다. 하지만 앙상블 학습에서는 이 같은 결정 트리 알고리즘의 단점을 수십~수천 개의 매우 많은 분류기를 결합해 다양한 상황을 학습하게 함으로써 극복하고 있습니다. 결정 트리 알고리즘의 장점은 그대로 취하고 단점은 보완하면서 편향-분산 트레이드오프의 효과를 극대화할 수 있다는 것입니다."

나중에 이 문단을 나만의 언어로 표현할 수 있게 되길 바라며..

앙상블 기법 끝!
