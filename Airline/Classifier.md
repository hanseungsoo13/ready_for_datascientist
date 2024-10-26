## 머신러닝 정리

Feature Engineering부터 Modeling까지 기본적인 개념들을 다시 한 번 정리하고, 분석기법을 논리적이게 활용할 pipeline을 짠다.

### Data Processing
1. **Outliers**
데이터의 일반적인 추세를 따르지 않거나 평균치에 영향을 줄 수 있기 때문에 제거하거나, 다른 값으로 대치한다 (예를 들면 clip?)
- Scatter Plot, Box plot 등을 활용해 outlier 탐색
``` 
#IQR 결측치 제거
Q1 = df['column'].quantile(0.25)
Q3 = df['column'].quantile(0.75)

IQR = Q3 - Q1

upper_limit = Q3 + 1.5*IQR
lower_limit = Q1 - 1.5*IQR

df[df['column']>limit_value]

#IQR 결측치 대치
df['column'].clip(lower_limit,upper_limit)
```

2. **Null**
누락된 값이 있으면 모델이 돌아가지 않기 때문에 이 값을 어떤 값으로 대치하느냐도 매우 중요하다. 평균값, 중앙값, 최빈값 으로 대치할 수 있다.
``` 
data.isnull() #null값 확인
data.fillna(0) #null값 0으로 대체
```

3. **Scaling**
상대적으로 값이 큰 변수에 가중치가 쏠리는 것을 방지하기 위해 모든 값을 적절하게 표준화 시켜주는 것이 중요하다. 
> 수치데이터에만 적용하는 것이 중요하다.

4. **Encoding**
카테고리 변수들을 수치형으로 변환시켜주는 방법론
- Label Encoding: 순서가 의미가 있는 categorical feature들에 대해서 적용해주는 것이 타당
- OnehotEncoding: **순서가 의미가 없는** 모두가 동등한 위치에 있는 categorical feature들에 대해서 적용해주는 것이 좋다.

5. **수치형 카테고리화**
```
bins = [1,2,3,4,5]
binned = pd.cut(df['column'],bins) #bins에 int값 넣어주면 bins개수에 의해 분할
```

6. **Imbalance dataset 처리**
타겟 변수의 불균형이 심한 경우 Minor Class를 명확하게 예측하지 못하는 경우가 많다. Minor class를 oversampling하여 Minor Class와 Major Class의 차이를 명확히 예측할 수 있도록 하는 것이 중요하다. -> Feature 생성 다하고 마지막에 적용
- **SMOTE**
minor class data들의 최근접 이웃을 활용하여 새로운 데이터 생성. 근접한 minor class의 data들과 일정한 거리에 떨어진 위치에 데이터를 생성함
- **Tomek Link**
두 샘플 A와 B가 있을 때 minority와 가까운 majority sample을 제외시키는 방법
```
#SMOTE
from imlearn.over_sampling import SMOTE
smote = SMOTE(random_state=11)
X_train_over, y_train_over = smote.fit_sample(X_train, y_train)

#SMOTETomek
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import TomekLinks

smoteto = SMOTETomek(tomek=TomekLinks(sampling_strategy='majority'))
X_churn_smt, Y_churn_smt = smoteto.fit_resample(X_churn, Y_churn)
```

### Data Sampling
1. **Holdout**
`TrainTestSPlit`으로 하는 가장 간단한 방법, 일정 비율에 따라 Train, Validation set으로 나뉜다.

2. **KFold**
dataset을 k개의 fold를 나눠 각 fold별로 반복해서 모델을 학습함
```
kf = KFold(n_splits = 5, shuffle = True)
for train_idx, test_idx in kf.split(X):
    X_train,X_test,y_train,y_test = X[train_idx], X[test_idx], y[train_idx], y[test_idx]
```

3. **Straited KFold**
KFold는 무작위로 fold가 정해지는 것에 비해, Straited KFold는 target class의 비율에 맞춰 분할되어 조금 더 robust한 결과값을 뽑을 수 있도록 해줌
```
kf = StraitedKFold(n_splits = 5, shuffle = True)
for train_idx, test_idx in kf.split(X,y):
    X_train,X_test,y_train,y_test = X[train_idx], X[test_idx], y[train_idx], y[test_idx]
```

### Modeling