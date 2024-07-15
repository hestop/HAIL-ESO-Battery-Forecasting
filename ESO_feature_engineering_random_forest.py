import pandas as pd
import numpy as np
import torch
from torch import nn, Tensor, optim, cuda
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns


# 데이터 로드
file_path = "./data/train_data.csv"
train_set = pd.read_csv(file_path, skip_blank_lines=True)

# 마지막 몇 행 확인
print(train_set.tail())

if torch.backends.mps.is_available():
    device = 'mps'
    mps_device = torch.device(device)
    torch.cuda.manual_seed_all(777)
    print(device)
else:
    print("MPS device not found.")

torch.manual_seed(777)

# 불필요한 열 제거
train_set_cleaned = train_set.loc[:, train_set.nunique() != 1]

# 결과 확인
print(train_set_cleaned.shape)

# Function to fill missing values with the average of the row above and below
def fill_missing_values(data):
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
    columns_to_process = [col for col in numeric_columns if col not in ['id']]

    for i in range(len(data)):
        if data.at[i, 'id'] % 2 == 0:
            for col in columns_to_process:
                if pd.isna(data.at[i, col]):
                    if i == 0:
                        data.at[i, col] = data.at[i + 1, col]
                    elif i == len(data) - 1:
                        data.at[i, col] = data.at[i - 1, col]
                    else:
                        data.at[i, col] = (data.at[i - 1, col] + data.at[i + 1, col]) / 2
    return data

# Apply the function to fill missing values
filled_trained_set_subset = fill_missing_values(train_set_cleaned)

# 시계열 인덱스 설정
filled_trained_set_subset['UTC_Settlement_DateTime'] = pd.to_datetime(filled_trained_set_subset['UTC_Settlement_DateTime'])
filled_trained_set_subset.set_index('UTC_Settlement_DateTime', inplace=True)

# 시간 관련 피처 추가
filled_trained_set_subset['year'] = filled_trained_set_subset.index.year
filled_trained_set_subset['month'] = filled_trained_set_subset.index.month
filled_trained_set_subset['day'] = filled_trained_set_subset.index.day
filled_trained_set_subset['hour'] = filled_trained_set_subset.index.hour

# 데이터 프레임 열 이름 확인
print(filled_trained_set_subset.columns)

# 데이터 셋 읽기
data = filled_trained_set_subset

# 실제 타깃 변수를 설정 (여기서는 'output'이라고 가정)
target_variable = 'battery_output'  # 실제 타깃 변수 이름으로 변경
x_data = data.drop(columns=[target_variable, 'id'])
y_data = data[target_variable]

# 숫자형 열만 선택 및 정규화
numeric_columns = x_data.select_dtypes(include=[np.number]).columns
scaler = MinMaxScaler()
x_data[numeric_columns] = scaler.fit_transform(x_data[numeric_columns])
x_data.fillna(0, inplace=True)
print(x_data.shape)

# TimeSeriesSplit 사용
tscv = TimeSeriesSplit(n_splits=17000)
train_index, test_index = next(tscv.split(x_data))

x_train, x_test = x_data.iloc[train_index], x_data.iloc[test_index]
y_train, y_test = y_data.iloc[train_index], y_data.iloc[test_index]

# Feature Engineering: Feature Importance 시각화
rf_model_temp = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model_temp.fit(x_train, y_train.values.ravel())
feature_importances = rf_model_temp.feature_importances_
features = x_train.columns

plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances, y=features)
plt.title('Feature Importances')
plt.show()

# 중요한 특징 선택
important_features = features[np.argsort(feature_importances)[-30:]]  # 상위 30개의 중요한 특징 선택
x_train_important = x_train[important_features]
x_test_important = x_test[important_features]

# Random Forest 모델 학습 (n_estimators를 줄여 학습 시간을 단축)
rf_model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
rf_model.fit(x_train_important, y_train.values.ravel())

# 예측
y_train_pred = rf_model.predict(x_train_important)
y_test_pred = rf_model.predict(x_test_important)

# 성능 평가
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)

print(f'Train MSE: {train_mse:.4f}')
print(f'Test MSE: {test_mse:.4f}')

# 예측 결과 시각화
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_test_pred, alpha=0.7)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.show()
