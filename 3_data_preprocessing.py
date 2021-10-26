"""
범주형 데이터의 가공 용이성을 위한 데이터 전처리 과정
"""

from typing import Tuple
from numpy.core.fromnumeric import std
import pandas as pd

df = pd.DataFrame(
    [["green", "M", 10.1, "class1"], ["red", "L", 13.5, "class2"], ["blue", "XL", 15.3, "class1"]]
)

df.columns = ["color", "size", "price", "classlabel"]


# 순서가 있는 특성의 mapping(XL > L > M) + mapping 원상복구
size_mapping = {"XL": 3, "L": 2, "M": 1}
df["size"] = df["size"].map(size_mapping)
print(df)

inv_size_mapping = {v: k for k, v in size_mapping.items()}
df["size"] = df["size"].map(inv_size_mapping)
print(df)

# 클래스 레이블 encoding

import numpy as np

class_mapping = {label: idx for idx, label in enumerate(np.unique(df["classlabel"]))}
df["classlabel"] = df["classlabel"].map(class_mapping)
print(df)

inv_class_mapping = {v: k for k, v in class_mapping.items()}
df["classlabel"] = df["classlabel"].map(inv_class_mapping)
print(df)

# 순서가 없는 특성에 One-hot encoding 적용

from sklearn.preprocessing import LabelEncoder

# 의도하지 않게 순서가 없는 특성에 순서를 만들게 됨

X = df[["color", "size", "price"]].values
color_le = LabelEncoder()
X[:, 0] = color_le.fit_transform(X[:, 0])
print(X)

# One-hot encoding

from sklearn.preprocessing import OneHotEncoder

X = df[["color", "size", "price"]].values
color_ohe = OneHotEncoder()
color_ohe.fit_transform(X[:, 0].reshape(-1, 1)).toarray()

from sklearn.compose import ColumnTransformer

X = df[["color", "size", "price"]].values
c_transf = ColumnTransformer([("onehot", OneHotEncoder(), [0]), ("nothing", "passthrough", [1, 2])])
c_transf.fit_transform(X)

pd.get_dummies(df[["price", "color", "size"]])

# 다중 공산성을 해소하기 위해 정보량에 영향이 없는 열 하나를 삭제
color_ohe = OneHotEncoder(categories="auto", drop="first")
c_transf = ColumnTransformer([("onehot", color_ohe, [0]), ("nothing", "passthrough", [1, 2])])
c_transf.fit_transform(X)

pd.get_dummies(df[["price", "color", "size"]], drop_first=True)

# Dataset - wine

df_wine = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data", header=None
)
df_wine.columns = [
    "Class label",
    "Alcohol",
    "Malic acid",
    "Ash",
    "Alcalinity of ash",
    "Magnesium",
    "total phenols",
    "Flavanoids",
    "Nonflavanoid phenols",
    "Proanthocyanins",
    "Color intensity",
    "Hue",
    "OD280/OD315 of diluted wines",
    "Proline",
]

print("클래스 테이블", np.unique(df_wine["Class label"]))
df_wine.head()

from sklearn.model_selection import train_test_split

X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

"""
특성 간 스케일 조정

정규화 - 최소-최대 스케일 조정
:
    (Xi - Xmin)/(Xmax - Xmin)

표준화
:
    (Xi - 샘플 평균)/표준 편차
"""

# 최소-최대 스케일 조정

from sklearn.preprocessing import MinMaxScaler

mms = MinMaxScaler()
X_train_norm = mms.fit_transform(X_train)
X_test_norm = mms.transform(X_test)

# 표준화

from sklearn.preprocessing import StandardScaler

stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)

# L1 규제

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(solver="liblinear", penalty="l1", C=1.0, random_state=1)
lr.fit(X_train_std, y_train)
print("훈련 정확도", lr.score(X_train_std, y_train))
print("테스트 정확도", lr.score(X_test_std, y_test))

# 규제 강도에 따른 특성의 가중치 변화 그래프

import matplotlib.pyplot as plt

fig = plt.figure()
ax = plt.subplot(111)

colors = {
    "blue",
    "green",
    "red",
    "cyan",
    "magenta",
    "yellow",
    "black",
    "pink",
    "lightgreen",
    "lightblue",
    "gray",
    "indigo",
    "orange",
}

weights, params = [], []

for c in np.arange(-4.0, 6.0):
    lr = LogisticRegression(solver="liblinear", penalty="l1", C=10.0 ** c, random_state=1)
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[1])
    params.append(10 ** c)

weights = np.array(weights)

for column, color in zip(range(weights.shape[1]), colors):
    plt.plot(params, weights[:, column], label=df_wine.columns[column + 1], color=color)
plt.axhline(0, color="black", linestyle="--", linewidth=3)
plt.xlim([10 ** (-5), 10 ** 5])
plt.ylabel("weight coefficient")
plt.xlabel("C")
plt.xscale("log")
plt.legend(loc="upper left")
ax.legend(loc="upper center", bbox_to_anchor=(1.38, 1.03), ncol=1, fancybox=True)

