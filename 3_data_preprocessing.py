"""
범주형 데이터의 가공 용이성을 위한 데이터 전처리 과정
"""

from typing import Tuple
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
