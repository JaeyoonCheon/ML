import numpy as np


class Perceptron(object):
    """
    매개변수 
    eta : float
        학습률 (0.0 ~ 1.0)
    n_iter : int
        훈련 데이터셋 반복 횟수
    random_state : int
        가중치 무작위 초기화를 위한 난수 시드

    속성(초기화에서 생성하지 않은 값, 끝에 _를 붙여 표시)
    w_ : ld_array
        학습된 가중치
    errors_: list
        에포크마다 누적된 분류 오류
    """

    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    """
    fit : 훈련 데이터 학습

    매개변수
    X : {array-like}, shape = {n_samples, n_features}
        n_sample개 샘플과 n_features개의 특성으로 이루어진 훈련 데이터.
        *현재 사용할 붓꽃 데이터에서는 2차원 array set
    y : array-like, shape = {n_samples}
        타깃 값

    반환
        self : object
    """

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))

                # 퍼셉트론 규칙 1. 절편이 아닌 경우(1~n개)
                # 가중치 변화량 w[1:] = 학습률 eta*(참 레이블-예측 레이블)*훈련샘플 x
                self.w_[1:] += update * xi

                # 퍼셉트론 규칙 2. 절편인 경우(0번 index)
                # 가중치 변화량 w[0] = 학습률 eta*(참 레이블-예측 레이블)
                self.w_[0] += update

                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    """
    net_input : 행렬 곱. x값과 가중치 w의 곱을 계산
    """

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    """
    predict : 단위 계단 함수를 사용하여 클래스 레이블(1/-1) 반환
    """

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)
