import numpy as np

from sklearn_classifier import X_train


class LogisticRegressionGD(object):
    """
    경사 하강법을 사용한 로지스틱 회귀 분류기
    
    매개변수
    eta : float
        학습률 (0.0 ~ 1.0)
    n_iter : int
        훈련 데이터셋 반복 횟수
    random_state : int
        가중치 무작위 초기화를 위한 난수 생성기 시드
        
    속성
    w_ : ld-array
        학습된 가중치
    cost_ : list
        에포크 마다 누적된 로지스틱 비용 함수 값
    """

    def __init__(self, eta=0.05, n_iter=100, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = y - output
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()

            # 제곱오차합이 아니라 로지스틱 비용 계산
            cost = -y.dot(np.log(output)) - ((1 - y).dot(np.log(1 - output)))
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, z):
        # 시그모이드 함수 활성화
        return 1.0 / (1.0 + np.exp(-np.clip(z, -250, 250)))

    def predict(self, X):
        # 다음과 동일
        return np.where(self.net_input(X) >= 0.0, 1, 0)
        # return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)

