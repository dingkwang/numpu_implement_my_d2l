import numpy as np


class LinearRegression:

    def __init__(self, learning_rate=0.01, n_iteration=1000, batch_size=None) -> None:
        self.learning_rate = learning_rate
        self.n_iteration = n_iteration
        self.weights = None
        self.bias = None
        self.batch_size = batch_size

    def fit(self, X, y):
        n_sample, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iteration):
            print(f"Iteration {_}", end="\r")
            if self.batch_size:
                indexed = np.random.randint(0, n_sample, self.batch_size)
                x_batch = X[indexed]
                y_batch = y[indexed]
            else:
                x_batch = X
                y_batch = y

            model = x_batch @ self.weights + self.bias
            prediction = model

            dw = 1 / n_sample * x_batch.T @ (prediction - y_batch)
            db = 1 / n_sample * np.sum(prediction - y_batch)

            self.weights -= self.weights * dw
            self.bias -= self.bias * db

    def predict(self, X):
        return X @ self.weights + self.bias


from sklearn import datasets
from sklearn.model_selection import train_test_split

X, y = datasets.load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

log_reg = LinearRegression()
log_reg.fit(X_train, y_train)
preds = log_reg.predict(X_test)

print(preds)
print(y_test)
