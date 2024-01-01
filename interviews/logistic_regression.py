import numpy as np


class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iteration=1000) -> None:
        self.learning_rate = learning_rate
        self.n_iteration = n_iteration
        self.weights = None
        self.bias = None
    
    @staticmethod   
    def _sigmod(x):
        
        return np.where(x>0, 1/(1+np.exp(-x)), np.exp(x)/(np.exp(x)+1))

    def fit(self, X, y):
        n_sample, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iteration):
            print(f"iteration: {_}", end="\r", flush=True)
            h = np.dot(X, self.weights) + self.bias
            pred = self._sigmod(h)

            dw = (1/n_sample) * X.T @ (pred - y)
            db = (1/n_sample) * np.sum(pred-y)
            
            self.weights -= self.learning_rate*dw
            self.bias -= self.learning_rate*db

    def predict(self, x):
        linear_part = x @ self.weights + self.bias
        y_pred = self._sigmod(linear_part)
        y_pred_cls = [1 if y_p > 0.5 else 0 for y_p in y_pred]
        return y_pred_cls
    


class LogisticRegressionWithOptimizer:
    def __init__(self):
        self.weights = None
        self.bias = None

    def fit(self, X, y, optimizer, n_iterations=1000):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(n_iterations):
            model = np.dot(X, self.weights) + self.bias
            predictions = self._sigmoid(model)

            dw = (1 / n_samples) * np.dot(X.T, (predictions - y))
            db = (1 / n_samples) * np.sum(predictions - y)

            # 使用优化器更新参数
            self.weights, self.bias = optimizer.update_params([self.weights, np.array([self.bias])], [dw, np.array([db])])

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        return np.array([1 if i > 0.5 else 0 for i in y_predicted])
    
if __name__ == "__main__":
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    X, y = datasets.load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    log_reg = LogisticRegression()
    log_reg.fit(X_train, y_train)
    preds = log_reg.predict(X_test)

    print(preds)
    print(y_test)