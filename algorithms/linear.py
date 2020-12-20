import pickle
from sklearn.linear_model import Lasso


class Linear:
    def __init__(self, config):
        pass

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)

    def load(self, path):
        with open(path, 'rb') as f:
            self.model = pickle.load(f)

    def forecast(self, data, n_forecasts):
        forecast = self.model.predict(data)[0]
        forecast = max(forecast, 0)

    def predict(self, X):
        return self.model.predict(X)

    def train(self, X_train, y_train, X_eval=None, y_eval=None):
        self.model = Lasso(alpha=0.1,
                           precompute=True,
                           max_iter=10000,
                           positive=True,
                           selection='random')

        # Fit model
        self.model.fit(X_train, y_train)
