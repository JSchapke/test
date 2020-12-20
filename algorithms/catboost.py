import pickle
import numpy as np
from catboost import Pool, CatBoostRegressor


class CatBoost:
    def __init__(self, config):
        self.params = config["model_params"]

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)

    def load(self, path):
        with open(path, 'rb') as f:
            self.model = pickle.load(f)

    def forecast(self, data, n_forecasts):
        test_pool = Pool(test_data)
        forecast = self.model.predict(data)[0]
        forecast = max(forecast, 0)

    def predict(self, X):
        test_pool = Pool(X, cat_features=[])
        preds = self.model.predict(test_pool)
        return np.maximum(preds, 0)

    def train(self, X_train, y_train, X_eval=None, y_eval=None):
        train_pool = Pool(X_train, y_train, cat_features=[])

        eval_set = None
        if X_eval is not None:
            eval_set = Pool(X_eval, y_eval, cat_features=[])

        self.model = CatBoostRegressor(**self.params)
        # Fit model
        self.model.fit(train_pool, eval_set=eval_set)
