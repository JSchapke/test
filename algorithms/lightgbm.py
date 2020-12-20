import pickle
import numpy as np
import lightgbm as lgb


class LightGBM:
    def __init__(self, config):
        self.params = config["model_params"]

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)

    def load(self, path):
        with open(path, 'rb') as f:
            self.model = pickle.load(f)

    def predict(self, X):
        return self.model.predict(X)

    def train(self, X_train, y_train, X_eval=None, y_eval=None, info=None):
        eval_set = None
        if X_eval is not None:
            eval_set = [(X_eval, y_eval)]

        categorical_feature = feature_name = "auto"
        if info is not None:
            categorical_feature = info.get("Categorical Features", "auto")
            feature_name = info.get("Feature Names", "auto")

        self.model = lgb.LGBMRegressor(**self.params)

        # Fit model
        self.model.fit(X_train, y_train, eval_set=eval_set, eval_metric='MAE',
                       feature_name=feature_name, categorical_feature=categorical_feature)
