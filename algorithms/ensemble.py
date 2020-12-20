import pickle
import numpy as np

import algorithms


class Ensemble:
    def __init__(self, config):
        self.params = config["model_params"]

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.models, f)

    def load(self, path):
        with open(path, 'rb') as f:
            self.models = pickle.load(f)

    def predict(self, X):
        preds = []
        for name, model in self.models.items():
            preds.append(model.predict(X))
        return np.mean(preds, axis=0)

    def train(self, X_train, y_train, X_eval=None, y_eval=None):
        eval_set = None
        if X_eval is not None:
            eval_set = [(X_eval, y_eval)]

        self.models = {}
        for name, config in self.params.items():
            print(f"\n Training: {name}")
            self.models[name] = algorithms.get_model(config)
            self.models[name].train(X_train, y_train, X_eval, y_eval)

