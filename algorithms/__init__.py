import yaml
import os

from algorithms.linear import Linear
from algorithms.catboost import CatBoost
from algorithms.lightgbm import LightGBM
from algorithms.ensemble import Ensemble

ALGORITHMS = dict(
    linear=Linear,
    catboost=CatBoost,
    lightgbm=LightGBM,
    ensemble=Ensemble,
)


def load_model(config):
    assert os.path.isfile(config["weights_path"])
    model = ALGORITHMS[config["algo"]](config)
    model.load(config["weights_path"])
    return model


def get_model(config):
    print(ALGORITHMS)
    print(config['algo'])
    return ALGORITHMS[config["algo"]](config)
