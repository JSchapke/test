import pandas as pd
import numpy as np
import argparse
import yaml
import os

from algorithms import load_model
from utils.data import Data

N_FORE = 14
FORE_DATES = [
    "2020-09-01", "2020-09-07", "2020-09-14", "2020-09-21",
    "2020-10-01", "2020-10-07", "2020-10-14", "2020-10-21",
    "2020-11-01", "2020-11-07",  # "2020-11-14",
]


def evaluate(config, fore_dates=FORE_DATES, n_fore=N_FORE):
    # Get data
    data = Data(config)
    model = load_model(config)
    mae_errors = []

    for fore_date in fore_dates:
        fdates = pd.date_range(fore_date, periods=n_fore, freq='1d')
        y = data.df.loc[fdates].NewCasesRM #.unstack(level=0)

        iterator = data.build_test_iter(fore_date, n_fore)
        sample = iterator()
        for i in range(N_FORE):
            fore = model.predict(sample)
            sample = iterator(fore)
            # Get ith sample and update df

        forecast = iterator(test=False)
        print("\nForecast:", forecast)
        print("Label:", y)
        mae_error = (forecast - y).abs().mean()
        mae_errors += [mae_error]

    return np.mean(mae_errors)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="experiments/linear.yaml")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)

#    name = config["name"]
#    log_path = f'./logs/{name}/'
#    results_path = f'./results/evaluate/{name}'
#    os.makedirs(log_path, exist_ok=True)
#    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    fore_dates = pd.to_datetime(FORE_DATES)
    n_fore = N_FORE

    mae_error = evaluate(config, fore_dates, n_fore)

    print('*', '='*25, '*')
    print('MAE error:', mae_error)
    print('*', '='*25, '*')
