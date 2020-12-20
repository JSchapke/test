import argparse
import os
import pickle
import yaml
import copy

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import train_test_split

from algorithms import get_model
from utils.data import get_data


# Helpful function to compute mae
def mae(pred, true):
    return np.mean(np.abs(pred - true))


def hyper_search(config, X_train, y_train, X_test, y_test, show_best=False, reset=False):

    def create_model(trial, config):
        config = copy.deepcopy(config)
        hyperparams = {}

        for name, (typ, mi, ma) in config["hyper_search"].items():
            if typ == 'int':
                hyperparams[name] = trial.suggest_int(name, mi, ma)
            elif typ == 'float':
                hyperparams[name] = trial.suggest_uniform(name, mi, ma)
            else:
                raise ValueError(typ)

        config["model_params"].update(hyperparams)
        model = get_model(config)
        return model

    def objective(trial):
        model = create_model(trial, config)
        # Train model
        model.train(X_train, y_train, X_test, y_test)

        # Evaluate model
        preds = model.predict(X_test)
        error = mae(preds, y_test)
        return error

    path = f"./results/optuna/{config['name']}.pkl"
    if not reset and os.path.isfile(path):
        with open(path, "rb") as f:
            study = pickle.load(f)
    else:
        sampler = TPESampler(seed=666)
        study = optuna.create_study(direction="minimize", sampler=sampler)

    if show_best:
        print("Best params:", study.best_params)
        return study

    study.optimize(objective, n_trials=50)

    with open(path, "wb") as f:
        pickle.dump(study, f)

    # uncomment to use optuna
    # final params is in study.best_params
    study.optimize(objective, n_trials=50)
    params = study.best_params
    print("Best params:", params)


def main(config, args):

    # Load historical intervention plans, since inception
    path_to_ips_file = config["data"]["ips_file"]
    df = pd.read_csv(path_to_ips_file,
                     parse_dates=['Date'],
                     encoding="ISO-8859-1",
                     dtype={"RegionName": str},
                     error_bad_lines=True)

    if args.eval or args.no_train:
        # For testing, restrict training data to that before a hypothetical predictor submission date
        HYPOTHETICAL_SUBMISSION_DATE = np.datetime64("2020-09-01")
        eval_df = df.copy()
        df = df[df.Date <= HYPOTHETICAL_SUBMISSION_DATE]

    X_samples, y_samples, _ = get_data(config, df)

    model = get_model(config)

    if args.no_train:
        pass

    elif args.hyper_search:
        X_train, X_test, y_train, y_test = train_test_split(X_samples,
                                                            y_samples,
                                                            test_size=0.2,
                                                            random_state=302)
        hyper_search(config, X_train, y_train, X_test, y_test,
                     reset=args.reset, show_best=args.show_best)

    else:
        if args.val:
            # Split data into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(X_samples,
                                                                y_samples,
                                                                test_size=0.2,
                                                                random_state=301)
            # Train model
            model.train(X_train, y_train, X_test, y_test)

            # Evaluate model
            test_preds = model.predict(X_test)
            print('Test MAE:', mae(test_preds, y_test))

        else:
            # Train model
            model.train(X_samples, y_samples)

        # Evaluate model
        preds = model.predict(X_samples)
        print('Samples MAE:', mae(preds, y_samples))

        model.save(config["weights_path"])

    if args.no_train or args.eval:
        X_eval, y_eval, info = get_data(config, eval_df)
        X_eval = X_eval[HYPOTHETICAL_SUBMISSION_DATE:]
        y_eval = y_eval[HYPOTHETICAL_SUBMISSION_DATE:]

        model.load(config["weights_path"])
        eval_preds = model.predict(X_eval)
        print('Eval MAE:', mae(eval_preds, y_eval))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config",
                        required=True,
                        help="Configuration file")
    parser.add_argument("-hs", "--hyper-search",
                        action="store_true",
                        help="Do hyperparameter search")
    parser.add_argument("-e", "--eval",
                        action="store_true",
                        help="Perform evaluation on model (hence, reduce train data size)")
    parser.add_argument("-v", "--val",
                        action="store_true",
                        help="Use validation data when training (hence, reduce train data size)")
    parser.add_argument("-nt", "--no-train",
                        action="store_true",
                        help="Perform only evaluation on model")
    parser.add_argument("--reset",
                        action="store_true",
                        help="Reset hyperparameter search")
    parser.add_argument("--show-best",
                        action="store_true",
                        help="Show hyper search results")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)

    main(config, args)
    print("Done!")
