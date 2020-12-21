# Copyright 2020 (c) Cognizant Digital Business, Evolutionary AI. All rights reserved. Issued under the Apache 2.0 License.

import argparse
import os
import yaml
import pickle

import numpy as np
import pandas as pd

from algorithms import load_model
from utils.data import Data

######################
CONFIG_PATH = "./experiments/lightgbm0.yaml"
######################

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(ROOT_DIR, 'data', "OxCGRT_latest.csv")
ID_COLS = ['CountryName',
           'RegionName',
           'GeoID',
           'Date']
CASES_COL = ['NewCases']
NPI_COLS = ['C1_School closing',
            'C2_Workplace closing',
            'C3_Cancel public events',
            'C4_Restrictions on gatherings',
            'C5_Close public transport',
            'C6_Stay at home requirements',
            'C7_Restrictions on internal movement',
            'C8_International travel controls',
            'H1_Public information campaigns',
            'H2_Testing policy',
            'H3_Contact tracing',
            'H6_Facial Coverings']
NB_LOOKBACK_DAYS = 30
# For testing, restrict training data to that before a hypothetical predictor submission date
HYPOTHETICAL_SUBMISSION_DATE = np.datetime64("2020-07-31")


def predict(start_date: str,
            end_date: str,
            path_to_ips_file: str,
            output_file_path) -> None:
    """
    Generates and saves a file with daily new cases predictions for the given countries, regions and intervention
    plans, between start_date and end_date, included.
    :param start_date: day from which to start making predictions, as a string, format YYYY-MM-DDD
    :param end_date: day on which to stop making predictions, as a string, format YYYY-MM-DDD
    :param path_to_ips_file: path to a csv file containing the intervention plans between inception date (Jan 1 2020)
     and end_date, for the countries and regions for which a prediction is needed
    :param output_file_path: path to file to save the predictions to
    :return: Nothing. Saves the generated predictions to an output_file_path CSV file
    with columns "CountryName,RegionName,Date,PredictedDailyNewCases"
    """
    # !!! YOUR CODE HERE !!!
    preds_df = predict_df(start_date, end_date,
                          path_to_ips_file, verbose=False)
    # Create the output path
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    # Save to a csv file
    preds_df.to_csv(output_file_path, index=False)
    print(f"Saved predictions to {output_file_path}")


def predict_df(start_date_str: str, end_date_str: str, path_to_ips_file: str, verbose=False):
    """
    Generates a file with daily new cases predictions for the given countries, regions and npis, between
    start_date and end_date, included.
    :param start_date_str: day from which to start making predictions, as a string, format YYYY-MM-DDD
    :param end_date_str: day on which to stop making predictions, as a string, format YYYY-MM-DDD
    :param path_to_ips_file: path to a csv file containing the intervention plans between inception_date and end_date
    :param verbose: True to print debug logs
    :return: a Pandas DataFrame containing the predictions
    """
    start_date = pd.to_datetime(start_date_str, format='%Y-%m-%d')
    end_date = pd.to_datetime(end_date_str, format='%Y-%m-%d')
    n_fore = (end_date - start_date).days

    # Load historical intervention plans, since inception
    ips = pd.read_csv(path_to_ips_file,
                      parse_dates=['Date'],
                      encoding="ISO-8859-1",
                      dtype={"RegionName": str},
                      error_bad_lines=True)

    with open(CONFIG_PATH, 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)

    data = Data(config, ips=ips)

    # Load model
    model = load_model(config)

    iterator = data.build_test_iter(start_date, n_fore)
    sample = iterator()
    for i in range(n_fore):
        fore = model.predict(sample)
        sample = iterator(fore)
        # Get ith sample and update df
    forecast = iterator(test=True)
    return forecast


# !!! PLEASE DO NOT EDIT. THIS IS THE OFFICIAL COMPETITION API !!!
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--start_date",
                        dest="start_date",
                        type=str,
                        required=True,
                        help="Start date from which to predict, included, as YYYY-MM-DD. For example 2020-08-01")
    parser.add_argument("-e", "--end_date",
                        dest="end_date",
                        type=str,
                        required=True,
                        help="End date for the last prediction, included, as YYYY-MM-DD. For example 2020-08-31")
    parser.add_argument("-ip", "--interventions_plan",
                        dest="ip_file",
                        type=str,
                        required=True,
                        help="The path to an intervention plan .csv file")
    parser.add_argument("-o", "--output_file",
                        dest="output_file",
                        type=str,
                        required=True,
                        help="The path to the CSV file where predictions should be written")
    args = parser.parse_args()
    print(
        f"Generating predictions from {args.start_date} to {args.end_date}...")
    predict(args.start_date, args.end_date, args.ip_file, args.output_file)
    print("Done!")
