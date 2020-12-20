import numpy as np
import pandas as pd
import xarray as xr

ID_COLS = ['CountryName',
           'RegionName',
           'GeoID',
           'Date']

CASES_COL = ['NewCasesRM',
             'NewCasesRMGrowthRM',
             'AccCasesRM',
             ]

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
            'H6_Facial Coverings',
            ]

MAX_LOOK_BACK = 14
WINDOW_NEW_CASES = 14
WINDOW_ACC_CASES = 14
WINDOW_GRO_CASES = 14


class Data:
    def __init__(self, config, df=None):
        self.config = config

        if df is None:
            IPS_PATH = './data/OxCGRT_latest.csv'
            df = pd.read_csv(IPS_PATH,
                             parse_dates=['Date'],
                             encoding="ISO-8859-1",
                             dtype={"RegionName": str},
                             error_bad_lines=True)

        df = process_df(df)
        df = df.set_index(["Date", "GeoID"])
        self.df = df.sort_index()

    def build_train(self):
        """
        Input: ips_df, configuration
        Output: X, y, info - np.array, np.array, dict
        """

        # print("Warning! Dropping datapoints of countries with 0 ConfirmedCases")
        # df = df[df.ConfirmedCases > 0]

        df = self.df.reorder_levels([1, 0])

        # Extract features
        X, info = extract_features(df)
        y = df[self.config["data"].get("y", "NewCasesRM")]

        X = X.reorder_levels([1, 0])
        X = X.sort_index()

        y = y.reorder_levels([1, 0])
        y = y.sort_index()

        mask = ~(y.isna().values | X.isna().any(1).values)
        y = y[mask]
        X = X[mask]

        return X, y, info

    def build_test_iter(self, fore_date, n_fore):
        forecasts = []
        df = self.df.copy()

        def iterator(forecast=None):
            #day = pd.Timedelta(days=1)
            if n_fore > len(forecasts):

                if forecast is not None:
                    # Parse forecast
                    forecasts.append(forecast)
                    date = fore_date + pd.Timedelta(days=len(forecasts))
                    print(df.loc[date])
                    self.update_df(df, forecast, date)

                date = fore_date + pd.Timedelta(days=len(forecasts))
                X = extract_last_features(
                    df.loc[date-pd.Timedelta(days=MAX_LOOK_BACK):date].copy())
                return X

            fore = np.array(forecasts).T
            regions = df.loc[fore_date].index
            dates = pd.date_range(fore_date, periods=n_fore, freq='1d')
            fore = pd.DataFrame(fore, index=regions, columns=dates)
            return fore

        return iterator

    def update_df(self, df, forecast, date):
        day = pd.Timedelta(days=1)

        y = self.config["data"].get("y", "NewCasesRM")
        if y == "NewCasesRM":
            df.loc[date, "NewCasesRM"] = forecast
            prev_new_cases = df.NewCasesRM.loc[date-day].values
            df.loc[date, "AccCases"] = forecast - prev_new_cases
            df.loc[date, "AccCasesRM"] = df.AccCases.loc[
                date - day*WINDOW_ACC_CASES:date].mean()
            df.loc[date, "NewCasesRMGrowth"] = forecast / prev_new_cases
            df.loc[date, "NewCasesRMGrowthRM"] = df.NewCasesRMGrowth.loc[
                date - day*WINDOW_GRO_CASES:date].mean()

        elif y == "AccCasesRM":
            acc = forecast - \
                df.AccCasesRM[date - day].values + \
                df.AccCases[date-(WINDOW_ACC_CASES-1)*day] / WINDOW_ACC_CASES
            acc = acc.values * WINDOW_ACC_CASES
            prev_vel = df.NewCasesRM.loc[date-day].values
            vel = df.NewCasesRM.loc[date-day].values + acc

            df.loc[date, "NewCasesRM"] = vel
            df.loc[date, "AccCases"] = acc
            df.loc[date, "AccCasesRM"] = forecast

            df.loc[date, "NewCasesRMGrowth"] = vel / prev_vel
            df.loc[date, "NewCasesRMGrowthRM"] = df.NewCasesRMGrowth.loc[
                date - day*WINDOW_GRO_CASES:date].mean()

#            print('= '*20)
#            print('vel:,', vel[:5])
#            print(df.loc[date])
#            print('- '*20)
#            ask
            if np.any(np.isnan(prev_vel)) or np.any(np.isnan(vel)):
                ask

        elif y == "NewCasesRMGrowthRM":
            raise NotImplementedError()


def process_df(df):
    """
    Input: ips_df
    Output: out_df
    """
    df = df.copy()

    # Add GeoID column that combines CountryName and RegionName for easier manipulation of data
    df['GeoID'] = df['CountryName'] + '__' + df['RegionName'].astype(str)

    print('Warning! Interpolating missing ConfirmedCases statistics. (This will affect the ground truth labels NewCases).')
    df.update(df.groupby('GeoID').ConfirmedCases.apply(
        lambda group: group.interpolate()).fillna(0))

    # Add new cases column
    df['NewCases'] = float('nan')
    df['NewCasesRM'] = float('nan')
    df['AccCases'] = float('nan')
    df['AccCasesRM'] = float('nan')
    df['NewCasesRMGrowth'] = float('nan')
    df['NewCasesRMGrowthRM'] = float('nan')

    for geo_id, gdf in df.groupby('GeoID'):
        new_cases = gdf.ConfirmedCases.diff().interpolate().fillna(0)

        new_cases_rm = new_cases.rolling(
            window=WINDOW_NEW_CASES, center=False, min_periods=5).mean()

        acc_cases = new_cases_rm.diff(1)
        new_cases_rm_growth = (new_cases_rm / new_cases_rm.shift(1)).fillna(0)
        gmask = df.GeoID == geo_id
        df.loc[gmask, 'NewCases'] = new_cases
        df.loc[gmask, 'NewCasesRM'] = new_cases_rm
        df.loc[gmask, "AccCases"] = acc_cases
        df.loc[gmask, "AccCasesRM"] = acc_cases.rolling(
            WINDOW_ACC_CASES, center=False, min_periods=2).mean()
        df.loc[gmask, 'NewCasesRMGrowth'] = new_cases_rm_growth
        df.loc[gmask, 'NewCasesRMGrowthRM'] = new_cases_rm_growth.rolling(
            7, center=False, min_periods=2).mean()

    # Keep only columns of interest
    # df = df[ID_COLS + CASES_COL + NPI_COLS]

    # Fill any missing case values by interpolation and setting NaNs to 0
    # print('Warning! Interpolating missing new cases statistics')
    # df.update(df.groupby('GeoID').NewCases.apply(
    #    lambda group: group.interpolate()).fillna(0))

    # Fill any missing NPIs by assuming they are the same as previous day
    for npi_col in NPI_COLS:
        df.update(df.groupby('GeoID')[npi_col].ffill().fillna(0))

    return df


def extract_last_features(df):
    day = pd.Timedelta(days=1)

    columns = ['NewCasesRM', 'NewCasesRMGrowthRM', 'AccCasesRM'] + NPI_COLS
    data = df[columns].to_xarray().to_array()

    npi = data[3:, -1].drop('Date')

    # Change of NPIs in the last (4) days
    npi_diff_4 = data[3:, -1] - data[3:, -5]
    npi_diff_4_cols = (np.array(NPI_COLS, dtype=object) + '_Diff4').tolist()

    # Change of NPI sum in the last (2, 4, 8) days
    npi_sum = data[3:, :, :].sum('variable')
    npi_sum_diff_2 = npi_sum[-1] - npi_sum[-3]
    npi_sum_diff_4 = npi_sum[-1] - npi_sum[-5]
    npi_sum_diff_8 = npi_sum[-1] - npi_sum[-9]
    npi_sum = npi_sum[-1]

    new = data[0, -2].drop('Date')
    growth = data[1, -2].drop('Date')
    acc = data[2, -2].drop('Date')
    var_acc = data[2, -2] - data[2, -3]

    X = xr.concat([
        new.assign_coords(variable="NewCasesRM_shift1"),
        growth.assign_coords(variable="NewCasesRMGrowthRM_shift1"),
        acc.assign_coords(variable="AccCasesRM_shift1"),
        var_acc.assign_coords(variable="VarAccCasesRM_shift1"),
        npi_diff_4,
        npi_sum.assign_coords(variable="NPISum"),
        npi_sum_diff_2.assign_coords(variable="NPISum-Diff2"),
        npi_sum_diff_4.assign_coords(variable="NPISum-Diff4"),
        npi_sum_diff_8.assign_coords(variable="NPISum-Diff8"),
        npi,
    ], dim='variable')
    X = X.to_pandas()

    X.columns = ['NewCasesRM_shift1', 'NewCasesRMGrowthRM_shift1',
                 'AccCasesRM_shift1', 'VarAccCasesRM_shift1'] + npi_diff_4_cols + \
        ['NPISum', 'NPISum-Diff2', 'NPISum-Diff4', 'NPISum-Diff8', ] + NPI_COLS
    return X


def extract_features(df):
    info = {"Categorical Features": [],
            "NewCasesRM": df.NewCasesRM.copy(),
            "NewCasesRMGrowth": df.NewCasesRMGrowth.copy(),
            }
    case, case_cols = [], []

    # NPIs
    info["Categorical Features"].extend(NPI_COLS)
    npi = df[NPI_COLS]

    g_npi, g_case = [], []
    for geo_id, gdf in df.groupby('GeoID'):
        # Change of NPIs in the last (4) days
        npi_diff_4 = gdf[NPI_COLS] - gdf[NPI_COLS].shift(4)
        cols = (np.array(NPI_COLS, dtype=object) + '_Diff4').tolist()
        npi_diff_4_dict = dict(zip(cols, npi_diff_4))

        # Change of NPI sum in the last (2, 4, 8) days
        npi_sum = gdf[NPI_COLS].sum(1)
        npi_sum_diff_2 = npi_sum - npi_sum.shift(2)
        npi_sum_diff_4 = npi_sum - npi_sum.shift(4)
        npi_sum_diff_8 = npi_sum - npi_sum.shift(8)

        g_npi.append(pd.concat([
            npi_diff_4,
            npi_sum,
            npi_sum_diff_2,
            npi_sum_diff_4,
            npi_sum_diff_8,
        ], axis=1))

        # NewCasesRMGrowthRM Diff 1
        # diff1 = gdf.NewCasesRMGrowthRM - gdf.NewCasesRMGrowthRM.shift(1)

        g_case.append(pd.concat([
            gdf.NewCasesRM.shift(1),
            gdf.NewCasesRMGrowthRM.shift(1),
            gdf.AccCasesRM.shift(1),
            gdf.AccCasesRM.shift(1).diff(),
        ], axis=1))

    g_case = pd.concat(g_case, axis=0)
    g_case.columns = ['NewCasesRM_shift1', 'NewCasesRMGrowthRM_shift1',
                      'AccCasesRM_shift1', 'VarAccCasesRM_shift1']
    g_npi = pd.concat(g_npi, axis=0)
    g_npi.columns = (np.array(NPI_COLS, dtype=object) + '_Diff4').tolist() + \
        ['NPISum', 'NPISum-Diff2', 'NPISum-Diff4', 'NPISum-Diff8']

    X = pd.concat([g_case, g_npi, npi], axis=1)  # Concat global features
    info["Feature Names"] = X.columns.tolist()
    return X, info


def get_data(config, df=None):
    data = Data(config, df)
    return data.build_train()


if __name__ == '__main__':
    config = dict()
    fore_date = pd.to_datetime('2020-10-15')
    n_fore = 14

    data = Data(config)
    iterable = data.build_test_iter(fore_date, n_fore)

    sample = iterable()
    for i in range(n_fore):
        print(sample)
        sample = iterable(15)
    print(iterable())
    print("done!")
