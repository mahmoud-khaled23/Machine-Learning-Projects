from nyc_taxi_trip_duration.data_manage import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import OneHotEncoder


def anymos_data():
    df = pd.DataFrame()
    df['age'] = [12, 23, 33, 24, 50]
    df['name'] = ['ali', 'ahmed', 'sameh', 'tarek', 'saber']
    df['salary'] = [12.45, 21.4, 33.76, 24.2, 50.34]

    return df


if __name__ == '__main__':

    df = anymos_data()
    columns_names = df.columns
    print(columns_names)

    data = df.to_numpy()
    print("ana hena:", data)

    cat_col_name = columns_names[1]
    cat_col = data[:, 1]
    cat_col = cat_col.reshape(-1, 1)

    OHE = OneHotEncoder(sparse_output=False)
    encoded_data = OHE.fit_transform(cat_col)
    encoded_columns_names = OHE.get_feature_names_out([cat_col_name])

    print(f'encoded data: {encoded_data}')
    print(f'encoded columns names: {encoded_columns_names}')

    print('Project main')



