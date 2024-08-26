import pandas as pd

from sklearn.preprocessing import MinMaxScaler, StandardScaler
# from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# load_dataset():
"""
TAKES ==> path to the datasets, train dataset name, and validation dataset name

RETURNS ==> train dataset (X), val dataset (y)
"""


def load_dataset(root='', path='datasets/split/', train_dataset='train.csv', val_dataset='val.csv'):
    root = '/home/ma7moud-5aled/PycharmProjects/Machine-Learning-Projects/nyc_taxi_trip_duration/'
    X = pd.read_csv(root + path + train_dataset)
    y = pd.read_csv(root + path + val_dataset)

    return X, y


# split_data():
"""
if apply manually splitting: (X) is train dataset, 
    (y) is val dataset

"""


def split_data(X, y, toNumpy=False):
    # split the data manually, and this is the suitable way for our data

    X_train = X.iloc[:, :-1]
    y_train = X.iloc[:, -1]

    X_test = y.iloc[:, :-1]
    y_test = y.iloc[:, -1]
    print(f'here: type of dataset {type(X_train)}')

    if toNumpy:
        X_train = X_train.to_numpy()
        y_train = y_train.to_numpy().reshape(-1, 1)

        X_test = X_test.to_numpy()
        y_test = y_test.to_numpy().reshape(-1, 1)
    # train input data (X_train), train target value (y_train)
    # val input data (X_val), val target value (y_val)
    return X_train, y_train, X_test, y_test

def cols_to_drop(X, y, cols, is_series=False):
    if is_series:
        X = X.drop([cols], axis=1)
        y = y.drop([cols], axis=1)
    else:
        X = X.drop(cols, axis=1)
        y = y.drop(cols, axis=1)

    return X, y

# get_cat_col():
"""
TAKES ==> Pandas DataFrame of train Input Data
RETURNS ==> categorical columns in the DataFrame
"""
def cat_cols_to_encode(X, y):
    # ATTENTION ===> ----- i choose just 'store_and_fwd_flag' column ----- <===
    # df_cat = df.select_dtypes(include=['object'])
    X_cat = X['store_and_fwd_flag']
    y_cat = y['store_and_fwd_flag']
    X_cat_names: str = ''

    is_series = False
    if X_cat.size == X_cat.shape[0]:
        X_cat_names = X_cat.name
        is_series = True
    else:
        X_cat_names = X_cat.columns

    return X_cat, y_cat, X_cat_names, is_series


# get_numerical_col():
"""
TAKES ==> Pandas DataFrame of train Input Data
RETURNS ==> numerical columns in the DataFrame
"""


def get_numeric_col(df):
    df_num = df.select_dtypes(include=['float32', 'float64', 'int32', 'int64'])

    return df_num


def obj_to_datetime(X, y):
    X['pickup_datetime'] = pd.to_datetime(X['pickup_datetime'], format='%Y-%m-%d %H:%M:%S')
    y['pickup_datetime'] = pd.to_datetime(y['pickup_datetime']).values

    X_date = X['pickup_datetime'].values
    y_date = y['pickup_datetime'].values
    # dfd = y['pickup_datetime'].values

    X_date = X_date.reshape(-1, 1)
    y_date = y_date.reshape(-1, 1)

    # print(type(X.iloc[0, 1]))
    # print(type(y.iloc[0, 1]))
    # print(type(dfd[0]))
    return X_date, y_date


# encode_data():
"""
TAKES ==> train Input Data (X_train), val Input Data (X_val)
            categorical columns (columns), chosen encoder (encoder) default(OneHotEncoder())
RETURNS ==> encoded data with the applied encoder (encoded) -array or list (idk)- 
            (Transformed array) in the documentation 
"""


def encode_data(features, is_series=False, encoder=OneHotEncoder()):
    # transformer = make_column_transformer((OneHotEncoder(), columns),
    #                                       remainder='passthrough')
    if is_series:
        features = features.reshape(-1, 1)

    OHE = OneHotEncoder(sparse_output=False)

    encoded = OHE.fit_transform(features)
    print(f'fit returns: {type(encoded)}')

    return encoded


def encode_(X, y, cols):
    # OHE = OneHotEncoder(sparse_output=False)
    OHE = OneHotEncoder(handle_unknown='ignore')
    # if not isinstance(cols, list):
    #     cols = [cols]

    # print(f'1- uu- {X.shape}')
    # print(f'1- yy- {y.shape}')

    enc_X = X[cols].to_numpy().reshape(-1, 1)
    enc_y = y[cols].to_numpy().reshape(-1, 1)

    # print(f'2- uu- {enc_X.shape}')
    # print(f'2- yy- {enc_y.shape}')
    enc_X = OHE.fit_transform(enc_X).toarray()
    enc_y = OHE.transform(enc_y).toarray()

    cols_names = OHE.get_feature_names_out([cols])
    print(f'feat uu- {cols_names}')
    # print(f'feat yy- {enc_y.get_feature_names_out([cols])}')

    # print(f'3- uu- {X.shape}')
    # print(f'3- yy- {y.shape}')
    #
    # print(f'3- enc_uu- {enc_X.shape}')
    # print(f'3- enc_yy- {enc_y.shape}')
    #
    # print(f'3- enc_uu- {type(enc_X)}')
    # print(f'3- enc_yy- {type(enc_y)}')

    df_x = pd.DataFrame(enc_X,
                        columns=cols_names)
    df_y = pd.DataFrame(enc_y,
                        columns=cols_names)

    X = pd.concat([X.drop(cols, axis=1), df_x], axis=1)
    y = pd.concat([y.drop(cols, axis=1), df_y], axis=1)

    print(f'encode_ done')
    return X, y


def extract_datetime(X_train, X_test, date):
    # print(f'is it datetime: {type(X_train[date])}')
    X_train['pickup_datetime'] = pd.to_datetime(X_train['pickup_datetime'])
    X_test['pickup_datetime'] = pd.to_datetime(X_test['pickup_datetime'])
    # print(f'is it datetime: {type(X_train[date])}')
    #
    # X_train['pickup_datetime'] = pd.to_datetime(X_train['pickup_datetime'], format='%Y-%m-%d %H:%M:%S')
    # print(f'is it datetime: {type(X_train[date])}')

    # X = pd.to_datetime(X_train['pickup_datetime']).to_datetime64()
    # print(f'X is it datetime: {type(X)}')

    print(X_train.info())

    X_train[date+':year'] = X_train[date].dt.year
    X_train[date+':month'] = X_train[date].dt.month
    X_train[date+':weekday'] = X_train[date].dt.weekday
    X_train[date+':day'] = X_train[date].dt.day
    X_train[date+':hour'] = X_train[date].dt.hour
    X_train[date+':minute'] = X_train[date].dt.minute
    X_train[date+':second'] = X_train[date].dt.second

    X_test[date+':year'] = X_test[date].dt.year
    X_test[date+':month'] = X_test[date].dt.month
    X_test[date+':weekday'] = X_test[date].dt.weekday
    X_test[date+':day'] = X_test[date].dt.day
    X_test[date+':hour'] = X_test[date].dt.hour
    X_test[date+':minute'] = X_test[date].dt.minute
    X_test[date+':second'] = X_test[date].dt.second

    X_train = X_train.drop(date, axis=1)
    X_test = X_test.drop(date, axis=1)
    # print(f'in for loop: {type(d)}')

    return X_train, X_test

# def concat_encoded_cols(df, features_names, transformed, categories):
#     df = df.loc[:, ~df.columns.isin([features_names])]
#
#     n_cat_feats = 0
#     for cat in categories:
#         for cat_transformed_feature_name, idx in zip(cat, range(n_cat_feats, n_cat_feats+len(cat))):
#             df[cat_transformed_feature_name] = transformed[:, idx]
#             # print(f'categorical feature of categories: {cat_transformed_feature_name}')
#             # print(f'value of x: {idx}')
#
#         n_cat_feats = n_cat_feats + len(cat)
#
#     return df


def datasets_to_numpy_array(X_train, X_val, y_train, y_val):
    X_train = X_train.to_numpy()
    X_val = X_val.to_numpy()
    y_train = y_train.to_numpy()
    y_val = y_val.to_numpy()

    return X_train, X_val, y_train, y_val


def cats_to_numpy_array(cat_cols):
    cat_cols = cat_cols.to_numpy()

    return cat_cols


# scale_data():
"""
TAKES ==> train Input Data (X_train), val Input Data (X_val), chosen scaler (scaler) default(MinMaxScaler())
RETURNS ==> scaled train Input Data with the applied scaler (X_train)
            scaled val Input Data with the applied scaler (X_val)
"""


def scale_data(X_train, X_test, processor=MinMaxScaler()):
    """
    MISTAKES I MADE IN THIS CODE
    processor = MinMaxScaler()
    1- i used processor to fit_transform(t_train), we dont transform or fit_transform the target data
    2- i used processor to fit_transform(X_val), must be transform(X_val)
        REASON:
        i fitted the transformer and transformed the data on the train data, and i want to see if the scaler works well on
        the train data, so it must directly transform the val data on the fitted train data
    3- i used processor to fit_transform(t_val), we dont transform or fit_transform the target data
    """

    X_train = processor.fit_transform(X_train)

    # in the X_val we used transform not fit_transform
    X_test = processor.transform(X_test)

    return X_train, X_test
