import pandas as pd
import numpy as np
import os

from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge


def predict_eval(model, train, train_features, name):
    y_train_pred = model.predict(train[train_features])
    rmse = mean_squared_error(train.log_trip_duration, y_train_pred, squared=False)
    r2 = r2_score(train.log_trip_duration, y_train_pred)
    print(f"{name} RMSE = {rmse:.4f} - R2 = {r2:.4f}")


def approach1(train, test): # direct
    numeric_features = ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']
    categorical_features = ['dayofweek', 'month', 'hour', 'dayofyear', 'passenger_count']
    train_features = categorical_features + numeric_features

    column_transformer = ColumnTransformer([
        ('ohe', OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ('scaling', StandardScaler(), numeric_features)
        ]
        , remainder = 'passthrough'
    )

    pipeline = Pipeline(steps=[
        ('ohe', column_transformer),
        ('regression', Ridge())
    ])

    model = pipeline.fit(train[train_features], train.log_trip_duration)
    predict_eval(model, train, train_features, "train")
    predict_eval(model, test, train_features, "test")


def prepare_data(train):
    train.drop(columns=['id'], inplace=True)

    train['pickup_datetime'] = pd.to_datetime(train['pickup_datetime'])
    train['dayofweek'] = train.pickup_datetime.dt.dayofweek
    train['month'] = train.pickup_datetime.dt.month
    train['hour'] = train.pickup_datetime.dt.hour
    train['dayofyear'] = train.pickup_datetime.dt.dayofyear

    train['log_trip_duration'] = np.log1p(train.trip_duration)


if __name__ == '__main__':
    root_dir = 'project-nyc-taxi-trip-duration'
    train = pd.read_csv(os.path.join(root_dir, 'split_sample/train.csv'))
    test = pd.read_csv(os.path.join(root_dir, 'split_sample/val.csv'))

    prepare_data(train)
    prepare_data(test)

    approach1(train, test)
