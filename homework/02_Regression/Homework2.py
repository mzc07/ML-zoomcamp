# Archive: Homework2.py 
# Author: @mzc07
# Description: Linear Regression practice
import pandas as pd
import numpy as np

# load dataset
df_car = pd.read_csv('data/car_fuel_efficiency_2026.csv')

cols = ['engine_displacement', 'horsepower', 'vehicle_weight', 'model_year', 'fuel_efficiency_mpg']
df_car = df_car[cols]
features = cols[:-1]
target = 'fuel_efficiency_mpg'


def split_data(df_car, seed):
    n = len(df_car)
    n_val = int(n * 0.2)
    n_test = int(n * 0.2)
    n_train = n - n_val - n_test

    np.random.seed(seed)
    idx = np.arange(n)
    np.random.shuffle(idx)

    train = df_car.iloc[idx[:n_train]]
    val = df_car.iloc[idx[n_train:n_train + n_val]]
    test = df_car.iloc[idx[n_train + n_val:]]
    return train, val, test


def prepare_X(df_car, fill=0):
    df_car_num = df_car[features]
    df_car_num = df_car_num.fillna(fill)
    return df_car_num.values


def train_linreg(X, y, r=0.0):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    XTX = X.T @ X
    reg = r * np.eye(XTX.shape[0])
    XTX = XTX + reg
    XTX_inv = np.linalg.inv(XTX)

    w = XTX_inv @ X.T @ y
    return w[0], w[1:]


def rmse(y, y_pred):
    error = y - y_pred
    mse = (error ** 2).mean()
    return np.sqrt(mse)


# eda
print(df_car[target].describe())
print('skew:', df_car[target].skew())

#1. Missing values (1 point)
print(df_car.isnull().sum())

#2. Median for horsepower (1 point)
print(df_car['horsepower'].median())

# split data seed 42
train, val, test = split_data(df_car, seed=42)
y_train = train[target].values
y_val = val[target].values

#3. Filling NAs (1 point)
hp_mean = train['horsepower'].mean()

# fill with 0
X_train = prepare_X(train, fill=0)
X_val = prepare_X(val, fill=0)
w0, w = train_linreg(X_train, y_train)
pred = w0 + X_val @ w
print('fill 0:', round(rmse(y_val, pred), 3))

# fill with mean
X_train = prepare_X(train, fill=hp_mean)
X_val = prepare_X(val, fill=hp_mean)
w0, w = train_linreg(X_train, y_train)
pred = w0 + X_val @ w
print('fill mean:', round(rmse(y_val, pred), 3))

#4. Best regularization (1 point)
X_train = prepare_X(train, fill=0)
X_val = prepare_X(val, fill=0)

for r in [0, 0.01, 0.1, 1, 5, 10, 100]:
    w0, w = train_linreg(X_train, y_train, r=r)
    pred = w0 + X_val @ w
    score = round(rmse(y_val, pred), 4)
    print(f'r={r}: {score}')

#5. RMSE standard deviation (1 point)
scores = []
for s in range(10):
    tr, va, _ = split_data(df_car, s)
    X_tr = prepare_X(tr, fill=0)
    X_va = prepare_X(va, fill=0)
    
    y_tr = tr[target].values
    y_va = va[target].values
    
    w0, w = train_linreg(X_tr, y_tr)
    pred = w0 + X_va @ w
    scores.append(rmse(y_va, pred))

print('std:', round(np.std(scores), 3))

#6. Evaluation on test (1 point)
tr, va, te = split_data(df_car, seed=9)
full = pd.concat([tr, va])

X_full = prepare_X(full, fill=0)
y_full = full[target].values
X_te = prepare_X(te, fill=0)
y_te = te[target].values

w0, w = train_linreg(X_full, y_full, r=0.001)
pred = w0 + X_te @ w
print('test rmse:', round(rmse(y_te, pred), 3))