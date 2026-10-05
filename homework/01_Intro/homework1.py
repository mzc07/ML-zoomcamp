# Archive: Homework1.py 
# Author: @mzc07
# Description: Introduction to ML using a car_fuel_efficiency dataset

import pandas as pd
import numpy as np
import statistics as stat

# 0. Load dataset
car_fuel_path = r'data/car_fuel_efficiency_2026.csv'
df_car = pd.read_csv(car_fuel_path)

# 1. Pandas version (1 point)
print('Pandas version:')
print(pd.__version__)

# 2. Records count (1 point)
print('\nRecord counts')
print(len(df_car))

# 3. Fuel types (1 point)
print('\nFuel types')
print(df_car['fuel_type'].nunique())

# 4. Missing values (1 point)
print('\nMissing Values')
print(df_car.isnull().any())
print(df_car.isnull().sum())

# 5. Max fuel efficiency (1 point)
print('\nMax fuel efficiency')
print(np.max(df_car['fuel_efficiency_mpg']))

# 6. Median value of horsepower (1 point)
print('\nFirst median value')
print(f"First value: {df_car['horsepower'].median()}")
mode_hp = stat.mode(df_car['horsepower'])
df_car['horsepower'] = df_car['horsepower'].fillna(mode_hp)
print(f"Second value: {df_car['horsepower'].median()}")

# 7. Sum of weights (1 point)
asian_cars = df_car[df_car['origin'] == "Asia"].copy()
asian_sample = asian_cars[['vehicle_weight', 'model_year']].head(7)
X = asian_sample.to_numpy()
XTX = X.T @ X
XTX_inv = np.linalg.inv(XTX)
y = np.array([1100, 1300, 800, 900, 1000, 1100, 1200])
w = XTX_inv @ X.T @ y
print(w)
print(w.sum())