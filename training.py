import pandas as pd
import joblib
import os

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

os.makedirs('models', exist_ok=True)

df = pd.read_csv(
    'data/cleaned_car_price.csv'
)

x = df.drop(
    'selling_price',
    axis=1
)

y = df['selling_price']

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

# ---------------- MODEL ----------------

model = RandomForestRegressor(
    random_state=42
)

# ---------------- HYPERPARAMETER TUNING ----------------

params = {

    'n_estimators': [100, 200],

    'max_depth': [5, 10, 15],

    'min_samples_split': [2, 5],

    'min_samples_leaf': [1, 2]
}

grid = GridSearchCV(

    estimator=model,

    param_grid=params,

    cv=3,

    scoring='r2',

    n_jobs=-1
)

grid.fit(
    x_train,
    y_train
)

best_model = grid.best_estimator_

# ---------------- PREDICTIONS ----------------

y_pred = best_model.predict(x_test)

# ---------------- METRICS ----------------

mae = mean_absolute_error(
    y_test,
    y_pred
)

mse = mean_squared_error(
    y_test,
    y_pred
)

rmse = mse ** 0.5

r2 = r2_score(
    y_test,
    y_pred
)

print('\nBest Parameters\n')
print(grid.best_params_)

print('\nModel Performance\n')

print('MAE :', mae)

print('MSE :', mse)

print('RMSE :', rmse)

print('R2 Score :', r2)

# ---------------- SAVE MODEL ----------------

joblib.dump(
    best_model,
    'models/random_forest_model.pkl'
)

joblib.dump(
    x.columns.tolist(),
    'models/features.pkl'
)

print('\nModel Saved Successfully')