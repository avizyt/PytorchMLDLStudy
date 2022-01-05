# import data analysis library
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score

train_path = '../data/superconduct/train.csv'

data = pd.read_csv(train_path)

# print(data.head())
# print(data.info())
# print(data.columns)
# print(data.isnull().sum())

X = data.drop(['critical_temp'], axis=1)
y = data['critical_temp']

# print(X.shape)
# print(y.shape)

# train , test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# print(X_train.shape)
# print(X_test.shape)
# print(y_train.shape)
# print(y_test.shape)

rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train)

y_preds = rf_model.predict(X_test)


print(f"Model R2 score: {r2_score(y_test, y_preds)}")
print(f"Model MSE: {mean_squared_error(y_test, y_preds)}")








