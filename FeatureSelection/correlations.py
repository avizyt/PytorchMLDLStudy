import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing

# fetch a regression datasets
data = fetch_california_housing()
X = data['data']
col_names = data['feature_names']
y = data['target']

# convert to pandas dataframe
df = pd.DataFrame(X, columns=col_names)

# introduce a highly correlated column
df.loc[:, "MedInc_Sqrt"] = df.MedInc.apply(np.sqrt)

# get correlation matrix (pearson)
print(df.corr())

