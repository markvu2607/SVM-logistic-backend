# solution by sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegression

data = pd.read_csv('btn3-training.csv').values
X = data[:, :-1]
y = data[:, -1]

modelLogistic = LogisticRegression(random_state=0)
modelLogistic.fit(X, y)
