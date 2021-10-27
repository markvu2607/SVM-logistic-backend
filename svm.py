# solution by sklearn
from sklearn.svm import SVC
import pandas as pd

data = pd.read_csv('btn3-training.csv').values
X = data[:, :-1]
y = data[:, -1]

modelSVC = SVC(kernel='linear', C=1e5)
modelSVC.fit(X, y)
