import numpy as np
from collections import Counter
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df=pd.read_csv("static/heartDataSet.csv")




def transform_label(value):
    if value>=1.0:
        return 1
    else:
        return 0

df["target"]=df.target.apply(transform_label)

df = pd.get_dummies(df, columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])
# from sklearn.preprocessing import StandardScaler
# standardScaler = StandardScaler()
# columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
# df[columns_to_scale] = standardScaler.fit_transform(df[columns_to_scale])


import numpy as np
from collections import Counter

def euclidean_distance(x1, x2):
    distance = np.sqrt(np.sum((x1-x2)**2))
    return distance

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        
        predictions = [self._predict(x) for x in X]
        return predictions

    def _predict(self, x):
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
    
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        most_common = Counter(k_nearest_labels).most_common()
        return most_common[0][0]
    def getAns(self, x):
        ans = self._predict(x)
        return ans
    
y = df['target'].values
X = df.drop(['target'], axis = 1).values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)
clf = KNN(k=5)
clf.fit(X_train,y_train)
# qrow = [63.0,1.0,1.0,145.0,233.0,1.0,2.0,150.0,0.0,2.3,3.0,0.0,6.0]
# qrow = np.array(qrow)
# qrow = qrow.astype(float)
# ans = clf.getAns(qrow)
# print(ans)
