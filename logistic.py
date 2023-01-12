import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix,accuracy_score


df=pd.read_csv("static/heartDataSet.csv")

def transform_label(value):
    if value>=1.0:
        return 1
    else:
        return 0
    
df['target']=df.target.apply(transform_label)
X=df.drop('target',axis=1).copy()
Y=df['target'].copy()

X_encoded = pd.get_dummies(X, columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])

from sklearn.preprocessing import StandardScaler
standardScaler = StandardScaler()
columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
X_encoded[columns_to_scale] = standardScaler.fit_transform(X_encoded[columns_to_scale])

X_train,X_test,Y_train,Y_test=train_test_split(X_encoded,Y,random_state=0)

class LogisticRegression():
    
    def __init__(self, learning_rate=0.001, no_of_iterations=1000):
        self.lr = learning_rate
        self.no_of_iterations = no_of_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.no_of_iterations):
            Z = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(Z)

            
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            
            
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        Z = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(Z)
        y_predicted_class = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_class)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def findAns3(self, x):
        Z = np.dot(x, self.weights) + self.bias
        
        y_predicted = self._sigmoid(Z)
        if y_predicted>0.5:
            y_predicted = 1
        else:
            y_predicted = 0
        return y_predicted

lr=LogisticRegression(learning_rate=0.001,no_of_iterations=50000)
lr.fit(X_train,Y_train)

#ans = lr.findAns3(X_test.iloc[0,:])
#print(ans)