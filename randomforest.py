import numpy as np
import pandas as pd
import random

df = pd.read_csv("static/heartDataSet.csv")
df.head(10)


def transform_label(value):
    if value >= 1.0:
        return 1
    else:
        return 0

df["target"] = df.target.apply(transform_label)
df.head(10)




def giniValue(y):
    distCounts = np.unique(y)
    sum = 0
    for ct in distCounts:
        probability = len(y[y == ct]) / len(y)
        sum += probability*probability
    return 1 - sum

def infoGain(parent,left, right):
    weightL = len(left)/len(parent)
    weightR = len(right)/len(parent)
    gain = giniValue(parent) - (weightL*giniValue(left) + weightR*giniValue(right))
    return gain

def split(dataset, feature_index, threshold):
        dataLeft = []
        for row in dataset:
            if row[feature_index]<=threshold:
                dataLeft.append(row)
        
        dataRight = []
        for row in dataset:
            if row[feature_index]>threshold:
                dataRight.append(row)
        
        dataLeft = np.array(dataLeft)
        dataRight = np.array(dataRight)
                
       
        return dataLeft, dataRight

class Node:
    def __init__(self, _colIdx = None, _threshold = None, _left = None, _right = None, _infoGain = None, _value = None):
        self.colIdx = _colIdx
        self.threshold = _threshold
        self.left = _left
        self.right = _right
        self.infoGain = _infoGain

        self.value = _value

class DecisionTree:
    def __init__(self,_minSamples = 10, _maxDepth=5):
        self.root = None
        self.minSamples = _minSamples
        self.maxDepth = _maxDepth
    
    def mostCommonLabel(self, Y):        
        Y = list(Y)
        return max(Y, key=Y.count)

    def buildTree(self,dataset,depth=0):
        data = dataset[:, :-1]
        numSamples, numCols = np.shape(data)

        if numSamples>=self.minSamples and depth<=self.maxDepth:
            
            best_split = self.findBestSplit(dataset, numSamples, numCols)
            
            if best_split["infoGain"]>0:
                left_subtree = self.buildTree(best_split["dataLeft"], depth+1)
                right_subtree = self.buildTree(best_split["dataRight"], depth+1)

                return Node(best_split["colNo"], best_split["threshold"], left_subtree, right_subtree, best_split["infoGain"])

        
        leaf = self.mostCommonLabel(dataset[:, -1])
        return Node(_value = leaf)


    def findBestSplit(self, dataset, numSamples, numCols):
        bestSplit = {}
        maxInfoGain = -float("inf")
        
        for col in range(numCols):
            feature_values = dataset[:, col]
            possibleThresholds = np.unique(feature_values)
           
            for threshold in possibleThresholds:             
                dataLeft, dataRight = split(dataset, col, threshold)
                
                if len(dataLeft)>0 and len(dataRight)>0:
                    parentRes, leftRes, rightRes = dataset[:, -1], dataLeft[:, -1], dataRight[:, -1]
                    
                    curr_info_gain = infoGain(parentRes, leftRes, rightRes)
                    
                    if curr_info_gain>maxInfoGain:
                        bestSplit["colNo"] = col
                        bestSplit["threshold"] = threshold
                        bestSplit["dataLeft"] = dataLeft
                        bestSplit["dataRight"] = dataRight
                        bestSplit["infoGain"] = curr_info_gain
                        maxInfoGain = curr_info_gain
        return bestSplit
    

    def train(self,X,Y):
        dataset = np.concatenate((X, Y), axis=1)
        self.root = self.buildTree(dataset)

    def getAns(self, x, node):    
        if node.value!=None: 
            return node.value

        val = x[node.colIdx]
        if val<=node.threshold:
            return self.getAns(x, node.left)
        else:
            return self.getAns(x, node.right)

    def predict(self, X):
        
        predictions = []
        for x in X:
            ans = self.getAns(x,self.root)
            predictions.append(ans)

        return predictions

from sklearn.model_selection import train_test_split
# forest = []
# qrow = []
# ct = 5
# while(ct):
#     bootstrap_indices = np.random.randint(low=0, high=len(df), size=75)
#     df_bootstrapped = df.iloc[bootstrap_indices]
    
#     _, n_columns = df_bootstrapped.shape
#     column_indices = list(range(n_columns - 1))
    
#     column_indices = random.sample(population=column_indices, k=5)
#     df2 = df_bootstrapped.iloc[:, column_indices]
#     X = df2.iloc[:, :-1].values
#     Y = df_bootstrapped.iloc[:, -1].values.reshape(-1,1)

#     X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=10)
#     classifier = DecisionTree(_minSamples=2, _maxDepth=3)
#     classifier.train(X_train,Y_train)
    
#     a = classifier.getAns(qrow, classifier.root)
#     forest.append(a)
#     ct = ct - 1
X = df.iloc[:, :-1].values
Y = df.iloc[:, -1].values.reshape(-1,1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=10)
classifier = DecisionTree(_minSamples=2, _maxDepth=3)
classifier.train(X_train,Y_train)

# Y_pred = classifier.predict(X_test) 
# from sklearn.metrics import accuracy_score
# print(accuracy_score(Y_test, Y_pred))