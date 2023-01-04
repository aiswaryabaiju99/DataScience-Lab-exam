import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import datasets
from sklearn.datasets import load_iris
import pandas as pd
#dataset=pd.read_csv('data.csv')
#x=dataset.iloc[:,: -1].values
#y=dataset.iloc[:, 1].values
LoadData=load_iris()
a=LoadData.data
b=LoadData.target

a_train,a_test,b_train,b_test=train_test_split(a,b,test_size=0.3,random_state=42)
knn=KNeighborsClassifier(n_neighbors=9)
#print(knn)
knn.fit(a_train,b_train)
c=knn.predict(a_test)
print(c)
acc=accuracy_score(b_test,c)
print('accuracy',acc)

