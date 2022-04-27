import re
import numpy as np
import pandas as pd
from stemmer import Stemmer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
data_temp = pd.read_excel ("Gujarati_Training.xlsx")
X = data_temp.iloc[:, :-2].values
Y = data_temp['prognosis- પૂર્વસૂચન'].values.reshape(-1,1)
    #print(X,Y)
    
y = labelencoder.fit_transform(Y)
#X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=1)
#from sklearn.ensemble import RandomForestClassifier
#classifier = RandomForestClassifier(n_estimators=100)
#classifier.fit(X_train, y_train)
#pr1= classifier.predict(X_test)
#accuracy = accuracy_score(y_test, pr1)
#print(accuracy)

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
classifier = LogisticRegression(random_state=0)
#X_train = scaler.transform(X_train)
#X_test = scaler.transform(X_test)
classifier.fit(X,y)
pr2 = classifier.predict(X)
accuracy = accuracy_score(y, pr2)
print(accuracy)