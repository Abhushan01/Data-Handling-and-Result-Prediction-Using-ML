#cleaning diabetes dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import  warnings
warnings.filterwarnings("ignore")

df=pd.read_csv(r'https://github.com/Abhushan01/Data-Handling-and-Result-Prediction-Using-ML/blob/master/diabetes.csv')
print('Data Cleaning:')
print(df.columns);print('\n')
print(df.info());print('\n')
print(df.isnull().sum()) ;print('\n')
#dropping useless columns
dropping=['name','body','home']
df.drop(dropping, axis=1, inplace=True)
df.fillna(df.mean(), inplace=True)
df.iloc[:,0:6]=df.iloc[:,0:6].replace(0,np.NaN)
df.fillna(df.mean(),inplace=True)
print(df.info());print('\n')
print(df.isnull().sum());print('\n')
df.to_csv(r'https://github.com/Abhushan01/Data-Handling-and-Result-Prediction-Using-ML/blob/master/diabetes_cln.csv',index=False)
#above command saves the cleaned dataset as diabetes_cln

#data visualization
print("Data Visualization:")
df=pd.read_csv(r'https://github.com/Abhushan01/Data-Handling-and-Result-Prediction-Using-ML/blob/master/diabetes_cln.csv')
#grouping of data to plot bar graphs 
outcome_grouping=df.groupby('Outcome').mean()
outcome_grouping1=df.groupby('Outcome').count()
outcome_grouping1['Age'].plot.bar()
outcome_grouping['Age'].plot.bar()
#scatter plot matrix for the data
df.plot.scatter(x='SkinThickness', y='BMI');
df.plot.scatter(x='BloodPressure',y='Age');
from pandas.plotting import scatter_matrix
scatter_matrix(df,figsize=(15,15))
plt.show()
#boxplot on features
feature_names= ['BloodPressure','Age']
plt.suptitle('Box Plot')
df.boxplot(column=feature_names,figsize=(10,10))
plt.show()

#feature selection
print("Feature Selection:");print('\n')
#from pandas import read_csv
#Feature Importance with Extra Tree Classifier
print("Using Extra Tree Classifier:")
from sklearn.ensemble import ExtraTreesClassifier
#Load data
df=pd.read_csv(r'https://github.com/Abhushan01/Data-Handling-and-Result-Prediction-Using-ML/blob/master/diabetes_cln.csv')
print(df.head());print('\n')
X=df.iloc[:,0:-1].values
y=df.iloc[:,-1].values
#feature extraction
model=ExtraTreesClassifier()
model.fit(X,y)
print(df.columns);print('\n')
print(model.feature_importances_);print('\n')
l=[]
for i in range(len(df.columns)-1):
    l.append((model.feature_importances_[i],df.columns[i]))
print(l);print('\n')
l.sort(reverse=True)
print(l);print('\n')
for i in range(4):
    print(l[i][1])
print('\n')    
#Using LinearRegression
print("Using Linear Regression:")
from sklearn.feature_selection import RFE
from sklearn import linear_model
print(df.head());print('\n')
X=df.iloc[:,0:-1].values
y=df.iloc[:,-1].values
regr=linear_model.LinearRegression()
estimator=linear_model.LinearRegression()
#feature ranking with recursive feature elemination
selector=RFE(estimator,4) #4 is the number of features to select
selector.fit(X,y)
print(selector.n_features_);print('\n')
print(selector.support_);print('\n')
print(selector.ranking_);print('\n')
p=selector.transform(X)
q=selector.inverse_transform(p)
l=[]
for i in range(len(df.columns)-1):
    l.append((selector.ranking_[i],df.columns[i]))
print(l);print('\n')
l.sort()
print(l);print('\n')
for i in range(4):
    print(l[i][1])
print('\n')
#Using SelectKBest
print("Using SelectKBest:")
import numpy 
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
array=df.values
X=array[:,0:8]
Y=array[:,8]
bestK=SelectKBest(score_func=chi2,k=4)
fitKbest=bestK.fit(X,Y)
numpy.set_printoptions(precision=3) #summarize scores
print(fitKbest.scores_);print('\n')
l=[]
for i in range(len(df.columns)-1):
    l.append((fitKbest.scores_[i],df.columns[i]))
print(l);print('\n')
l.sort(reverse=True)
print(l);print('\n')
for i in range(4):
    print(l[i][1])
print(fitKbest.pvalues_);print('\n')
featTransformed=fitKbest.transform(X)
#summarized selected features
print(featTransformed[0:5,:]);print('\n')
featBack=fitKbest.inverse_transform(featTransformed)
print(featBack[0:5,:]);print('\n')

#implementing different classifiers and accuracy score bar graphs
print("Implementation of different classifiers:")
print(df.head())
from sklearn.model_selection import train_test_split
X=df.iloc[:,0:-1].values
y=df.iloc[:,-1].values
#split into train and test set
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3, random_state=0)
l=[ ]

#using knn classifier
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()
knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)
pred1=knn.predict(X_test)
from sklearn.metrics import accuracy_score
print("Accuracy score of test data using knn method:", accuracy_score(pred1,y_test));print('\n')
l.append(accuracy_score(y_test,pred1))

#using decision Tree
from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier()
classifier.fit(X_train,y_train)
pred2=classifier.predict(X_test)
from sklearn.metrics import accuracy_score
print("Accuracy score of test data using decision Tree method:", accuracy_score(pred2,y_test));print('\n')
l.append(accuracy_score(pred2,y_test))

#using logistic Regression
from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()
logreg.fit(X_train,y_train)
pred3=logreg.predict(X_test)
from sklearn.metrics import accuracy_score
print("Accuracy score of test data using logistic Regression method:", accuracy_score(pred3,y_test));print('\n')
l.append(accuracy_score(pred3,y_test))

#using support Vector machine (svm)
from sklearn import svm
svc=svm.SVC()
svc.fit(X_train,y_train)
pred4=svc.predict(X_test)
from sklearn.metrics import accuracy_score
print("Accuracy score of test data using SVM method:",accuracy_score(pred4,y_test));print('\n')
l.append(accuracy_score(pred4,y_test))

#using randomForest
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier()
model.fit(X_train,y_train)
pred5=model.predict(X_test)
from sklearn.metrics import accuracy_score
print("Accuracy score of test  data using random Forest method:", accuracy_score(pred5,y_test));print('\n')
l.append(accuracy_score(pred5,y_test))

#drawing bar graphs
a=np.arange(1,6)
plt.figure(figsize=(10,10))
plt.title("Bar graph")
plt.ylabel('Y axis')
plt.yticks(np.arange(0,1,0.05))
plt.xticks(np.arange(1,6),('knn','dt','Lr','svm','rF'))
plt.ylim=(0,1)
plt.xlabel('X axis')
plt.bar(a,l,color='red')
plt.show()

#confusion matrix
print("Confusion Matrix:")
print(df.head())
#machine learning classifiers
X=df.iloc[:,0:8].values
y=df.iloc[:,8].values
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)
y_pred=knn.predict(X_test)
#accuracy score
from sklearn.metrics  import accuracy_score
print('knn accuracy',accuracy_score(y_pred,y_test))
#Confusion Matrix
from sklearn.metrics import confusion_matrix
print("#===Confusion matrix")
label_lst=[0,   1]
print(confusion_matrix(y_test,y_pred, labels=label_lst))
