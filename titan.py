import numpy as np
import pandas as pd
from sklearn import tree
from sklearn import ensemble
from sklearn import cross_validation
import  csv

df = pd.read_csv('train.csv', header = 0)
df = df.drop(['Ticket','Cabin'], axis=1)

df['Sex'] = df['Sex'].map({'female' : 1, 'male' : 0}).astype(int)

df.loc[(df.Fare.isnull()) & (df.Pclass==1),'Fare']=np.median(df[df['Pclass']==1]['Fare'].dropna())
df.loc[(df.Fare.isnull()) & (df.Pclass==2),'Fare']=np.median(df[df['Pclass']==2]['Fare'].dropna())
df.loc[(df.Fare.isnull()) & (df.Pclass==3),'Fare']=np.median(df[df['Pclass']==3]['Fare'].dropna())

#process Pclass

df['Pclass'][df.Pclass.isnull()] = df['Pclass'].median()
age_df = df[['Age','Fare','Parch','SibSp','Pclass','Sex']]
known_age = age_df[df.Age.notnull()]
unknown_age = age_df[df.Age.isnull()]
y = known_age.values[:,0]
x = known_age.values[:,1:]
rfr = ensemble.RandomForestClassifier(n_estimators=2000,n_jobs=1)
rfr.fit(x,y)
predictedAges = rfr.predict(unknown_age.values[:,1:])
df['Age'][df.Age.isnull()] = predictedAges


x = df.values
x = np.delete(x, 1, axis=1)
y = df['Survived'].values
x_train,x_test,y_train,y_test = cross_validation.train_test_split(x, y, test_size = 0.25,\
                                                                   random_state = 33)

dt = ensemble.RandomForestClassifier(n_estimators=1000)
dt.fit(x_train, y_train)
print dt.score(x_test, y_test)

test = pd.read_csv('test.csv', header=0)
tf = test.drop(['Ticket', 'Name', 'Cabin'],axis=1)

tf['Sex'] = tf['Sex'].map({'female' : 1, 'male' : 0}).astype(int)

tf.loc[(tf.Fare.isnull()) & (tf.Pclass==1),'Fare']=np.median(tf[tf['Pclass']==1]['Fare'].dropna())
tf.loc[(tf.Fare.isnull()) & (tf.Pclass==2),'Fare']=np.median(tf[tf['Pclass']==2]['Fare'].dropna())
tf.loc[(tf.Fare.isnull()) & (tf.Pclass==3),'Fare']=np.median(tf[tf['Pclass']==3]['Fare'].dropna())

age_df = tf[['Age','Fare','Parch','Sibsp','Pclass','Sex']]
known_age = age_df[tf.Age.notnull()]
unknown_age = age_df[tf.Age.isnull()]
y = known_age.values[:,0]
x = known_age.values[:,1:]

rfr = ensemble.RandomForestClassifier(n_estimators=2000,n_jobs=1)
rfr.fit(x,y)
predictedAges = rfr.predict(unknown_age.values[:,1:])
tf['Age'][tf.Age.isnull()] = predictedAges

predicts = dt.predict(tf)
ids = tf['PassengerId'].values
prediction_file = open('dt_submission.csv','wb')
open_file = csv.writer(prediction_file)
open_file.writerow(["PassengerId","Survived"])
open_file.writerows(zip(ids, predicts))
prediction_file.close()