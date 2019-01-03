#!/usr/bin/env python
import pandas as pd
import time

from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals import joblib
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

train_data = pd.read_csv('output_train.csv')
test_data=pd.read_csv('output_test.csv')
#train_data, test_data = train_test_split(all_data, test_size=1/3, random_state=10)
#if you use random_state,no matter its value,every time you run this code,the
#training data & test data would be the same

# establish the training & test data set
X_train = train_data.iloc[:,1:].values
sel = VarianceThreshold(threshold=(0.8 * (1 - 0.8)))
sel.fit_transform(X_train)
y_train = train_data.iloc[:,0].values
X_test = test_data.iloc[:,1:].values
y_test = test_data.iloc[:,0].values
print('In total {} dimension features.'.format(X_train.shape[1]))
#train_data.iloc[[i]]           #to select a line

parameters={'n_estimators':[50,60,90,120,150]}

results_df = pd.DataFrame(columns=['Iteration Times','Accuracy (%)', 'Time (s)'])

for i in list(parameters.values())[0]:
    clf=AdaBoostClassifier(DecisionTreeClassifier(max_depth=9, min_samples_leaf=1),
                           n_estimators=i,
                           algorithm='SAMME',
                           learning_rate=1)
    start=time.time()
    clf.fit(X_train,y_train)
    end=time.time()
    duration=end-start
    results_df.loc[i, 'Iteration Times'] = i
    results_df.loc[i, 'Accuracy (%)'] = clf.score(X_test,y_test)*100
    results_df.loc[i, 'Time (s)'] = duration
    print('time consumption:{:.3f}s'.format(duration))
    print('model accuracy:{:3f}%'.format(clf.score(X_test,y_test)*100))
    joblib.dump(clf, "train_model{:1d}.m".format(i))
    # save the model,with different name
results_df.to_csv('estimators_outcome.csv',index=0)  #save the outcome,no index

#clf = joblib.load("train_model.m")
# y_pred=clf.predict(X_test)
# print('accuracy:{:3f} %'.format(clf.score(X_test,y_test)))